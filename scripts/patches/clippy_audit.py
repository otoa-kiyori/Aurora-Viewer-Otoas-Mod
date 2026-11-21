#!/usr/bin/env python3
"""Run clippy on the Linux build and report only what is new since the committed baseline.

Findings are keyed by (lint, file, normalised message), never by line number, so
moved code does not look new. The exit code is non-zero only for a new
`correctness` or `suspicious` finding (and, with --strict, a new
unwrap/expect/panic in the crates that parse external data). See README.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXIT_OK, EXIT_NEW, EXIT_TOOL = 0, 1, 2
BASELINE_NAME = "clippy-baseline.json"
BASELINE_VERSION = 1

# Clippy's JSON does not name a lint's group, so a pass that wants it runs
# `-A clippy::all -W clippy::<group>`: every clippy finding of that pass belongs
# to that group. The five groups together are the default lint set.
GROUPS = ("correctness", "suspicious", "style", "complexity", "perf")
GATING_LABEL = "correctness/suspicious"  # the quick mode runs both groups in one pass
DEFAULT_LABEL = "default"  # any other lint of the project's own configuration
RUSTC_GROUP = "rustc"
GATING_GROUPS = {"correctness", "suspicious", GATING_LABEL}

# --strict: AGENTS.md forbids unwrap()/expect() on external data (network,
# files, assets), so these crates are checked with the restriction lints.
EXTERNAL_DATA_CRATES = ("aurora-llsd", "aurora-msg", "aurora-prim", "aurora-assets", "aurora-net")
STRICT_GROUP = "restriction"
STRICT_LINTS = ("unwrap_used", "expect_used", "panic", "indexing_slicing")
# indexing_slicing is recorded in the baseline for the audit but never fails a run: it is too noisy.
STRICT_GATING = {"clippy::unwrap_used", "clippy::expect_used", "clippy::panic"}

# CI: cargo clippy --profile ci --workspace --all-targets --locked -- -D warnings -A dead-code
CARGO_BASE = ["clippy", "--profile", "ci", "--workspace", "--locked"]
NUMBER_RE = re.compile(r"\d+")


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Finding:
    lint: str  # "clippy::needless_return" or a rustc lint such as "unused_variables"
    file: str  # relative to the root, forward slashes
    line: int
    message: str
    group: str
    col: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.lint, self.file, normalise(self.message))


# ---------------------------------------------------------------- parsing


def normalise(message: str) -> str:
    """Drop the numbers: thresholds ("120/100") and counts change without the code changing."""
    return NUMBER_RE.sub("N", message).strip()


def relative_path(name: str, root: Path) -> str:
    path = Path(name)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            pass  # outside the tree: keep as is
    return path.as_posix()


def parse_messages(lines, root: Path, group: str | None = None) -> tuple[list[Finding], list[str]]:
    """Findings and compile errors of one `--message-format=json` stream.

    `group` forces the group of the clippy lints of a pass; rustc lints are
    always `rustc`. Identical (lint, file, line, column, message) reports of
    several targets (a lib and its test build) count once.
    """
    findings: dict[tuple, Finding] = {}
    errors: list[str] = []
    for raw in lines:
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("reason") != "compiler-message":
            continue
        msg = event.get("message") or {}
        level = msg.get("level")
        code = (msg.get("code") or {}).get("code")
        primary = next((s for s in msg.get("spans", []) if s.get("is_primary")), None)
        if level == "error" and not code:
            errors.append(msg.get("message", "?"))
            continue
        if level not in ("warning", "error") or not code or primary is None:
            continue  # "N warnings emitted" and friends
        lint = code
        is_clippy = lint.startswith("clippy::")
        fnd = Finding(
            lint=lint,
            file=relative_path(primary["file_name"], root),
            line=int(primary.get("line_start", 0)),
            message=msg.get("message", ""),
            group=(group or "clippy") if is_clippy else RUSTC_GROUP,
            col=int(primary.get("column_start", 0)),
        )
        findings.setdefault((lint, fnd.file, fnd.line, fnd.col, fnd.message), fnd)
    return sorted(findings.values(), key=lambda f: (f.file, f.line, f.lint)), errors


# ---------------------------------------------------------------- running clippy


def find_cargo() -> str:
    candidate = Path.home() / ".cargo" / "bin" / "cargo"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("cargo")
    if not found:
        raise ToolError("cargo not found (expected ~/.cargo/bin/cargo)")
    return found


def toolchain_env(cargo: str, root: Path) -> dict[str, str]:
    """Environment that makes `cargo clippy` use the toolchain's own clippy.

    cargo looks for `cargo-clippy` on PATH, so a distribution package earlier on
    PATH (Debian's is older than rustc) silently wins over the rustup one.
    Putting the sysroot first makes the run match CI.
    """
    env = dict(os.environ)
    rustc = str(Path(cargo).with_name("rustc"))
    try:
        run = subprocess.run([rustc, "--print", "sysroot"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return env
    sysroot_bin = Path(run.stdout.strip()) / "bin"
    if sysroot_bin.is_dir():
        env["PATH"] = f"{sysroot_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def clippy_version(cargo: str, root: Path, env: dict[str, str]) -> str:
    try:
        run = subprocess.run(
            [cargo, "clippy", "--version"], cwd=root, env=env, capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    return run.stdout.strip() or "unknown"


def run_clippy(cargo: str, root: Path, env: dict[str, str], extra_cargo: list[str], clippy_args: list[str], lock=None):
    """One clippy invocation; returns (stdout lines, exit code). `lock` is a file taken with flock(1) first."""
    cmd = [
        *(["flock", lock] if lock else []),
        cargo,
        *CARGO_BASE,
        *extra_cargo,
        "--message-format=json",
        "--",
        *clippy_args,
    ]
    try:
        run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ToolError(f"cannot run {cargo}: {exc}") from exc
    if run.returncode != 0 and '"reason":"compiler-message"' not in run.stdout:
        raise ToolError(f"cargo clippy failed before producing diagnostics:\n{run.stderr[-2000:]}")
    return run.stdout.splitlines(), run.returncode


def passes(groups: bool, allow_dead: list[str]) -> list[tuple[str, list[str]]]:
    """(group label, clippy arguments) of the all-targets passes.

    Quick mode: the gating groups first (so a lint found by both passes keeps
    that label), then the project's own configuration, which is what CI runs.
    `groups` splits it into one pass per group, for a full audit.
    """
    if groups:
        return [(g, ["-A", "clippy::all", f"-Wclippy::{g}", *allow_dead]) for g in GROUPS]
    return [
        (GATING_LABEL, ["-A", "clippy::all", "-Wclippy::correctness", "-Wclippy::suspicious", *allow_dead]),
        (DEFAULT_LABEL, list(allow_dead)),
    ]


def collect(cargo: str, root: Path, strict: bool, dead_code: bool, groups: bool = False, runner=None, env=None):
    """Run every pass; returns (findings, errors)."""
    runner = runner or run_clippy
    env = env if env is not None else toolchain_env(cargo, root)
    allow_dead = [] if dead_code else ["-A", "dead-code"]
    merged: dict[tuple, Finding] = {}
    errors: list[str] = []

    def add(found: list[Finding]) -> None:
        for f in found:
            merged.setdefault((f.lint, f.file, f.line, f.col, f.message), f)

    for label, args in passes(groups, allow_dead):
        print(f"clippy pass: {label}", file=sys.stderr)
        found, errs = parse_messages(runner(cargo, root, env, ["--all-targets"], args)[0], root, label)
        add(found)
        errors += errs
    if strict:
        print("clippy pass: external-data restriction lints", file=sys.stderr)
        # Library and binary code only: tests may unwrap (cfg(test) is not compiled here).
        args = ["-A", "clippy::all", *(f"-Wclippy::{lint}" for lint in STRICT_LINTS), *allow_dead]
        found, errs = parse_messages(runner(cargo, root, env, ["--lib", "--bins"], args)[0], root, STRICT_GROUP)
        add([f for f in found if f.lint.startswith("clippy::") and is_external_data(f.file)])
        errors += errs
    allowed = configured_allows(root)
    kept = [f for f in merged.values() if f.lint.removeprefix("clippy::") not in allowed_for(allowed, f.file)]
    return sorted(kept, key=lambda f: (f.file, f.line, f.lint)), sorted(set(errors))


def _allowed_lints(table: dict) -> set[str]:
    """Names of the lints a `[lints.clippy]`-style table sets to "allow"."""
    out = set()
    for name, value in table.items():
        level = value.get("level") if isinstance(value, dict) else value
        if level == "allow":
            out.add(name)
    return out


def configured_allows(root: Path) -> dict[str, set[str]]:
    """Lints that Cargo.toml allows on purpose, per crate directory ("" = the whole workspace).

    A group pass (`-W clippy::complexity`) would switch them back on, because
    command-line levels win over the `[lints]` table; the project decided
    otherwise (too_many_arguments, type_complexity), so they are dropped.
    """
    result: dict[str, set[str]] = {"": set()}

    def read(manifest: Path) -> dict:
        try:
            return tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    result[""] = _allowed_lints(read(root / "Cargo.toml").get("workspace", {}).get("lints", {}).get("clippy", {}))
    for manifest in sorted((root / "crates").glob("*/Cargo.toml")):
        local = read(manifest).get("lints", {}).get("clippy", {})
        result[manifest.parent.name] = _allowed_lints(local) if isinstance(local, dict) else set()
    return result


def allowed_for(allowed: dict[str, set[str]], file: str) -> set[str]:
    parts = Path(file).parts
    crate = parts[1] if len(parts) > 1 and parts[0] == "crates" else ""
    return allowed[""] | allowed.get(crate, set())


def is_external_data(file: str) -> bool:
    parts = Path(file).parts
    # build.rs runs at build time on the repository's own files, not on external data.
    return len(parts) > 2 and parts[0] == "crates" and parts[1] in EXTERNAL_DATA_CRATES and parts[2:] != ("build.rs",)


# ---------------------------------------------------------------- baseline


def counts(findings: list[Finding]) -> Counter:
    return Counter(f.key for f in findings)


def baseline_document(findings: list[Finding], strict: bool, version: str) -> dict:
    group_of = {}
    for f in findings:
        group_of.setdefault(f.key, f.group)  # informational: the diff ignores it
    entries = [
        {"lint": lint, "file": file, "message": message, "group": group_of[(lint, file, message)], "count": n}
        for (lint, file, message), n in sorted(counts(findings).items())
    ]
    return {"version": BASELINE_VERSION, "clippy": version, "strict": strict, "findings": entries}


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"unreadable baseline {path}: {exc}") from exc
    if doc.get("version") != BASELINE_VERSION:
        raise ToolError(f"baseline {path} has version {doc.get('version')}, expected {BASELINE_VERSION}")
    return doc


def new_findings(findings: list[Finding], baseline: dict, include_strict: bool) -> list[Finding]:
    """Findings above the baseline count of their key (line numbers ignored)."""
    known = Counter()
    for e in baseline["findings"]:
        known[(e["lint"], e["file"], e["message"])] += e["count"]
    seen = Counter()
    out = []
    for f in findings:
        if f.group == STRICT_GROUP and not include_strict:
            continue
        seen[f.key] += 1
        if seen[f.key] > known[f.key]:
            out.append(f)
    return out


def gating(f: Finding, strict: bool) -> bool:
    if f.group in GATING_GROUPS:
        return True
    return strict and f.group == STRICT_GROUP and f.lint in STRICT_GATING


# ---------------------------------------------------------------- report


def summarise(findings: list[Finding]) -> str:
    by_group: dict[str, Counter] = {}
    for f in findings:
        by_group.setdefault(f.group, Counter())[f.lint] += 1
    lines = []
    for group in sorted(by_group, key=lambda g: (-sum(by_group[g].values()), g)):
        lines.append(f"{group:<12} {sum(by_group[group].values()):>5}")
        for lint, n in sorted(by_group[group].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {n:>5}  {lint}")
    return "\n".join(lines) if lines else "no findings"


def format_finding(f: Finding, strict: bool) -> str:
    flag = "FAIL" if gating(f, strict) else "new "
    return f"{flag}  {f.group:<11} {f.lint}  {f.file}:{f.line}  {f.message}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="tree to check (default: this repository)")
    ap.add_argument("--update-baseline", action="store_true", help="write the current findings as the baseline")
    ap.add_argument(
        "--strict", action="store_true", help="also fail on a new unwrap/expect/panic in the external-data crates"
    )
    ap.add_argument("--baseline", help=f"baseline file (default: {BASELINE_NAME} next to this script)")
    ap.add_argument("--dead-code", action="store_true", help="keep dead_code findings (CI allows them on Linux)")
    ap.add_argument("--lock", metavar="FILE", help="take this flock(1) file around every cargo run (shared machines)")
    ap.add_argument("--jobs", type=int, metavar="N", help="set CARGO_BUILD_JOBS")
    ap.add_argument("--groups", action="store_true", help="one pass per clippy group (slower; exact group names)")
    ap.add_argument("--summary", action="store_true", help="also print the totals by group and lint")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here.parents[1]
    baseline_path = Path(args.baseline) if args.baseline else here / BASELINE_NAME
    try:
        if not (root / "Cargo.toml").is_file():
            raise ToolError(f"{root} has no Cargo.toml")
        cargo = find_cargo()
        env = toolchain_env(cargo, root)
        runner = functools.partial(run_clippy, lock=args.lock) if args.lock else None
        if args.jobs:
            env["CARGO_BUILD_JOBS"] = str(args.jobs)
        version = clippy_version(cargo, root, env)
        findings, errors = collect(cargo, root, args.strict, args.dead_code, args.groups, runner=runner, env=env)
        if errors:
            raise ToolError("the tree does not compile:\n  " + "\n  ".join(errors[:10]))
        baseline = load_baseline(baseline_path)
    except ToolError as exc:
        print(f"clippy_audit: {exc}", file=sys.stderr)
        return EXIT_TOOL

    print(f"{version}; {len(findings)} findings")
    if args.summary or args.update_baseline:
        print(summarise(findings))
    if args.update_baseline:
        if baseline and baseline.get("strict") and not args.strict:
            print("note: the old baseline had --strict findings; rerun with --strict to keep them", file=sys.stderr)
        doc = baseline_document(findings, args.strict, version)
        baseline_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"baseline written: {baseline_path} ({len(doc['findings'])} entries)")
        return EXIT_OK
    if baseline is None:
        print(f"no baseline at {baseline_path}: run with --update-baseline first", file=sys.stderr)
        return EXIT_TOOL
    if args.strict and not baseline.get("strict"):
        print("the baseline has no --strict findings: run --update-baseline --strict first", file=sys.stderr)
        return EXIT_TOOL
    if baseline.get("clippy") != version:
        print(f"note: baseline made with {baseline.get('clippy')}, new lints may appear after a clippy upgrade")

    fresh = new_findings(findings, baseline, args.strict)
    for f in fresh:
        print(format_finding(f, args.strict))
    failing = [f for f in fresh if gating(f, args.strict)]
    print(f"{len(fresh)} new, {len(failing)} failing")
    return EXIT_NEW if failing else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

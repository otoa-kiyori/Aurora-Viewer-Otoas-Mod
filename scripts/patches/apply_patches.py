#!/usr/bin/env python3
"""Apply the semantic local patches of scripts/patches/*.toml to a build tree.

Each patch is located by pattern or function name when it is applied, so it
survives upstream moving lines around, and it fails loudly (DRIFT) when its
target changed or disappeared. See README.md for the manifest format.

Ops: `regex` and `literal` rewrite matched text, `replace_fn_body` and
`insert_fn_start` edit a function found by name, and `add_file` brings a
whole new file (kept under the patches dir, e.g. `files/`) into the tree.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

EXIT_OK, EXIT_DRIFT, EXIT_MANIFEST, EXIT_GUARD = 0, 1, 2, 3
# Build output and scratch space are never patch targets, even under a broad glob.
IGNORED_DIRS = {"target", ".git", ".scratch"}
ID_RE = re.compile(r"[a-z0-9-]+")
FN_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ManifestError(Exception):
    pass


class Drift(Exception):
    pass


# ---------------------------------------------------------------- manifests

COMMON_REQUIRED = {"id", "why", "paths", "op"}
COMMON_OPTIONAL = {"required", "enabled"}
OPS = {  # op -> (required keys, optional keys)
    "regex": ({"pattern", "replace"}, {"expect", "dotall", "fn"}),
    "literal": ({"find", "replace"}, {"expect", "fn"}),
    "replace_fn_body": ({"fn", "body"}, {"impl", "expect_sig"}),
    "insert_fn_start": ({"fn", "body"}, {"impl", "expect_sig"}),
    "add_file": ({"source"}, set()),
}
GUARD_REQUIRED = {"id", "why", "paths", "pattern", "level"}
GUARD_OPTIONAL = {"allow"}
FIELD_TYPES = {
    "id": str, "why": str, "op": str, "required": bool, "enabled": bool,
    "pattern": str, "replace": str, "expect": int, "dotall": bool, "fn": str,
    "find": str, "body": str, "impl": str, "expect_sig": str, "level": str, "source": str,
}  # fmt: skip


@dataclass
class Patch:
    id: str
    why: str
    paths: list[str]
    op: str
    required: bool
    enabled: bool
    source: str
    opts: dict


@dataclass
class Guard:
    id: str
    why: str
    paths: list[str]
    pattern: re.Pattern
    allow: list[re.Pattern]
    level: str
    source: str


def marker(patch_id: str) -> str:
    return f"// aurora-local-patch: {patch_id}"


def marker_re(patch_id: str) -> re.Pattern:
    # The lookahead keeps id "a" from matching the marker of id "a-b".
    # `#` is the comment of TOML targets such as Cargo.lock (`//` would not parse there).
    return re.compile(r"(?://|#)[ \t]*aurora-local-patch:[ \t]*" + re.escape(patch_id) + r"(?![a-z0-9-])")


def _check_table(table: dict, where: str, required: set, optional: set) -> None:
    unknown, missing = set(table) - required - optional, required - set(table)
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing:
        raise ManifestError(f"{where}: missing key(s) {sorted(missing)}")
    for key, value in table.items():
        if key in ("paths", "allow"):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ManifestError(f"{where}: '{key}' must be a list of strings")
        elif not isinstance(value, FIELD_TYPES[key]) or (FIELD_TYPES[key] is int and isinstance(value, bool)):
            raise ManifestError(f"{where}: '{key}' must be {FIELD_TYPES[key].__name__}")


def _check_paths(table: dict, where: str) -> None:
    if not table["paths"]:
        raise ManifestError(f"{where}: 'paths' is empty")
    for pattern in table["paths"]:
        if not pattern or pattern.startswith("/") or ".." in Path(pattern).parts:
            raise ManifestError(f"{where}: bad path glob {pattern!r} (must be relative, no '..')")


def _compile(pattern: str, flags: int, where: str, what: str) -> re.Pattern:
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ManifestError(f"{where}: invalid {what} regex: {exc}") from exc


GLOB_CHARS = set("*?[]")
# A new file is created with these permissions (mkstemp would leave 0600).
NEW_FILE_MODE = 0o644


def _parse_add_file(table: dict, opts: dict, where: str, base: Path | None) -> None:
    """Checks of an `add_file` patch; reads its source into opts["content"].

    The source is read when the manifests load, so a missing or unreadable
    file is a manifest error (exit 2) before anything is computed.
    """
    paths = table["paths"]
    if len(paths) != 1 or GLOB_CHARS & set(paths[0]):
        raise ManifestError(f"{where}: add_file takes exactly one literal destination in 'paths', no glob characters")
    dest = Path(paths[0])
    if not IGNORED_DIRS.isdisjoint(dest.parts):
        raise ManifestError(f"{where}: destination {paths[0]!r} is under an ignored folder ({sorted(IGNORED_DIRS)})")
    src = opts["source"]
    if not src or Path(src).is_absolute() or ".." in Path(src).parts:
        raise ManifestError(f"{where}: bad source {src!r} (must be relative to the patches dir, no '..')")
    if base is None:
        raise ManifestError(f"{where}: add_file needs the patches dir to find its source")
    path = (base / src).resolve()
    if not path.is_relative_to(base.resolve()):  # a symlink pointing out of the patches dir
        raise ManifestError(f"{where}: source {src!r} resolves outside the patches dir")
    if not path.is_file():
        raise ManifestError(f"{where}: source {src!r} not found")
    try:
        content = path.read_bytes().decode("utf-8")  # bytes as they are: CRLF and final newline kept
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{where}: source {src!r} is not UTF-8: {exc}") from exc
    if not marker_re(table["id"]).search(content):
        raise ManifestError(f"{where}: source {src!r} must contain the comment '{marker(table['id'])}'")
    opts["dest"] = dest.as_posix()
    opts["content"] = content


def parse_patch(table: dict, source: str, base: Path | None = None) -> Patch:
    where = f"{source}: patch {table.get('id', '?')!r}"
    op = table.get("op")
    if not isinstance(op, str) or op not in OPS:
        raise ManifestError(f"{where}: unknown op {op!r} (expected one of {sorted(OPS)})")
    req, opt = OPS[op]
    _check_table(table, where, COMMON_REQUIRED | req, COMMON_OPTIONAL | opt)
    if not ID_RE.fullmatch(table["id"]):
        raise ManifestError(f"{where}: id must match [a-z0-9-]+")
    _check_paths(table, where)
    opts = {k: table[k] for k in req | opt if k in table}
    if op in ("regex", "literal"):
        opts.setdefault("expect", 1)
        if opts["expect"] < 1:
            raise ManifestError(f"{where}: 'expect' must be >= 1")
        if not marker_re(table["id"]).search(opts["replace"]):
            raise ManifestError(f"{where}: 'replace' must contain the comment '{marker(table['id'])}'")
    if op == "regex":
        opts["rx"] = _compile(
            opts["pattern"], re.MULTILINE | (re.DOTALL if opts.get("dotall") else 0), where, "pattern"
        )
        try:
            opts["rx"].sub(opts["replace"], "")  # surfaces bad group references now, not at apply time
        except re.error as exc:
            raise ManifestError(f"{where}: invalid replace template: {exc}") from exc
    elif op == "literal":
        words = opts["find"].split()
        if not words:
            raise ManifestError(f"{where}: 'find' is empty")
        opts["rx"] = re.compile(r"\s+".join(map(re.escape, words)))
    elif op == "add_file":
        _parse_add_file(table, opts, where, base)
    else:
        if opts.get("expect_sig"):
            opts["sig_rx"] = _compile(opts["expect_sig"], 0, where, "expect_sig")
    if "fn" in opts and not FN_NAME_RE.fullmatch(opts["fn"]):
        raise ManifestError(f"{where}: 'fn' must be a bare function name")
    return Patch(table["id"], table["why"], table["paths"], op, table.get("required", True),
                 table.get("enabled", True), source, opts)  # fmt: skip


def parse_guard(table: dict, source: str) -> Guard:
    where = f"{source}: guard {table.get('id', '?')!r}"
    _check_table(table, where, GUARD_REQUIRED, GUARD_OPTIONAL)
    if not ID_RE.fullmatch(table["id"]):
        raise ManifestError(f"{where}: id must match [a-z0-9-]+")
    if table["level"] not in ("warn", "fail"):
        raise ManifestError(f"{where}: level must be 'warn' or 'fail'")
    _check_paths(table, where)
    allow = [_compile(a, 0, where, "allow") for a in table.get("allow", [])]
    return Guard(table["id"], table["why"], table["paths"], _compile(table["pattern"], re.MULTILINE, where, "pattern"),
                 allow, table["level"], source)  # fmt: skip


def load_manifests(directory: Path) -> tuple[list[Patch], list[Guard]]:
    patches: list[Patch] = []
    guards: list[Guard] = []
    errors: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            extra = set(data) - {"patch", "guard"}
            if extra:
                raise ManifestError(f"{path.name}: unknown top-level key(s) {sorted(extra)}")
            for table in data.get("patch", []):
                patch = parse_patch(table, path.name, directory)
                if patch.id in seen:
                    raise ManifestError(f"{path.name}: patch {patch.id!r} duplicates the one in {seen[patch.id]}")
                seen[patch.id] = path.name
                patches.append(patch)
            guard_ids = {g.id for g in guards}
            for table in data.get("guard", []):
                guard = parse_guard(table, path.name)
                if guard.id in guard_ids:
                    raise ManifestError(f"{path.name}: guard {guard.id!r} is defined twice")
                guard_ids.add(guard.id)
                guards.append(guard)
        except (ManifestError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            errors.append(str(exc) if isinstance(exc, ManifestError) else f"{path.name}: {exc}")
    if errors:
        raise ManifestError("\n".join(errors))
    return patches, guards


# -------------------------------------------------------------- Rust scanner

_TOKEN = re.compile(r"//[^\n]*|/\*|(?<![A-Za-z0-9_])(?:br|cr|r)(#*)\"|\"|'")
_BLOCK = re.compile(r"/\*|\*/")
_STRING_END = re.compile(r"[^\"\\]*(?:\\.[^\"\\]*)*\"", re.DOTALL)


def _char_literal_end(src: str, quote: int) -> int:
    """End of the char literal opening at `quote`, or -1 when it is a lifetime or label."""
    if src[quote + 1 : quote + 2] == "\\":
        end = src.find("'", quote + 3)  # skips the escaped char, so '\'' works
        return end + 1 if end >= 0 else -1
    return quote + 3 if src[quote + 2 : quote + 3] == "'" else -1


@functools.lru_cache(maxsize=512)
def mask(src: str) -> str:
    """Blank comments, strings and char literals; same length, newlines kept.

    On the masked text a plain regex or brace count only sees real code, and
    every index maps back to the original text.
    """
    out: list[str] = []
    pos = last = 0
    while match := _TOKEN.search(src, pos):
        token, end = match.group(), match.end()
        if token.startswith("//"):
            pass
        elif token == "/*":
            depth = 1
            while depth:
                nested = _BLOCK.search(src, end)
                if not nested:
                    end = len(src)
                    break
                depth += 1 if nested.group() == "/*" else -1
                end = nested.end()
        elif token == "'":
            end = _char_literal_end(src, match.start())
            if end < 0:  # lifetime or loop label: ordinary code
                pos = match.end()
                continue
        elif token == '"':
            closing = _STRING_END.match(src, end)
            end = closing.end() if closing else len(src)
        else:  # raw string r#"..."#, br"...", cr"..."
            close = '"' + match.group(1)
            found = src.find(close, end)
            end = len(src) if found < 0 else found + len(close)
        out.append(src[last : match.start()])
        out.append(re.sub(r"[^\r\n]", " ", src[match.start() : end]))
        pos = last = end
    out.append(src[last:])
    return "".join(out)


_BRACES = re.compile(r"[{}]")
_STRUCTURE = re.compile(r"[()\[\]{};]")


def match_brace(masked: str, open_idx: int) -> int | None:
    depth = 0
    for m in _BRACES.finditer(masked, open_idx):
        depth += 1 if m.group() == "{" else -1
        if depth == 0:
            return m.start()
    return None


def body_span(masked: str, pos: int) -> tuple[int, int] | None:
    """Span of the `{...}` that follows a signature or impl header starting at `pos`.

    Braces inside parentheses or brackets (const generics, array lengths) are
    skipped, and a `;` at top level means a declaration without a body.
    """
    depth = 0
    for m in _STRUCTURE.finditer(masked, pos):
        char = m.group()
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0:
            if char != "{":
                return None
            close = match_brace(masked, m.start())
            return (m.start(), close) if close is not None else None
    return None


_QUALIFIERS = re.compile(r"(?:(?<![A-Za-z0-9_])(?:pub(?:\s*\([^)]*\))?|async|const|unsafe|extern|default)\s+)+\Z")
_IMPL = re.compile(r"(?<![A-Za-z0-9_])impl(?![A-Za-z0-9_])")
_FN_ANY = re.compile(r"(?<![A-Za-z0-9_])fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def _impl_spans(masked: str, wanted: str) -> list[tuple[int, int]]:
    want = " ".join(wanted.split())
    spans = []
    for m in _IMPL.finditer(masked):
        before = masked[: m.start()].rstrip()
        # `-> impl Trait` and `x: impl Trait` are types, not impl blocks.
        if before and before[-1] not in "};{]" and not before.endswith(("unsafe", "default")):
            continue
        span = body_span(masked, m.end())
        if span and re.search(r"(?<!\w)" + re.escape(want) + r"(?!\w)", " ".join(masked[m.end() : span[0]].split())):
            spans.append(span)
    return spans


@dataclass
class Fn:
    file: str
    start: int  # first qualifier or `fn`
    open: int
    close: int
    sig: str  # qualifiers through just before `{`, whitespace collapsed
    line: int


def find_fns(rel: str, text: str, name: str, impl: str | None = None) -> list[Fn]:
    masked = mask(text)
    spans = _impl_spans(masked, impl) if impl else []
    found = []
    for m in re.finditer(r"(?<![A-Za-z0-9_])fn\s+" + re.escape(name) + r"(?![A-Za-z0-9_])", masked):
        if impl and not any(lo < m.start() < hi for lo, hi in spans):
            continue
        span = body_span(masked, m.end())
        if not span:
            continue
        quals = _QUALIFIERS.search(masked[max(0, m.start() - 120) : m.start()])
        start = m.start() - (len(quals.group()) if quals else 0)
        sig = " ".join(text[start : span[0]].split())
        found.append(Fn(rel, start, span[0], span[1], sig, text.count("\n", 0, m.start()) + 1))
    return found


def locate_fn(tree: Tree, files: list[str], name: str, impl: str | None = None) -> Fn:
    cands = [fn for rel in files for fn in find_fns(rel, tree.text(rel), name, impl)]
    if len(cands) == 1:
        return cands[0]
    if cands:
        raise Drift(f"{len(cands)} definitions of fn {name} match: " + ", ".join(f"{c.file}:{c.line}" for c in cands))
    reason = f"function {name} not found" + (f" in impl {impl!r}" if impl else "")
    if impl:
        outside = [f"{c.file}:{c.line}" for rel in files for c in find_fns(rel, tree.text(rel), name)]
        if outside:
            reason += f" (exists outside that impl: {', '.join(outside)})"
    names: dict[str, str] = {}
    for rel in files:
        text = tree.text(rel)
        for m in _FN_ANY.finditer(mask(text)):
            names.setdefault(m.group(1), f"{rel}:{text.count(chr(10), 0, m.start()) + 1}")
    near = difflib.get_close_matches(name, list(names), n=3, cutoff=0.6)
    if near:
        reason += "; similar: " + ", ".join(f"{n} ({names[n]})" for n in near)
    raise Drift(reason)


# --------------------------------------------------------------------- tree


class Tree:
    """The files under `root`, with the patches applied so far kept in memory."""

    def __init__(self, root: Path):
        self.root = root
        # None: a file `add_file` created, not on disk yet.
        self.orig: dict[str, str | None] = {}
        self.cur: dict[str, str] = {}

    def files(self, globs: list[str]) -> list[str]:
        found = set()
        for pattern in globs:
            for rel in glob.glob(pattern, root_dir=self.root, recursive=True):
                path = Path(rel)
                if IGNORED_DIRS.isdisjoint(path.parts) and (self.root / path).is_file():
                    found.add(path.as_posix())
            # files added in memory are seen by later patches and guards too
            found.update(rel for rel in self.new_files() if _glob_match(pattern, rel))
        return sorted(found)

    def new_files(self) -> list[str]:
        return [rel for rel, text in self.orig.items() if text is None]

    def exists(self, rel: str) -> bool:
        return rel in self.cur or (self.root / rel).is_file()

    def add(self, rel: str, text: str) -> None:
        """A file that is not on disk, written by `write_atomically` with the rest."""
        self.orig.setdefault(rel, None)
        self.cur[rel] = text

    def text(self, rel: str) -> str:
        if rel not in self.cur:
            # surrogateescape + newline="" round-trip any bytes, CRLF included.
            with open(self.root / rel, encoding="utf-8", errors="surrogateescape", newline="") as fh:
                self.orig[rel] = self.cur[rel] = fh.read()
        return self.cur[rel]

    def changed(self) -> dict[str, str]:
        return {rel: text for rel, text in self.cur.items() if text != self.orig[rel]}


def _glob_match(pattern: str, rel: str) -> bool:
    """Whether `rel` (posix) matches `pattern` as `glob.glob(recursive=True)` would."""
    if hasattr(glob, "translate"):  # Python 3.13+
        return re.fullmatch(glob.translate(pattern, recursive=True), rel) is not None
    # 3.11/3.12: fnmatch's `*` also crosses '/', close enough for `**/*.rs` globs
    return fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(rel, pattern.replace("**/", ""))


def write_atomically(root: Path, changes: dict[str, str]) -> None:
    staged: list[tuple[str, Path]] = []
    try:
        for rel, text in changes.items():
            dst = root / rel
            fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".", suffix=".tmp")
            staged.append((tmp, dst))
            with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
                fh.write(text)
            if dst.exists():
                shutil.copymode(dst, tmp)
            else:
                os.chmod(tmp, NEW_FILE_MODE)
    except OSError:
        for tmp, _ in staged:
            Path(tmp).unlink(missing_ok=True)
        raise
    for tmp, dst in staged:
        os.replace(tmp, dst)


# ---------------------------------------------------------------------- ops


def _nearest_line(tree: Tree, files: list[str], needle: str) -> str:
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    best, where = 0.0, ""
    for rel in files:
        for number, line in enumerate(tree.text(rel).splitlines(), 1):
            matcher.set_seq1(line.strip())
            if len(line.strip()) > 3 and matcher.real_quick_ratio() > best and matcher.quick_ratio() > best:
                ratio = matcher.ratio()
                if ratio > best:
                    best, where = ratio, f"{rel}:{number}: {line.strip()[:80]}"
    return f"; nearest line {where}" if best >= 0.6 else ""


def _rewrite(tree: Tree, p: Patch, files: list[str], repl) -> dict[str, str]:
    if p.opts.get("fn"):
        fn = locate_fn(tree, files, p.opts["fn"])
        scopes = [(fn.file, fn.open + 1, fn.close)]
    else:
        scopes = [(rel, 0, len(tree.text(rel))) for rel in files]
    plans = {}
    for rel, lo, hi in scopes:
        found = list(p.opts["rx"].finditer(tree.text(rel), lo, hi))
        if found:
            plans[rel] = found
    total = sum(len(v) for v in plans.values())
    if total != p.opts["expect"]:
        reason = f"{total} matches, expected {p.opts['expect']}"
        if len(plans) > 1:
            reason += " (" + ", ".join(f"{rel}: {len(v)}" for rel, v in plans.items()) + ")"
        if total == 0 and p.op == "literal":
            reason += _nearest_line(tree, [s[0] for s in scopes], p.opts["find"].strip().splitlines()[0].strip())
        raise Drift(reason)
    out = {}
    for rel, found in plans.items():
        text, parts, last = tree.text(rel), [], 0
        for m in found:
            parts += [text[last : m.start()], repl(m)]
            last = m.end()
        out[rel] = "".join(parts) + text[last:]
    return out


def op_regex(tree: Tree, p: Patch, files: list[str]) -> dict[str, str]:
    return _rewrite(tree, p, files, lambda m: m.expand(p.opts["replace"]))


def op_literal(tree: Tree, p: Patch, files: list[str]) -> dict[str, str]:
    return _rewrite(tree, p, files, lambda m: p.opts["replace"])


def _render_body(body: str, indent: str, eol: str) -> str:
    lines = textwrap.dedent(body).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return eol.join((indent + line).rstrip() for line in lines)


def op_fn(tree: Tree, p: Patch, files: list[str]) -> dict[str, str]:
    fn = locate_fn(tree, files, p.opts["fn"], p.opts.get("impl"))
    sig_rx = p.opts.get("sig_rx")
    if sig_rx and not sig_rx.search(fn.sig):
        raise Drift(f"signature of {fn.file}:{fn.line} changed: {fn.sig!r} does not match /{p.opts['expect_sig']}/")
    text = tree.text(fn.file)
    eol = "\r\n" if "\r\n" in text else "\n"
    line_start = text.rfind("\n", 0, fn.start) + 1
    base = re.match(r"[ \t]*", text[line_start:]).group()
    inner = base + "    "
    body = _render_body(p.opts["body"], inner, eol)
    head = "{" + eol + inner + marker(p.id) + (eol + body if body else "")
    if p.op == "replace_fn_body":
        new = head + eol + base + "}"
        return {fn.file: text[: fn.open] + new + text[fn.close + 1 :]}
    rest = text[fn.open + 1 :]
    if not re.match(r"[ \t]*\r?\n", rest):  # one-line fn: keep its code on its own line
        head += eol + inner
    return {fn.file: text[: fn.open] + head + rest}


def op_add_file(tree: Tree, p: Patch) -> tuple[str, dict[str, str]]:
    """(status, changes) of an `add_file` patch.

    The file's own marker says it is ours: identical bytes are ALREADY,
    different ones are refreshed (the source changed); a file there without
    the marker is upstream's, and is never overwritten.
    """
    rel, content = p.opts["dest"], p.opts["content"]
    if tree.exists(rel):
        text = tree.text(rel)
        if not marker_re(p.id).search(text):
            raise Drift(f"{rel} exists without '{marker(p.id)}' (upstream added a file there?); not overwritten")
        return ("ALREADY", {}) if text == content else ("APPLIED", {rel: content})
    path = tree.root / rel
    if path.exists():
        raise Drift(f"{rel} exists and is not a regular file")
    if not path.parent.is_dir():
        raise Drift(f"folder {Path(rel).parent.as_posix()} does not exist (crate moved?)")
    return "APPLIED", {rel: content}


OP_FUNCS = {"regex": op_regex, "literal": op_literal, "replace_fn_body": op_fn, "insert_fn_start": op_fn}


# ------------------------------------------------------------------- engine


@dataclass
class Result:
    id: str
    status: str
    files: list[str]
    detail: str = ""


def run_patches(tree: Tree, patches: list[Patch], only: set[str], skip: set[str]) -> list[Result]:
    results = []
    for p in patches:
        if not p.enabled:
            results.append(Result(p.id, "DISABLED", []))
            continue
        if (only and p.id not in only) or p.id in skip:
            results.append(Result(p.id, "SKIPPED", []))
            continue
        if p.op == "add_file":
            try:
                status, changes = op_add_file(tree, p)
            except Drift as exc:
                results.append(Result(p.id, "DRIFT" if p.required else "OPTIONAL-MISSING", [], str(exc)))
                continue
            for rel, text in changes.items():
                tree.add(rel, text)
            results.append(Result(p.id, status, [p.opts["dest"]]))
            continue
        files = tree.files(p.paths)
        done = [rel for rel in files if marker_re(p.id).search(tree.text(rel))]
        if done:
            results.append(Result(p.id, "ALREADY", done))
            continue
        try:
            if not files:
                raise Drift(f"no files match {p.paths}")
            changes = OP_FUNCS[p.op](tree, p, files)
        except Drift as exc:
            results.append(Result(p.id, "DRIFT" if p.required else "OPTIONAL-MISSING", [], str(exc)))
            continue
        tree.cur.update(changes)
        results.append(Result(p.id, "APPLIED", sorted(changes)))
    return results


@dataclass
class Finding:
    guard: Guard
    file: str
    line: int
    text: str


def run_guards(tree: Tree, guards: list[Guard]) -> list[Finding]:
    findings = []
    for g in guards:
        for rel in tree.files(g.paths):
            text = tree.text(rel)
            for m in g.pattern.finditer(text):
                if not any(a.fullmatch(m.group()) for a in g.allow):
                    findings.append(Finding(g, rel, text.count("\n", 0, m.start()) + 1, m.group()))
    return findings


# Branches that merge upstream: patched sources committed here would conflict with it.
PROTECTED_BRANCHES = {"main", "chore/local-patches"}


def on_main_checkout(root: Path) -> bool:
    def git(*args: str) -> str:
        try:
            run = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
        except OSError:
            return ""
        return run.stdout.strip() if run.returncode == 0 else ""

    top = git("rev-parse", "--show-toplevel")
    # symbolic-ref also works on a repository without commits yet.
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    return bool(top) and Path(top).resolve() == root and branch in PROTECTED_BRANCHES


def format_result(r: Result) -> str:
    line = f"{r.status:<16} {r.id}  {','.join(r.files) or '-'}"
    return f"{line}  {r.detail}" if r.detail else line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="tree to patch (default: this repository)")
    ap.add_argument("--check", action="store_true", help="compute and report, write nothing")
    ap.add_argument("--list", action="store_true", help="list the patches and guards, then exit")
    ap.add_argument("--only", action="append", default=[], metavar="ID")
    ap.add_argument("--skip", action="append", default=[], metavar="ID")
    ap.add_argument("--patches-dir", help="folder of *.toml manifests (default: next to this script)")
    ap.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here.parents[1]
    try:
        patches, guards = load_manifests(Path(args.patches_dir) if args.patches_dir else here)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return EXIT_MANIFEST

    if args.list:
        for p in patches:
            flags = ("on" if p.enabled else "off") + "," + ("required" if p.required else "optional")
            print(f"{p.id}  {p.op}  {flags}  {p.source}  {p.why}")
        for g in guards:
            print(f"{g.id}  guard  {g.level}  {g.source}  {g.why}")
        return EXIT_OK

    unknown = (set(args.only) | set(args.skip)) - {p.id for p in patches}
    if unknown:
        print(f"unknown patch id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return EXIT_MANIFEST
    if not root.is_dir():
        print(f"root is not a directory: {root}", file=sys.stderr)
        return EXIT_MANIFEST
    if not args.check and on_main_checkout(root):
        print(f"refusing to patch {root}: it is a checkout of a protected branch (use a build tree, or --check)", file=sys.stderr)
        return EXIT_MANIFEST

    tree = Tree(root)
    results = run_patches(tree, patches, set(args.only), set(args.skip))
    findings = run_guards(tree, guards)
    for r in results:
        print(format_result(r))
    for f in findings:
        shown = f.text[:100].replace("\n", "\\n")
        print(f"{'GUARD-' + f.guard.level.upper():<16} {f.guard.id}  {f.file}:{f.line}  {shown}")

    drift = any(r.status == "DRIFT" for r in results)
    failed = any(f.guard.level == "fail" for f in findings)
    code = EXIT_DRIFT if drift else EXIT_GUARD if failed else EXIT_OK
    changes = tree.changed()
    if code == EXIT_OK and changes and not args.check:
        write_atomically(root, changes)
    elif code != EXIT_OK and not args.check:
        print("nothing written: fix the DRIFT or guard failure first", file=sys.stderr)
    if args.json:
        report = {
            "exit": code,
            "patches": [vars(r) for r in results],
            "guards": [{"id": f.guard.id, "level": f.guard.level, "file": f.file, "line": f.line, "text": f.text} for f in findings],
        }  # fmt: skip
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())

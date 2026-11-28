#!/usr/bin/env python3
"""Apply the semantic local patches of scripts/patches/*.toml to a build tree.

Each patch is located by pattern or function name when it is applied, so it
survives upstream moving lines around, and it fails loudly (DRIFT) when its
target changed or disappeared. See README.md for the manifest format.

Ops: `regex` and `literal` rewrite matched text, `replace_fn_body` and
`insert_fn_start` edit a function found by name, `add_file` brings a
whole new file (kept under the patches dir, e.g. `files/`) into the tree,
and `add_dir` a whole folder of data files, any bytes, owned through a
record file inside it.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import functools
import glob
import hashlib
import json
import os
import re
import shutil
import stat
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
COMMON_OPTIONAL = {"required", "enabled", "glue_for"}
OPS = {  # op -> (required keys, optional keys)
    "regex": ({"pattern", "replace"}, {"expect", "dotall", "fn"}),
    "literal": ({"find", "replace"}, {"expect", "fn"}),
    "replace_fn_body": ({"fn", "body"}, {"impl", "expect_sig"}),
    "insert_fn_start": ({"fn", "body"}, {"impl", "expect_sig"}),
    "add_file": ({"source"}, set()),
    "add_dir": ({"source"}, set()),
}
GUARD_REQUIRED = {"id", "why", "paths", "pattern", "level"}
GUARD_OPTIONAL = {"allow"}
FIELD_TYPES = {
    "id": str, "why": str, "op": str, "required": bool, "enabled": bool,
    "pattern": str, "replace": str, "expect": int, "dotall": bool, "fn": str,
    "find": str, "body": str, "impl": str, "expect_sig": str, "level": str, "source": str,
    "glue_for": str,
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
    # Glue for another manifest's step: required when that step applied (or
    # was already there) in this run, optional otherwise. "" for a plain step.
    glue_for: str = ""


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


# The file inside an `add_dir` folder that says which patch owns it.
RECORD = ".aurora-local-patch"
BAD_NAME_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\\]")


def staging_of(dest_rel: str) -> str:
    """The sibling a first `add_dir` apply builds in, then renames into place."""
    dest = Path(dest_rel)
    return (dest.parent / f".{dest.name}{RECORD}.tmp").as_posix()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_add_dir(table: dict, opts: dict, where: str, base: Path | None) -> None:
    """Checks of an `add_dir` patch; reads every file of its source folder.

    opts["files"] maps each file's path inside the folder (posix) to its
    bytes. Read when the manifests load, like `add_file`: a missing, empty
    or unreadable folder is a manifest error before anything is computed.
    """
    paths = table["paths"]
    if len(paths) != 1 or GLOB_CHARS & set(paths[0]):
        raise ManifestError(
            f"{where}: add_dir takes exactly one literal destination folder in 'paths', no glob characters"
        )
    dest = Path(paths[0].rstrip("/"))
    if dest.as_posix() == ".":  # the root itself would own the whole tree
        raise ManifestError(f"{where}: destination {paths[0]!r} must be a folder below the root")
    if not IGNORED_DIRS.isdisjoint(dest.parts):
        raise ManifestError(f"{where}: destination {paths[0]!r} is under an ignored folder ({sorted(IGNORED_DIRS)})")
    src = opts["source"]
    if not src or Path(src).is_absolute() or ".." in Path(src).parts:
        raise ManifestError(f"{where}: bad source {src!r} (must be relative to the patches dir, no '..')")
    if base is None:
        raise ManifestError(f"{where}: add_dir needs the patches dir to find its source")
    top = (base / src).resolve()
    if not top.is_relative_to(base.resolve()):
        raise ManifestError(f"{where}: source {src!r} resolves outside the patches dir")
    if not top.is_dir():
        raise ManifestError(f"{where}: source folder {src!r} not found")
    files: dict[str, bytes] = {}
    for path in sorted(top.rglob("*")):
        rel = path.relative_to(top).as_posix()
        # the record is UTF-8 JSON: a name it cannot hold would crash the
        # write after the folder exists; control characters and backslashes
        # are refused too, they read differently on another system
        try:
            rel.encode("utf-8")
        except UnicodeEncodeError:
            raise ManifestError(f"{where}: source {src!r} holds a name that is not valid UTF-8 ({rel!r})") from None
        if BAD_NAME_RE.search(rel):
            raise ManifestError(
                f"{where}: source {src!r} holds a name with a control character or a backslash ({rel!r})"
            )
        if path.is_symlink():
            raise ManifestError(f"{where}: source {src!r} holds a symbolic link ({rel}): copy the file instead")
        if path.name == RECORD:
            raise ManifestError(f"{where}: source {src!r} holds a file named {RECORD} (the record's name)")
        if path.is_file():
            try:
                files[rel] = path.read_bytes()
            except OSError as exc:
                raise ManifestError(f"{where}: source file {rel!r} cannot be read: {exc}") from exc
    if not files:
        raise ManifestError(f"{where}: source folder {src!r} holds no file")
    opts["dest"] = dest.as_posix()
    opts["files"] = files
    opts["hashes"] = {rel: _sha256(data) for rel, data in files.items()}


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
    glue_for = table.get("glue_for", "")
    if "glue_for" in table:
        if not ID_RE.fullmatch(glue_for):
            raise ManifestError(f"{where}: 'glue_for' must be a patch id ([a-z0-9-]+)")
        if table.get("required") is True:
            raise ManifestError(
                f"{where}: 'glue_for' with required = true: glue is required only while its partner "
                f"{glue_for!r} applies; drop 'required'"
            )
        if glue_for == table["id"]:
            raise ManifestError(f"{where}: 'glue_for' names the patch itself")
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
    elif op == "add_dir":
        _parse_add_dir(table, opts, where, base)
    else:
        if opts.get("expect_sig"):
            opts["sig_rx"] = _compile(opts["expect_sig"], 0, where, "expect_sig")
    if "fn" in opts and not FN_NAME_RE.fullmatch(opts["fn"]):
        raise ManifestError(f"{where}: 'fn' must be a bare function name")
    return Patch(table["id"], table["why"], table["paths"], op, table.get("required", not glue_for),
                 table.get("enabled", True), source, opts, glue_for)  # fmt: skip


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


def _overlaps(patches: list[Patch]) -> list[str]:
    """An `add_dir` folder is one patch's, whole: a second `add_dir` on it,
    around it or inside it, or an `add_file` inside it, would fight over its
    files and its record. Disabled patches count: a disabled duplicate is
    still a mistake, found before someone enables it."""

    def inside(a: str, b: str) -> bool:
        return a == b or a.startswith(b + "/")

    dirs = [p for p in patches if p.op == "add_dir"]
    errors = []
    for i, d in enumerate(dirs):
        owner = f"the add_dir destination {d.opts['dest']!r} of patch {d.id!r} ({d.source})"
        for other in dirs[i + 1 :]:
            if inside(other.opts["dest"], d.opts["dest"]) or inside(d.opts["dest"], other.opts["dest"]):
                errors.append(
                    f"{other.source}: patch {other.id!r}: destination {other.opts['dest']!r} overlaps {owner}"
                )
        for f in patches:
            if f.op == "add_file" and inside(f.opts["dest"], d.opts["dest"]):
                errors.append(f"{f.source}: patch {f.id!r}: destination {f.opts['dest']!r} overlaps {owner}")
    return errors


def _glue_errors(patches: list[Patch]) -> list[str]:
    """A glue step lives in the later patch: its partner is a step of another
    manifest that comes earlier in the run. A partner not loaded is fine (the
    glue is then optional), a later one or one of the same manifest is not."""
    order = {p.id: i for i, p in enumerate(patches)}
    errors = []
    for i, p in enumerate(patches):
        if not p.glue_for or p.glue_for not in order:
            continue
        partner = patches[order[p.glue_for]]
        where = f"{p.source}: patch {p.id!r}: glue_for {partner.id!r}"
        if partner.source == p.source:
            errors.append(f"{where} is in the same manifest (glue joins two patches)")
        elif order[partner.id] > i:
            errors.append(f"{where} ({partner.source}) applies later; glue lives in the patch that applies later")
    return errors


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
    errors += _overlaps(patches) + _glue_errors(patches)
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
        # The `add_dir` folders: data, never seen by text patches or guards.
        self.data_dirs: set[str] = set()
        # Their write plans, run after the text files under the same gate.
        self.dir_plans: list[DirPlan] = []

    def in_data_dir(self, rel: str) -> bool:
        return any(rel == d or rel.startswith(d + "/") for d in self.data_dirs)

    def files(self, globs: list[str]) -> list[str]:
        found = set()
        for pattern in globs:
            for rel in glob.glob(pattern, root_dir=self.root, recursive=True):
                path = Path(rel)
                if (
                    IGNORED_DIRS.isdisjoint(path.parts)
                    and not self.in_data_dir(path.as_posix())
                    and (self.root / path).is_file()
                ):
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


@dataclass
class DirPlan:
    """What an `add_dir` patch writes, run only when the whole run may write."""

    dest: str  # the folder, relative to the root (posix)
    write: dict[str, bytes]  # path inside the folder -> bytes
    remove: list[str]  # recorded files no longer in the source
    dirs_new: list[str]  # folders to create, parents first
    dirs_drop: list[str]  # folders it created that the removals empty, deepest first
    interim: dict  # the record written before any file: old and new files
    record: dict  # the record written last
    staging: str = ""  # first apply: the sibling built whole, then renamed into place
    clear_staging: bool = False  # a staging sibling of ours left by a run cut short


def _parents(rel: str) -> list[str]:
    """The folders above `rel` inside the folder, outermost first ('a', 'a/b')."""
    parts = rel.split("/")[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def _read_record(dest_rel: str, path: Path) -> dict:
    """The record file of an existing folder; DRIFT when it is not ours to touch."""
    if not path.is_file() or path.is_symlink():
        raise Drift(f"{dest_rel} exists without its record {RECORD} (not this patch's folder); not touched")
    try:
        rec = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Drift(f"{dest_rel}/{RECORD} is not a readable record ({exc}); not touched") from exc
    ok = (
        isinstance(rec, dict)
        and isinstance(rec.get("id"), str)
        and isinstance(rec.get("files"), dict)
        and isinstance(rec.get("dirs"), list)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in rec["files"].items())
        and all(isinstance(d, str) for d in rec["dirs"])
    )
    # a record never points out of its folder
    if ok:
        ok = all(r and not r.startswith("/") and ".." not in r.split("/") for r in [*rec["files"], *rec["dirs"]])
    if not ok:
        raise Drift(f"{dest_rel}/{RECORD} is not a record of this tool; not touched")
    return rec


def _lstat_below(root: Path, rel: str) -> os.stat_result | None:
    """lstat of root/rel, never through a link; None when a part is missing.

    `is_file()` and `is_dir()` follow links: a linked folder on the way would
    lead a write or a delete out of the destination, or out of `--root`. So
    every part of `rel` is looked at with lstat, and a symbolic link anywhere
    (the last part included) or a non-folder before the last is DRIFT.
    """
    path, parts = root, rel.split("/")
    for i, part in enumerate(parts):
        path = path / part
        shown = "/".join(parts[: i + 1])
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(st.st_mode):
            raise Drift(f"{shown} is a symbolic link; not followed, not touched")
        if i < len(parts) - 1 and not stat.S_ISDIR(st.st_mode):
            raise Drift(f"{shown} is not a folder; not touched")
    return st


def _is_dir(st: os.stat_result | None) -> bool:
    return st is not None and stat.S_ISDIR(st.st_mode)


def _staging_is_ours(path: Path, patch_id: str) -> bool:
    """Whether a staging sibling was left by this patch's own first apply.

    Its record is the first file written into it, so it is ours when that
    record names this patch, or, cut before the record was in place, when
    it holds nothing but the record's own temporary file.
    """
    record = path / RECORD
    if os.path.lexists(record):
        try:
            return _read_record(path.name, record)["id"] == patch_id
        except Drift:
            return False
    return all(
        e.name.startswith(RECORD + ".") and e.name.endswith(".tmp") and stat.S_ISREG(e.lstat().st_mode)
        for e in path.iterdir()
    )


def op_add_dir(tree: Tree, p: Patch) -> tuple[str, DirPlan | None]:
    """(status, plan) of an `add_dir` patch.

    The folder is ours when its record names this patch: then its files are
    brought to the source's (changed ones rewritten, recorded ones gone from
    the source removed, with the folders that leaves empty), files nobody
    recorded are kept, and a wanted path holding an unrecorded file is DRIFT.
    A folder there without our record is someone else's: DRIFT, untouched.
    """
    dest_rel, files, hashes = p.opts["dest"], p.opts["files"], p.opts["hashes"]
    dest = tree.root / dest_rel
    wanted_dirs = sorted({d for rel in files for d in _parents(rel)}, key=lambda d: (d.count("/"), d))

    def record_of(fs: dict[str, str], ds) -> dict:
        return {"id": p.id, "source": p.opts["source"], "files": dict(sorted(fs.items())), "dirs": sorted(ds)}

    def below(rel: str) -> os.stat_result | None:
        return _lstat_below(tree.root, f"{dest_rel}/{rel}")

    # the destination and its parents inside the root too: a linked parent
    # would put the folder outside the tree
    dest_st = _lstat_below(tree.root, dest_rel)
    if dest_st is not None and not stat.S_ISDIR(dest_st.st_mode):
        raise Drift(f"{dest_rel} exists and is not a folder")
    if dest_st is None:
        parent = Path(dest_rel).parent.as_posix()
        if parent != "." and not _is_dir(_lstat_below(tree.root, parent)):
            raise Drift(f"folder {parent} does not exist (crate moved?)")
        staging = staging_of(dest_rel)
        st = _lstat_below(tree.root, staging)
        if st is not None and not (stat.S_ISDIR(st.st_mode) and _staging_is_ours(tree.root / staging, p.id)):
            raise Drift(f"{staging} is in the way and is not this patch's unfinished copy; not touched")
        record = record_of(hashes, wanted_dirs)
        plan = DirPlan(dest_rel, dict(files), [], wanted_dirs, [], record, record, staging, st is not None)
        return "APPLIED", plan
    rec = _read_record(dest_rel, dest / RECORD)
    if rec["id"] != p.id:
        raise Drift(f"{dest_rel} belongs to patch {rec['id']!r} (its {RECORD}), not {p.id!r}; not touched")
    old_files, old_dirs = rec["files"], set(rec["dirs"])
    present: set[str] = set()
    for d in wanted_dirs:
        st = below(d)
        if st is not None and not stat.S_ISDIR(st.st_mode):
            raise Drift(f"{dest_rel}/{d} is not a folder")
        if st is not None:
            present.add(d)
    write: dict[str, bytes] = {}
    for rel, data in files.items():
        st = below(rel)
        if st is not None and not stat.S_ISREG(st.st_mode):
            raise Drift(f"{dest_rel}/{rel} is not a regular file")
        if st is None:
            write[rel] = data
        elif rel not in old_files:
            raise Drift(f"{dest_rel}/{rel} is not in the record (someone else's file?); not touched")
        elif _sha256((dest / rel).read_bytes()) != hashes[rel]:
            write[rel] = data
    remove = []
    for rel in sorted(old_files):
        if rel not in files and (st := below(rel)) is not None and stat.S_ISREG(st.st_mode):
            remove.append(rel)
    dirs_new = [d for d in wanted_dirs if d not in present]
    old_present = {d for d in old_dirs if _is_dir(below(d))}
    # folders it created, no longer wanted, that hold nothing once the removals are done
    gone = {dest / rel for rel in remove}
    drop: list[str] = []
    for d in sorted(old_present - set(wanted_dirs), key=lambda d: (-d.count("/"), d)):
        path = dest / d
        left = [e for e in path.iterdir() if e not in gone and (e.relative_to(dest).as_posix() not in drop)]
        if not left:
            drop.append(d)
    final_dirs = (old_present - set(drop)) | set(dirs_new)
    record = record_of(hashes, final_dirs)
    if not write and not remove and not dirs_new and not drop and rec == record:
        return "ALREADY", None
    interim = record_of({**old_files, **hashes}, old_dirs | set(dirs_new))
    return "APPLIED", DirPlan(dest_rel, write, remove, dirs_new, drop, interim, record)


def _write_bytes(path: Path, data: bytes) -> None:
    """One file, atomically; a new one 0644, an existing one keeps its mode."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if path.exists():
            shutil.copymode(path, tmp)
        else:
            os.chmod(tmp, NEW_FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _record_bytes(record: dict) -> bytes:
    # sorted, relative, no time: the same source gives the same bytes
    return (json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def apply_dir_plan(root: Path, plan: DirPlan) -> None:
    """Record first (old and new files), then the files, then the final record:
    a run cut short leaves every file it may have written recorded, and the
    next run finishes the work.

    A first apply builds the whole folder in its staging sibling instead (the
    record first, so a leftover is known to be ours) and renames it into
    place: a run cut short leaves no destination without its record, and the
    next run clears the leftover and builds again."""
    dest = root / plan.dest
    if plan.staging:
        build = root / plan.staging
        if plan.clear_staging:
            shutil.rmtree(build)
        build.mkdir()
        _write_bytes(build / RECORD, _record_bytes(plan.record))
        for d in plan.dirs_new:
            (build / d).mkdir()
        for rel, data in plan.write.items():
            _write_bytes(build / rel, data)
        if os.path.lexists(dest):  # appeared since the plan: never replaced
            raise FileExistsError(f"{plan.dest} appeared during the run; its copy is left in {plan.staging}")
        os.rename(build, dest)
        return
    dest.mkdir(exist_ok=True)
    record = dest / RECORD
    if not record.exists() or record.read_bytes() != _record_bytes(plan.interim):
        _write_bytes(record, _record_bytes(plan.interim))
    for d in plan.dirs_new:
        (dest / d).mkdir(exist_ok=True)
    for rel, data in plan.write.items():
        _write_bytes(dest / rel, data)
    for rel in plan.remove:
        (dest / rel).unlink(missing_ok=True)
    kept = []
    for d in plan.dirs_drop:
        try:
            (dest / d).rmdir()
        except OSError:
            kept.append(d)  # not empty after all: kept, and still ours
    final = dict(plan.record, dirs=sorted({*plan.record["dirs"], *kept}))
    _write_bytes(record, _record_bytes(final))


OP_FUNCS = {"regex": op_regex, "literal": op_literal, "replace_fn_body": op_fn, "insert_fn_start": op_fn}


# ------------------------------------------------------------------- engine


@dataclass
class Result:
    id: str
    status: str
    files: list[str]
    detail: str = ""
    glue_for: str = ""


# A partner in one of these states is part of the tree: its glue must follow.
PARTNER_PRESENT = {"APPLIED", "ALREADY"}


def _missed(p: Patch, exc: Drift, results: list[Result]) -> Result:
    """The result of a step whose target is not there.

    A glue step is required exactly when its partner applied or was already
    there in this run; with the partner absent (not loaded, filtered out,
    disabled, missing itself) it is optional, and the reason says why.
    """
    reason = str(exc)
    if p.glue_for:
        partner = next((r.status for r in results if r.id == p.glue_for), None)
        if partner in PARTNER_PRESENT:
            return Result(p.id, "DRIFT", [], f"glue for {p.glue_for!r} ({partner}): {reason}", p.glue_for)
        absent = "not in this run" if partner is None else partner
        return Result(p.id, "OPTIONAL-MISSING", [], f"partner {p.glue_for!r} {absent}: {reason}", p.glue_for)
    return Result(p.id, "DRIFT" if p.required else "OPTIONAL-MISSING", [], reason)


def run_patches(tree: Tree, patches: list[Patch], only: set[str], skip: set[str]) -> list[Result]:
    results = []
    # every `add_dir` folder, whether its patch runs or not: an old copy of
    # it stays data for the text patches and guards
    # (their staging siblings too: a leftover copy is data as well)
    tree.data_dirs = {d for p in patches if p.op == "add_dir" for d in (p.opts["dest"], staging_of(p.opts["dest"]))}
    for p in patches:
        if not p.enabled:
            results.append(Result(p.id, "DISABLED", []))
            continue
        if (only and p.id not in only) or p.id in skip:
            results.append(Result(p.id, "SKIPPED", []))
            continue
        if p.op == "add_dir":
            try:
                status, plan = op_add_dir(tree, p)
            except Drift as exc:
                results.append(_missed(p, exc, results))
                continue
            written, removed = (len(plan.write), len(plan.remove)) if plan else (0, 0)
            if plan:
                tree.dir_plans.append(plan)
            detail = f"{len(p.opts['files'])} files: {written} written, {removed} removed"
            results.append(Result(p.id, status, [p.opts["dest"]], detail))
            continue
        if p.op == "add_file":
            try:
                status, changes = op_add_file(tree, p)
            except Drift as exc:
                results.append(_missed(p, exc, results))
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
            results.append(_missed(p, exc, results))
            continue
        tree.cur.update(changes)
        results.append(Result(p.id, "APPLIED", sorted(changes)))
    glue = {p.id: p.glue_for for p in patches}
    for r in results:
        r.glue_for = glue[r.id]
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
            need = f"glue-for:{p.glue_for}" if p.glue_for else "required" if p.required else "optional"
            flags = ("on" if p.enabled else "off") + "," + need
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
        print(
            f"refusing to patch {root}: it is a checkout of a protected branch (use a build tree, or --check)",
            file=sys.stderr,
        )
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
    if code == EXIT_OK and (changes or tree.dir_plans) and not args.check:
        write_atomically(root, changes)
        for plan in tree.dir_plans:
            apply_dir_plan(root, plan)
    elif code != EXIT_OK and not args.check:
        print("nothing written: fix the DRIFT or guard failure first", file=sys.stderr)
    if args.json:
        report = {
            "exit": code,
            # `glue_for` only on glue steps: the other rows keep their shape
            "patches": [{k: v for k, v in vars(r).items() if k != "glue_for" or v} for r in results],
            "guards": [{"id": f.guard.id, "level": f.guard.level, "file": f.file, "line": f.line, "text": f.text} for f in findings],
        }  # fmt: skip
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())

# Local patches

Private modifications on top of upstream that keep working after `git pull`.
A plain `git apply` diff breaks when lines move, so each change is described
**semantically** in a TOML manifest, located by pattern or function name when
it is applied, and applied to a **throwaway build tree**. If the target changed
or disappeared the tool reports `DRIFT` and exits non-zero: it never produces a
build without its patches. Upstream files stay untouched in git; only new files
under `scripts/patches/` are committed, so upstream merges cannot conflict.

```bash
cp -r <checkout> <build-tree>            # or a worktree on any branch but main
python3 scripts/patches/apply_patches.py --root <build-tree>
python3 scripts/patches/apply_patches.py --root <build-tree> --check   # dry run
python3 -m unittest                      # tests, from scripts/patches
```

Python 3.11+, stdlib only. `--root` defaults to this repository, but a real run
is refused when the root is a checkout of `main` (`--check` is always allowed).
Files under `target/`, `.git/` and `.scratch/` are never patched or scanned.

## CLI

`apply_patches.py [--root DIR] [--check] [--list] [--only ID]... [--skip ID]... [--patches-dir DIR] [--json FILE]`

Manifests are every `*.toml` of the patches dir, by filename (`NN-name.toml`);
patches apply in file order, then in-file order, and later ones see the results
of earlier ones. Everything is computed in memory first: if any required patch
drifts (or a `fail` guard fires) **nothing is written**. Files are written
atomically, bytes are preserved (CRLF, final newline). A second run reports
`ALREADY` everywhere and changes nothing.

One line per patch: `APPLIED | ALREADY | SKIPPED | DISABLED | DRIFT |
OPTIONAL-MISSING`, the id, the file, and for a drift the reason plus the
nearest hint.

Exit codes: `0` ok, `1` a required patch drifted, `2` manifest or usage error
(or refusal to patch `main`), `3` a `fail` guard fired (`1` wins if both).

## Manifest reference

```toml
[[patch]]
id = "telemetry-news"          # unique across all manifests, [a-z0-9-]+
why = "one-line reason"
paths = ["crates/aurora-viewer/src/**/*.rs"]   # broad on purpose: a moved file is still found
op = "replace_fn_body"
required = true                # default; false => OPTIONAL-MISSING instead of DRIFT
enabled = true                 # default; false => DISABLED
```

Unknown keys, duplicate ids, bad regexes and missing fields are manifest errors
that name the file and the patch.

| op | fields | what it does |
|---|---|---|
| `regex` | `pattern`, `replace`, `expect = 1`, `dotall`, `fn` | MULTILINE regex; `replace` is a `re` template (`\1`); `expect` is the exact match count across all `paths` |
| `literal` | `find`, `replace`, `expect = 1`, `fn` | text match where every run of whitespace in `find` matches any whitespace (newlines, CRLF), so rustfmt reflows do not break it; `replace` is inserted as is |
| `replace_fn_body` | `fn`, `body`, `impl`, `expect_sig` | replaces what is between the function's braces, keeps the signature exactly as upstream wrote it |
| `insert_fn_start` | `fn`, `body`, `impl`, `expect_sig` | inserts `body` right after the opening brace |
| `add_file` | `source` | brings a whole new file into the tree: `paths` is its one destination (no glob), `source` a file under the patches dir (`files/…`) |

- `fn` (for `regex`/`literal`: restrict matching to that function's body) finds
  `fn <name>` across all `paths`, whatever the qualifiers, generics, attributes
  or line breaks. **Exactly one** definition must match, otherwise DRIFT with
  the candidates. Comments and strings are ignored; the brace matcher handles
  nested block comments, strings, raw and byte strings, char literals versus
  lifetimes, and `where` clauses.
- `impl = "Display for App"` keeps only functions inside an `impl` block whose
  header contains that text (`impl = "App"` matches `impl App` and
  `impl<'a> Foo for App<'a>`).
- `expect_sig` is a regex searched in the signature: from the qualifiers
  (`pub async ...`) to just before `{`, whitespace collapsed to single spaces.
  A mismatch is DRIFT, because a changed signature may make `body` wrong.
- `body` is dedented and re-indented to the function; use `'''` literal strings.

**`add_file`** is for a new module too large to inline in a `replace` string
(it gets rustfmt, editor support and real line numbers where it lives, under
`scripts/patches/files/`, which is not scanned for manifests). The source must
be UTF-8, stay inside the patches dir, and contain the patch's marker itself;
it is copied byte for byte (CRLF, final newline) with mode 0644. The
destination's folder must exist (else DRIFT: the crate moved?). Absent: the
file is created (APPLIED). Present with the marker: ALREADY when identical,
else refreshed from the source (APPLIED). Present **without** the marker:
DRIFT, never overwritten (upstream added a file of that name). Later patches
and guards see the new file like any other.

**Idempotence marker.** `// aurora-local-patch: <id>` (`# aurora-local-patch:
<id>` in a TOML target such as `Cargo.lock`) is the only "already applied"
signal. `replace_fn_body` and `insert_fn_start` add it themselves.
For `regex` and `literal` your `replace` **must contain it** (manifest error
otherwise), as a comment that stays valid Rust where it lands:

```toml
replace = "let enabled = false; // aurora-local-patch: telemetry-off"
```

If the marker of an id exists in any file under its `paths`, the patch reports
`ALREADY`.

### Guards

Guards detect **new** things appearing upstream. They run on the patched
content, are read-only, and report every unallowed match as `file:line`.

```toml
[[guard]]
id = "new-egress-hosts"
why = "a new network endpoint needs a human look"
paths = ["crates/**/*.rs"]
pattern = 'https?://[^"\s]+'
allow = ['https://example\.org/.*']   # a match fully matching one of these is fine
level = "warn"                        # "warn": report only; "fail": exit 3, write nothing
```

## Adding a patch

1. Pick the narrowest op: `literal` for a one-line change, `replace_fn_body` /
   `insert_fn_start` for a whole behaviour. Prefer `fn` + `expect_sig` over line
   patterns: function names survive upstream churn better than text.
2. Write it in `NN-topic.toml`, with a `why`. Keep `paths` broad.
3. `apply_patches.py --check --only <id>` on a fresh tree: `APPLIED`. Apply for
   real, run it again: `ALREADY`.

## When a patch reports DRIFT

Upstream changed the target. Nothing was written. Read the reason:

- `0 matches` / `function ... not found`: look at the hint (similar function
  names, nearest line). Renamed or moved: widen `paths` or update `fn`/`find`.
  Removed upstream: the patch may be obsolete, set `enabled = false` and say why.
- `N matches, expected 1` / `N definitions`: upstream added a look-alike. Tighten
  the pattern, add `impl`, or `fn` scoping; raise `expect` only if all are meant.
- `signature ... changed`: read the new signature and check that `body` still
  fits before updating `expect_sig`.

Never "fix" a drift by loosening a pattern until it matches: check the new code
first. A patch for something optional can be `required = false`.

## Merging upstream: `README.md` stays ours

The top-level `README.md` is this branch's own short page, and upstream edits its `README.md` in almost every commit, so each "Merge upstream" would conflict on it. A local merge driver keeps our file. It is not tracked; on a fresh clone, once:

```bash
git config merge.ours.driver true
echo 'README.md merge=ours' >> .git/info/attributes
```

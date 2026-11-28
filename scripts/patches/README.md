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
ruff check --line-length 120 . && ruff format --check --line-length 120 .   # style, from scripts/patches
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
glue_for = "other-step-id"     # glue: required only while that step is in the tree (below)
```

Unknown keys, duplicate ids, bad regexes and missing fields are manifest errors
that name the file and the patch.

**`glue_for`** marks a step as glue between two patches: it names the step of
an earlier patch that creates what this step anchors on (the step that adds the
line, the function, or the file: for an anchor in a file brought by `add_file`,
that `add_file` step).

- The partner step reported `APPLIED` or `ALREADY` in this run: the glue step
  is required, a miss is `DRIFT` (exit 1, nothing written), and the reason
  starts with `glue for '<id>'`.
- The partner is not part of this run (its manifest is not loaded, or `--only`
  / `--skip` leave it out), or it is `DISABLED`, or itself `OPTIONAL-MISSING`:
  the glue step is tried, and a miss is `OPTIONAL-MISSING` with the reason
  `partner '<id>' <state>`.
- Manifest errors: a partner in the same manifest, a partner that applies
  later (glue lives in the later patch), the step naming itself, and
  `glue_for` with `required = true` (a glue step is required exactly while its
  partner applies). A partner id that is not loaded is fine. Guards take no
  `glue_for`.
- `--list` shows `glue-for:<id>` in place of `required` / `optional`; the
  `--json` report carries `glue_for` on the rows of glue steps.

| op | fields | what it does |
|---|---|---|
| `regex` | `pattern`, `replace`, `expect = 1`, `dotall`, `fn` | MULTILINE regex; `replace` is a `re` template (`\1`); `expect` is the exact match count across all `paths` |
| `literal` | `find`, `replace`, `expect = 1`, `fn` | text match where every run of whitespace in `find` matches any whitespace (newlines, CRLF), so rustfmt reflows do not break it; `replace` is inserted as is |
| `replace_fn_body` | `fn`, `body`, `impl`, `expect_sig` | replaces what is between the function's braces, keeps the signature exactly as upstream wrote it |
| `insert_fn_start` | `fn`, `body`, `impl`, `expect_sig` | inserts `body` right after the opening brace |
| `add_file` | `source` | brings a whole new file into the tree: `paths` is its one destination (no glob), `source` a file under the patches dir (`files/…`) |
| `add_dir` | `source` | brings a whole folder of data files (any bytes) into the tree: `paths` is its one destination folder (no glob), `source` a folder under the patches dir (`files/…/`) |

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

**`add_dir`** is for data a patch ships (pictures, JSON): files that cannot
carry a marker. Every file under `source` is read when the manifests load
(no symbolic link, no file named `.aurora-local-patch`, at least one file,
names in valid UTF-8 without control characters or backslashes) and copied
byte for byte, new files with mode 0644. A destination is one patch's
whole: two `add_dir` destinations that are the same folder or one inside
the other, or an `add_file` inside an `add_dir` folder, are a manifest
error, disabled patches included. Ownership is a record
file, `.aurora-local-patch`, inside the destination: JSON with the patch id,
the source, each file's SHA-256 and the folders it created (sorted, relative,
no date: the same source gives the same bytes).

- Destination absent (its parent must exist, else DRIFT): created, APPLIED.
- Our record and every file as the source has it: ALREADY, nothing written
  (the record included).
- Our record, something differs: APPLIED. Changed files are rewritten,
  recorded files gone from the source are deleted, and the folders it
  created that this leaves empty are removed. Files nobody recorded are
  kept; an unrecorded file at a path the source wants is DRIFT.
- No record, or another patch's: DRIFT, the folder untouched.
- A symbolic link, or a file where a folder should be, anywhere on the way
  to a path it writes or deletes (the destination and its parents in the
  tree included): DRIFT, nothing followed, nothing touched.

It writes under the same rule as the rest (nothing on DRIFT, a `fail` guard
or `--check`). A new folder is built whole in a sibling,
`.<name>.aurora-local-patch.tmp` (its record first), then renamed into
place; a sibling of that name left by a run cut short is removed and built
again when it holds this patch's record or nothing but an unfinished
record, else DRIFT. An
existing folder gets the record first (old and new files), then the files
and deletions, then the final record. Either way a run cut short is
finished by the next. The line reads `<folder>  N files: x written, y removed`. Text patches
and guards never see an `add_dir` folder, even under `**/*`, whether its
patch runs or not.

**Idempotence marker.** `// aurora-local-patch: <id>` (`# aurora-local-patch:
<id>` in a TOML target such as `Cargo.lock`) is the only "already applied"
signal. `replace_fn_body` and `insert_fn_start` add it themselves.
(`add_dir` uses its record file instead.)
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

## Patch design policy

A patch is one feature as the user sees it, worn like a backpack: the feature is its own module, and where it must, a wire reaches into the body, politely. These rules hold for every new patch and for every change to an existing one.

Why: the original may catch up or ship a similar feature, and then one patch must be dropped or reworked without breaking another. A patch may also be offered to the original as a pull request, and it has to stand on plain upstream for that.

### Where new work goes

1. It extends a feature that an existing patch owns: it goes into that patch. A second patch for the same feature is not made.
2. It is a feature of its own: a new patch.
3. Two patches need the same code: a helper patch holds it. Its file name sorts before both (manifests apply in file-name text order), it needs neither of them, and both name it in their header.

The line test settles doubts: if this patch alone were merged, would every line of it have a reason in this patch? A line that only serves another feature leaves, and becomes optional glue of that feature's patch.

Settle the scope first: what the feature includes, what data it carries, and which policies of the original apply to it (licences, languages, generated content). Changing the scope after the code is written costs a second pass.

### Stand alone

A patch applies, compiles and passes its tests on plain upstream with nothing else but the patches its header names.

### Dependencies

- One way only, and declared. The header says `Needs patch NN-name: <the exact reason>` (the symbol, the anchor or the data it uses).
- No cross dependency. Two patches that need each other are one patch when they are one feature for the user; otherwise what they share moves into a helper patch (rule 3 above).
- Glue between two features lives in the patch that applies later, as steps with `glue_for = "<partner step>"`, the step that creates what the glue anchors on. With the other patch absent, the steps report `OPTIONAL-MISSING` and each feature works on its own; with it present, a glue step that no longer finds its anchor is `DRIFT`, so no function is lost silently after a catch-up.
- Plain `required = false` is for a step that is optional for another reason: a line in upstream documentation that may move or go away without any function being lost. It is never used for glue: it turns every miss into `OPTIONAL-MISSING`, the partner present or not.
- When losing a step would weaken a protection, a `fail` guard also watches for the state "the other patch is there, the glue is not"; guards are kept even where `glue_for` covers the step.
- The privacy and security patches are never weakened by a cut. Their tripwire guards depend on other patches on purpose: declare them, do not cut them.

### One copy of everything (DRY, best effort)

- A patch calls code; it does not copy it, neither from another patch nor from upstream.
- A constant, a layout or a rule has one owner. Others read it from there.
- A helper that two parts of one feature need lives once, in that feature's own module.
- An upstream helper is reused as it is. It is not moved and its signature is not changed for convenience; when that seems needed, the owner decides.
- When cutting a link would need a copy, the dependency stays and is declared.

### No function lost

A cut or a move leaves the behaviour with all patches applied exactly as it was. The header says what the patch does alone when a partner is absent.

### Seams that keep a patch modular

- The feature's logic lives in its own files (`add_file`, `add_dir`). Upstream files get thin hooks only: one call per hook point, no logic inline. Shader code too: its own `.wgsl` file from the start, joined the way upstream already joins its shader files.
- Every seam follows a pattern upstream already uses (how it joins shader sources, embeds data files, names catalogue keys, builds windows from the shared widgets). A seam of our own invention is the last choice.
- Data sits behind one loader. The feature does nothing on empty data, and a feature that is switched off changes nothing for the user.
- Code that waits for data stays whole and inert; code that only served something left out is removed, not hidden.
- Measure the backpack after the first draft, not at the end: lines in new files against lines in existing files (`git diff --numstat`). Hooks in existing files stay a small share.
- Interface text uses catalogue keys with the feature's own prefix.
- Tests live in the patch's own files. A test that needs another patch is a `glue_for` step of the patch that applies later.
- Two patches avoid editing the same lines. Where they must (a shared count in a test, a list of modules), each pattern works with or without the other patch and in either order.

### Proof before a patch is folded in

1. `--check` with only its steps, on a fresh tree: every step `APPLIED` (`glue_for` steps of absent partners `OPTIONAL-MISSING`). Apply, run again: `ALREADY`.
2. It compiles alone (the lint command of the continuous integration) and its tests pass alone.
3. The full set: no `DRIFT`, no new guard finding, lint clean, its tests pass.
4. Leave-one-out: without it, no other patch drifts except the ones whose header names it.
5. The header is complete: what the patch does, what it needs, which optional glue it carries for which patch.

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
first. A patch for something optional can be `required = false`; glue for
another patch is `glue_for` instead (see "Patch design policy").

## Merging upstream: `README.md` stays ours

The top-level `README.md` is this branch's own short page, and upstream edits its `README.md` in almost every commit, so each "Merge upstream" would conflict on it. A local merge driver keeps our file. It is not tracked; on a fresh clone, once:

```bash
git config merge.ours.driver true
echo 'README.md merge=ours' >> .git/info/attributes
```

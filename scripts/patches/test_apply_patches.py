"""Tests for apply_patches.py. Run from scripts/patches: python3 -m unittest"""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import apply_patches as ap

LIB = """\
use std::fmt;

pub struct App;

impl App {
    pub fn fetch_news(&self, url: &str) -> Result<String, Error> {
        let body = http_get(url)?;
        Ok(body)
    }

    fn other(&self) {
        println!("other");
    }
}

impl fmt::Display for App {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "app")
    }
}
"""


def toml_patch(**fields):
    """One [[patch]] table; values that are not str/bool/int are written as given."""

    def lit(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(lit(v) for v in value) + "]"
        return "'''" + value + "'''"

    fields = {"id": "p1", "why": "test", "paths": ["src/**/*.rs"], **fields}
    return "[[patch]]\n" + "".join(f"{k} = {lit(v)}\n" for k, v in fields.items()) + "\n"


class Env(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root, self.pdir = base / "tree", base / "patches"
        self.root.mkdir()
        self.pdir.mkdir()

    def put(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def get(self, rel):
        return (self.root / rel).read_bytes().decode("utf-8")

    def manifest(self, text, name="10-test.toml"):
        (self.pdir / name).write_text(text, encoding="utf-8")

    def run_main(self, *extra, root=None):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--root", str(root or self.root), "--patches-dir", str(self.pdir), *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ap.main(argv)
        return code, out.getvalue(), err.getvalue()


class ScannerTests(unittest.TestCase):
    def body(self, src, name="f", impl=None):
        fns = ap.find_fns("x.rs", src, name, impl)
        self.assertEqual(len(fns), 1, fns)
        return src[fns[0].open : fns[0].close + 1]

    def test_mask_keeps_length_and_newlines(self):
        src = 'let a = "x\ny"; // c\r\n/* z */ let b = 1;'
        masked = ap.mask(src)
        self.assertEqual(len(masked), len(src))
        self.assertEqual(masked.count("\n"), src.count("\n"))
        self.assertIn("let b = 1;", masked)
        self.assertNotIn("x", masked.replace("let", ""))

    def test_line_comment_braces(self):
        self.assertEqual(self.body("fn f() { // }\n a(); }"), "{ // }\n a(); }")

    def test_nested_block_comment(self):
        src = "fn f() { /* } /* } */ } */ a(); } fn g() {}"
        self.assertEqual(self.body(src), "{ /* } /* } */ } */ a(); }")

    def test_string_with_braces_and_escaped_quote(self):
        src = r'fn f() { let s = "}\" {"; a(); } fn g() {}'
        self.assertEqual(self.body(src), r'{ let s = "}\" {"; a(); }')

    def test_raw_strings(self):
        src = 'fn f() { let a = r#"}" "#; let b = r##"}"#}"##; z(); } fn g() {}'
        self.assertTrue(self.body(src).endswith("z(); }"))

    def test_byte_and_raw_byte_strings(self):
        src = 'fn f() { let a = b"}"; let b = br#"}"#; let c = b"\\""; z(); } fn g() {}'
        self.assertTrue(self.body(src).endswith("z(); }"))

    def test_char_literals_versus_lifetimes(self):
        for lit in ("'{'", "'}'", "'\\''", "'\\\\'"):
            src = f"fn f<'a>(x: &'a str) {{ let a = {lit}; z(); }} fn g() {{}}"
            self.assertTrue(self.body(src).endswith("z(); }"), lit)

    def test_lifetime_label_and_unicode_escape(self):
        src = "fn f() { 'outer: loop { let c = '\\u{7D}'; break 'outer; } z(); } fn g() {}"
        self.assertTrue(self.body(src).endswith("z(); }"))

    def test_where_clause(self):
        src = "fn f<T>(x: T) -> T\nwhere\n    T: Clone + Fn() -> u8,\n{\n    x\n}\n"
        self.assertEqual(self.body(src), "{\n    x\n}")

    def test_multiline_signature_and_qualifiers(self):
        src = "#[inline]\npub(crate) async unsafe fn f<T: Copy>(\n    a: [u8; 4],\n    b: T,\n) -> u8 {\n    1\n}\n"
        fn = ap.find_fns("x.rs", src, "f")[0]
        self.assertEqual(fn.sig, "pub(crate) async unsafe fn f<T: Copy>( a: [u8; 4], b: T, ) -> u8")

    def test_ignores_definitions_in_comments_and_strings(self):
        src = '// fn f() {}\n/* fn f() {} */\nconst S: &str = "fn f() {}";\nfn f() { real(); }\n'
        self.assertEqual(self.body(src), "{ real(); }")

    def test_declaration_without_body_is_skipped(self):
        src = "trait T { fn f(&self); }\nimpl T for X { fn f(&self) { go(); } }"
        self.assertEqual(self.body(src), "{ go(); }")

    def test_fn_pointer_type_is_not_a_definition(self):
        self.assertEqual(ap.find_fns("x.rs", "type F = fn(u8) -> u8; fn f() {}", "f")[0].file, "x.rs")

    def test_impl_filter(self):
        src = LIB
        self.assertIn("write!", self.body(src, "fmt", "fmt::Display for App"))
        self.assertIn("write!", self.body(src, "fmt", "Display"))
        self.assertEqual(ap.find_fns("x.rs", src, "fmt", "Debug"), [])
        self.assertEqual(ap.find_fns("x.rs", src, "other", "Display"), [])

    def test_impl_trait_in_signature_is_not_an_impl_block(self):
        src = "fn g() -> impl Iterator<Item = u8> { a() }\nimpl Foo { fn f() { b() } }"
        self.assertEqual(ap.find_fns("x.rs", src, "g", "Iterator"), [])
        self.assertEqual(len(ap.find_fns("x.rs", src, "f", "Foo")), 1)


class ManifestTests(Env):
    def assertManifestError(self, text, *needles):
        self.manifest(text)
        code, _, err = self.run_main()
        self.assertEqual(code, 2, err)
        for needle in needles:
            self.assertIn(needle, err)

    def test_unknown_key(self):
        self.assertManifestError(
            toml_patch(op="literal", find="a", replace="// aurora-local-patch: p1", bogus="x"),
            "10-test.toml",
            "p1",
            "bogus",
        )

    def test_unknown_op(self):
        self.assertManifestError(toml_patch(op="sed"), "unknown op")

    def test_missing_field(self):
        self.assertManifestError(toml_patch(op="regex", pattern="a"), "missing", "replace")

    def test_key_of_other_op(self):
        self.assertManifestError(
            toml_patch(op="regex", pattern="a", replace="// aurora-local-patch: p1", impl="X"), "impl"
        )

    def test_bad_regex(self):
        self.assertManifestError(
            toml_patch(op="regex", pattern="(", replace="// aurora-local-patch: p1"), "p1", "regex"
        )

    def test_bad_replace_template(self):
        self.assertManifestError(
            toml_patch(op="regex", pattern="a", replace=r"\1 // aurora-local-patch: p1"), "template"
        )

    def test_bad_expect_sig(self):
        self.assertManifestError(toml_patch(op="replace_fn_body", fn="f", body="x", expect_sig="("), "expect_sig")

    def test_bad_id(self):
        self.assertManifestError(toml_patch(id="Bad_Id", op="replace_fn_body", fn="f", body="x"), "id")

    def test_wrong_type(self):
        self.assertManifestError(
            toml_patch(op="literal", find="a", replace="// aurora-local-patch: p1", expect="1"), "expect"
        )

    def test_duplicate_id_across_files(self):
        body = toml_patch(op="replace_fn_body", fn="f", body="x")
        self.manifest(body, "10-a.toml")
        self.manifest(body, "20-b.toml")
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("20-b.toml", err)
        self.assertIn("10-a.toml", err)

    def test_missing_marker_in_replace(self):
        self.assertManifestError(toml_patch(op="literal", find="a", replace="b"), "aurora-local-patch: p1")

    def test_marker_of_longer_id_does_not_count(self):
        self.assertManifestError(
            toml_patch(op="literal", find="a", replace="// aurora-local-patch: p1-x"), "aurora-local-patch: p1"
        )

    def test_bad_guard(self):
        self.assertManifestError(
            '[[guard]]\nid = "g"\nwhy = "w"\npaths = ["*.rs"]\npattern = "x"\nlevel = "loud"\n', "guard", "level"
        )

    def test_bad_toml_and_top_level_key(self):
        self.assertManifestError("[[patch]\n", "10-test.toml")
        self.assertManifestError("stray = 1\n", "stray")

    def test_unknown_only_id(self):
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="x"))
        code, _, err = self.run_main("--only", "nope")
        self.assertEqual(code, 2)
        self.assertIn("nope", err)

    def test_path_escape_rejected(self):
        self.assertManifestError(toml_patch(op="replace_fn_body", fn="f", body="x", paths=["../x/*.rs"]), "path")


class PatchTests(Env):
    def test_replace_fn_body(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="fetch_news", impl="App", body="\n    Ok(String::new())\n"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("APPLIED"), out)
        text = self.get("src/lib.rs")
        self.assertIn(
            "    pub fn fetch_news(&self, url: &str) -> Result<String, Error> {\n"
            "        // aurora-local-patch: p1\n"
            "        Ok(String::new())\n"
            "    }\n",
            text,
        )
        self.assertNotIn("http_get", text)
        self.assertIn('println!("other")', text)  # neighbours untouched

    def test_insert_fn_start(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="insert_fn_start", fn="fetch_news", body="if off() { return Ok(String::new()); }"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertIn(
            "-> Result<String, Error> {\n"
            "        // aurora-local-patch: p1\n"
            "        if off() { return Ok(String::new()); }\n"
            "        let body = http_get(url)?;\n",
            self.get("src/lib.rs"),
        )

    def test_insert_fn_start_one_line_fn(self):
        self.put("src/lib.rs", "fn f() { go(); }\n")
        self.manifest(toml_patch(op="insert_fn_start", fn="f", body="early();"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(
            self.get("src/lib.rs"), "fn f() {\n    // aurora-local-patch: p1\n    early();\n     go(); }\n"
        )

    def test_regex_with_groups_and_marker(self):
        self.put("src/lib.rs", LIB)
        self.manifest(
            toml_patch(
                op="regex",
                pattern=r"let body = (\w+)\(url\)\?;",
                replace="let body = \\1(url).unwrap_or_default(); // aurora-local-patch: p1",
            )
        )
        self.assertEqual(self.run_main()[0], 0)
        self.assertIn("let body = http_get(url).unwrap_or_default(); // aurora-local-patch: p1", self.get("src/lib.rs"))

    def test_regex_dotall(self):
        self.put("src/lib.rs", "a(\n1,\n2\n);\n")
        self.manifest(
            toml_patch(op="regex", pattern=r"a\(.*?\);", replace="b(); // aurora-local-patch: p1", dotall=True)
        )
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.get("src/lib.rs"), "b(); // aurora-local-patch: p1\n")

    def test_regex_fn_scope(self):
        self.put("src/lib.rs", "fn a() { x(); }\nfn b() { x(); }\n")
        self.manifest(toml_patch(op="regex", pattern=r"x\(\);", replace="y(); // aurora-local-patch: p1", fn="b"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.get("src/lib.rs"), "fn a() { x(); }\nfn b() { y(); // aurora-local-patch: p1 }\n")

    def test_literal_is_whitespace_tolerant(self):
        self.put("src/lib.rs", "fn f() {\n    call(\n        one,\n        two\n    );\n}\n")
        self.manifest(
            toml_patch(op="literal", find="call( one, two )", replace="// aurora-local-patch: p1\ncall_other();")
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("call_other();", self.get("src/lib.rs"))
        self.assertNotIn("call(", self.get("src/lib.rs"))

    def test_literal_whitespace_matches_crlf(self):
        self.put("src/lib.rs", "call(\r\n    one\r\n);\r\n")
        self.manifest(toml_patch(op="literal", find="call( one );", replace="// aurora-local-patch: p1"))
        self.assertEqual(self.run_main()[0], 0)

    def test_literal_replacement_is_not_a_template(self):
        self.put("src/lib.rs", "a;\n")
        self.manifest(toml_patch(op="literal", find="a;", replace=r"b(r" + '"\\1"' + r"); // aurora-local-patch: p1"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertIn(r"\1", self.get("src/lib.rs"))

    def test_idempotent_second_run(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="other", body="noop();"))
        self.assertEqual(self.run_main()[0], 0)
        after_first = self.get("src/lib.rs")
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("ALREADY"), out)
        self.assertEqual(self.get("src/lib.rs"), after_first)

    def test_hash_marker_for_toml_targets(self):
        # Cargo.lock cannot hold a `//` comment: a `#` marker is accepted and makes the second run ALREADY.
        self.put("Cargo.lock", 'name = "x"\nversion = "1"\n')
        self.manifest(
            toml_patch(
                paths=["Cargo.lock"],
                op="literal",
                find='version = "1"',
                replace='version = "2" # aurora-local-patch: p1',
            )
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("APPLIED"), out)
        self.assertEqual(self.get("Cargo.lock"), 'name = "x"\nversion = "2" # aurora-local-patch: p1\n')
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("ALREADY"), out)

    def test_marker_prefix_id_is_not_already(self):
        self.put("src/lib.rs", "fn f() { x(); } // aurora-local-patch: p1-other\n")
        self.manifest(toml_patch(op="insert_fn_start", fn="f", body="y();"))
        self.assertTrue(self.run_main()[1].startswith("APPLIED"))

    def test_check_writes_nothing(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="other", body="noop();"))
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("APPLIED"))
        self.assertEqual(self.get("src/lib.rs"), LIB)

    def test_all_or_nothing_on_drift(self):
        self.put("src/lib.rs", LIB)
        self.manifest(
            toml_patch(id="good", op="replace_fn_body", fn="other", body="noop();")
            + toml_patch(id="bad", op="replace_fn_body", fn="gone", body="x();")
        )
        code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("APPLIED", out)
        self.assertIn("DRIFT", out)
        self.assertIn("nothing written", err)
        self.assertEqual(self.get("src/lib.rs"), LIB)

    def test_later_patch_sees_earlier_result(self):
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(
            toml_patch(id="one", op="literal", find="a();", replace="b(); // aurora-local-patch: one")
            + toml_patch(id="two", op="literal", find="b();", replace="c(); // aurora-local-patch: two")
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.get("src/lib.rs").count("c();"), 1)

    def test_ambiguity_two_matches(self):
        self.put("src/lib.rs", "fn f() { a(); a(); }\n")
        self.manifest(toml_patch(op="literal", find="a();", replace="// aurora-local-patch: p1"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("2 matches, expected 1", out)
        self.assertEqual(self.get("src/lib.rs"), "fn f() { a(); a(); }\n")

    def test_expect_counts_across_files(self):
        self.put("src/a.rs", "x\n")
        self.put("src/b.rs", "x\n")
        self.manifest(toml_patch(op="literal", find="x", replace="y // aurora-local-patch: p1", expect=2))
        self.assertEqual(self.run_main()[0], 0)
        self.assertIn("y", self.get("src/a.rs") + self.get("src/b.rs"))

    def test_zero_matches_gives_nearest_line(self):
        self.put("src/lib.rs", "fn f() {\n    let value = compute_total(a, b);\n}\n")
        self.manifest(
            toml_patch(op="literal", find="let value = compute_totals(a, b);", replace="// aurora-local-patch: p1")
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("0 matches", out)
        self.assertIn("src/lib.rs:2", out)

    def test_ambiguous_fn_lists_candidates(self):
        self.put("src/a.rs", "fn dup() { 1 }\n")
        self.put("src/b.rs", "fn dup() { 2 }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="dup", body="3"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("2 definitions", out)
        self.assertIn("src/a.rs:1", out)
        self.assertIn("src/b.rs:1", out)

    def test_fn_not_found_suggests_similar_name(self):
        self.put("src/lib.rs", "fn fetch_new() { 1 }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="fetch_news", body="3"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("not found", out)
        self.assertIn("fetch_new (src/lib.rs:1)", out)

    def test_moved_file_is_still_found(self):
        self.put("src/deep/er/moved.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="other", body="noop();", paths=["src/**/*.rs"]))
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("src/deep/er/moved.rs", out)

    def test_impl_filter_selects_one_of_two_same_names(self):
        self.put("src/lib.rs", "impl A { fn run(&self) { a(); } }\nimpl B { fn run(&self) { b(); } }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="run", impl="B", body="z();"))
        self.assertEqual(self.run_main()[0], 0)
        text = self.get("src/lib.rs")
        self.assertIn("a();", text)
        self.assertNotIn("b();", text)

    def test_impl_filter_miss_reports_other_location(self):
        self.put("src/lib.rs", "impl A { fn run(&self) { a(); } }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="run", impl="B", body="z();"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("outside that impl", out)

    def test_expect_sig_drift(self):
        self.put("src/lib.rs", "pub fn f(a: u8, extra: u8) -> u8 { a }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="0", expect_sig=r"fn f\(a: u8\) -> u8"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("signature", out)
        self.assertIn("extra: u8", out)

    def test_expect_sig_match_across_lines(self):
        self.put("src/lib.rs", "pub fn f(\n    a: u8,\n) -> u8 { a }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="0", expect_sig=r"^pub fn f\( a: u8, \) -> u8$"))
        self.assertEqual(self.run_main()[0], 0)

    def test_crlf_and_final_newline_preserved(self):
        self.put("src/lib.rs", "fn f() {\r\n    a();\r\n}\r\n\r\nfn g() {}")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();\nc();"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(
            self.get("src/lib.rs"),
            "fn f() {\r\n    // aurora-local-patch: p1\r\n    b();\r\n    c();\r\n}\r\n\r\nfn g() {}",
        )

    def test_non_utf8_bytes_round_trip(self):
        (self.root / "src").mkdir()
        (self.root / "src/lib.rs").write_bytes(b"// \xff\xfe\nfn f() { a(); }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertTrue((self.root / "src/lib.rs").read_bytes().startswith(b"// \xff\xfe\n"))

    def test_optional_missing(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="gone", body="x();", required=False))
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("OPTIONAL-MISSING"), out)

    def test_disabled_skipped_only_and_skip(self):
        self.put("src/lib.rs", LIB)
        self.manifest(
            toml_patch(id="off", op="replace_fn_body", fn="gone", body="x();", enabled=False)
            + toml_patch(id="one", op="replace_fn_body", fn="other", body="x();")
            + toml_patch(id="two", op="insert_fn_start", fn="fmt", body="y();")
        )
        _, out, _ = self.run_main("--check")
        self.assertIn("DISABLED", out)
        _, out, _ = self.run_main("--check", "--only", "one")
        self.assertRegex(out, r"SKIPPED\s+two")
        self.assertRegex(out, r"APPLIED\s+one")
        _, out, _ = self.run_main("--check", "--skip", "one")
        self.assertRegex(out, r"SKIPPED\s+one")

    def test_no_matching_files_is_drift(self):
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="x();"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("no files match", out)

    def test_manifests_apply_in_filename_order(self):
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(
            toml_patch(id="second", op="literal", find="b();", replace="c(); // aurora-local-patch: second"),
            "20-b.toml",
        )
        self.manifest(
            toml_patch(id="first", op="literal", find="a();", replace="b(); // aurora-local-patch: first"), "10-a.toml"
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertLess(out.index("first"), out.index("second"))

    def test_target_dir_is_never_patched(self):
        self.put("target/debug/gen.rs", "fn f() { a(); }\n")
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();", paths=["**/*.rs"]))
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.get("target/debug/gen.rs"), "fn f() { a(); }\n")

    def test_json_report_and_list(self):
        self.put("src/lib.rs", LIB)
        self.manifest(toml_patch(op="replace_fn_body", fn="other", body="x();"))
        report = Path(self._tmp.name) / "report.json"
        self.assertEqual(self.run_main("--check", "--json", str(report))[0], 0)
        data = json.loads(report.read_text())
        self.assertEqual(data["patches"][0]["status"], "APPLIED")
        code, out, _ = self.run_main("--list")
        self.assertEqual(code, 0)
        self.assertIn("p1", out)


class GuardTests(Env):
    GUARD = (
        '[[guard]]\nid = "egress"\nwhy = "new hosts"\npaths = ["src/**/*.rs"]\n'
        "pattern = 'https?://[^\"\\s]+'\nallow = ['https://example\\.org/.*']\nlevel = \"{level}\"\n"
    )

    def setUp(self):
        super().setUp()
        self.put("src/lib.rs", 'const A: &str = "https://example.org/ok";\nconst B: &str = "https://evil.test/x";\n')

    def test_warn_reports_but_exit_zero(self):
        self.manifest(self.GUARD.format(level="warn"))
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("GUARD-WARN", out)
        self.assertIn("src/lib.rs:2", out)
        self.assertNotIn("src/lib.rs:1", out)

    def test_fail_exits_three_and_writes_nothing(self):
        self.manifest(
            self.GUARD.format(level="fail")
            + toml_patch(op="literal", find="ok", replace="fine // aurora-local-patch: p1")
        )
        before = self.get("src/lib.rs")
        code, out, _ = self.run_main()
        self.assertEqual(code, 3)
        self.assertIn("GUARD-FAIL", out)
        self.assertEqual(self.get("src/lib.rs"), before)

    def test_guard_sees_post_patch_content(self):
        self.manifest(
            self.GUARD.format(level="fail")
            + toml_patch(
                op="literal", find="https://evil.test/x", replace="https://example.org/y // aurora-local-patch: p1"
            )
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)

    def test_drift_beats_guard_exit_code(self):
        self.manifest(self.GUARD.format(level="fail") + toml_patch(op="replace_fn_body", fn="gone", body="x();"))
        self.assertEqual(self.run_main()[0], 1)


class AddFileTests(Env):
    SRC = "// aurora-local-patch: p1\n//! the new module\npub fn added() {}\n"

    def source(self, text=None, rel="files/new.rs"):
        path = self.pdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((self.SRC if text is None else text).encode("utf-8"))

    def add_patch(self, **fields):
        fields = {"op": "add_file", "paths": ["src/new.rs"], "source": "files/new.rs", **fields}
        return toml_patch(**fields)

    def assertManifestError(self, text, *needles):
        self.manifest(text)
        code, _, err = self.run_main()
        self.assertEqual(code, 2, err)
        for needle in needles:
            self.assertIn(needle, err)

    def test_new_file_written_with_mode_0644(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.manifest(self.add_patch())
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("APPLIED"), out)
        self.assertIn("src/new.rs", out)
        self.assertEqual(self.get("src/new.rs"), self.SRC)
        self.assertEqual((self.root / "src/new.rs").stat().st_mode & 0o777, 0o644)
        self.assertEqual(sorted(p.name for p in (self.root / "src").iterdir()), ["lib.rs", "new.rs"])

    def test_second_run_is_already(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.manifest(self.add_patch())
        self.assertEqual(self.run_main()[0], 0)
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("ALREADY"), out)

    def test_changed_source_refreshes_and_keeps_mode(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.manifest(self.add_patch())
        self.assertEqual(self.run_main()[0], 0)
        (self.root / "src/new.rs").chmod(0o640)
        self.source(self.SRC + "pub fn more() {}\n")
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("APPLIED"), out)
        self.assertTrue(self.get("src/new.rs").endswith("pub fn more() {}\n"))
        self.assertEqual((self.root / "src/new.rs").stat().st_mode & 0o777, 0o640)

    def test_foreign_file_is_drift_and_untouched(self):
        self.put("src/lib.rs", LIB)
        self.put("src/new.rs", "// upstream's own file\n")
        self.source()
        self.manifest(self.add_patch())
        code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("DRIFT", out)
        self.assertIn("without", out)
        self.assertIn("nothing written", err)
        self.assertEqual(self.get("src/new.rs"), "// upstream's own file\n")

    def test_missing_parent_folder_is_drift(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.manifest(self.add_patch(paths=["moved/crate/new.rs"]))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("moved/crate", out)
        self.assertFalse((self.root / "moved").exists())

    def test_optional_missing_parent(self):
        self.source()
        self.manifest(self.add_patch(paths=["gone/new.rs"], required=False))
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("OPTIONAL-MISSING"), out)

    def test_check_writes_nothing(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.manifest(self.add_patch())
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("APPLIED"), out)
        self.assertFalse((self.root / "src/new.rs").exists())

    def test_crlf_and_final_newline_preserved(self):
        self.put("src/lib.rs", LIB)
        text = "// aurora-local-patch: p1\r\nfn a() {}\r\nfn b() {}"
        self.source(text)
        self.manifest(self.add_patch())
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual((self.root / "src/new.rs").read_bytes(), text.encode("utf-8"))

    def test_later_patch_and_guard_see_the_new_file(self):
        self.put("src/lib.rs", LIB)
        self.source(self.SRC + 'const U: &str = "https://evil.test/x";\n')
        self.manifest(
            self.add_patch()
            + toml_patch(
                id="p2",
                op="literal",
                find="pub fn added() {}",
                replace="pub fn added() { go(); } // aurora-local-patch: p2",
            )
            + '[[guard]]\nid = "egress"\nwhy = "w"\npaths = ["src/**/*.rs"]\npattern = "https?://[^\\"]+"\nlevel = "warn"\n'
        )
        code, out, _ = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("APPLIED          p2  src/new.rs", out)
        self.assertIn("GUARD-WARN       egress  src/new.rs:4", out)
        self.assertIn("go();", self.get("src/new.rs"))

    def test_manifest_errors(self):
        self.source()
        self.source("// no marker here\n", rel="files/plain.rs")
        self.source("// aurora-local-patch: p1\n", rel="../outside.rs")
        cases = [
            (self.add_patch(source="files/missing.rs"), "not found"),
            (self.add_patch(source="../outside.rs"), "bad source"),
            (self.add_patch(source="/etc/hostname"), "bad source"),
            (self.add_patch(source="files/plain.rs"), "aurora-local-patch: p1"),
            (self.add_patch(paths=["src/a.rs", "src/b.rs"]), "exactly one"),
            (self.add_patch(paths=["src/*.rs"]), "exactly one"),
            (self.add_patch(paths=["target/new.rs"]), "ignored folder"),
            (self.add_patch(fn="f"), "unknown key"),
            (toml_patch(op="add_file", paths=["src/new.rs"]), "missing key"),
        ]
        for text, needle in cases:
            with self.subTest(needle=needle, text=text):
                self.assertManifestError(text, needle)

    def test_symlinked_source_outside_is_refused(self):
        outside = Path(self._tmp.name) / "elsewhere.rs"
        outside.write_text(self.SRC, encoding="utf-8")
        (self.pdir / "files").mkdir()
        (self.pdir / "files/link.rs").symlink_to(outside)
        self.assertManifestError(self.add_patch(source="files/link.rs"), "outside the patches dir")

    def test_non_utf8_source_is_refused(self):
        (self.pdir / "files").mkdir()
        (self.pdir / "files/new.rs").write_bytes(b"// aurora-local-patch: p1\n\xff\n")
        self.assertManifestError(self.add_patch(), "not UTF-8")

    def test_files_dir_is_not_scanned_for_manifests(self):
        self.put("src/lib.rs", LIB)
        self.source()
        self.source("not toml [[[", rel="files/broken.toml")
        self.manifest(self.add_patch())
        self.assertEqual(self.run_main()[0], 0)


class MainBranchGuardTests(Env):
    def git_init(self, branch):
        if shutil.which("git") is None:
            self.skipTest("git not installed")
        subprocess.run(["git", "init", "-q", "-b", branch, str(self.root)], check=True)

    def test_refuses_main_checkout_but_allows_check(self):
        self.git_init("main")
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();"))
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("protected branch", err)
        self.assertEqual(self.get("src/lib.rs"), "fn f() { a(); }\n")
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_refuses_the_patch_branch_checkout(self):
        self.git_init("chore/local-patches")
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();"))
        self.assertEqual(self.run_main()[0], 2)
        self.assertEqual(self.get("src/lib.rs"), "fn f() { a(); }\n")

    def test_other_branch_is_allowed(self):
        self.git_init("build")
        self.put("src/lib.rs", "fn f() { a(); }\n")
        self.manifest(toml_patch(op="replace_fn_body", fn="f", body="b();"))
        self.assertEqual(self.run_main()[0], 0)
        self.assertIn("b();", self.get("src/lib.rs"))


if __name__ == "__main__":
    unittest.main()

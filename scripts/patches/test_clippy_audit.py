"""Tests for clippy_audit.py, on canned cargo JSON. Run from scripts/patches: python3 -m unittest"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import clippy_audit as ca

ROOT = Path("/work/tree")


def diag(code, message, file="crates/aurora-msg/src/lib.rs", line=10, level="warning", primary=True, col=5):
    """One `compiler-message` line of `cargo clippy --message-format=json`."""
    msg = {
        "level": level,
        "message": message,
        "code": {"code": code} if code else None,
        "spans": [{"file_name": file, "line_start": line, "column_start": col, "is_primary": primary}],
    }
    return json.dumps({"reason": "compiler-message", "message": msg})


def finding(lint="clippy::needless_return", file="crates/aurora-msg/src/lib.rs", line=1, message="m", group="style"):
    return ca.Finding(lint, file, line, message, group)


class ParseTests(unittest.TestCase):
    def test_reads_lints_and_skips_noise(self):
        lines = [
            '{"reason":"compiler-artifact","package_id":"x"}',
            "not json at all",
            diag("clippy::needless_return", "unneeded `return` statement"),
            diag(None, "3 warnings emitted"),  # summary: no code
            diag("unused_variables", "unused variable: `x`", line=3),
        ]
        found, errors = ca.parse_messages(lines, ROOT, "style")
        self.assertEqual(errors, [])
        self.assertEqual(
            [(f.lint, f.group) for f in found],
            [("clippy::needless_return", "style"), ("unused_variables", "rustc")][::-1],
        )

    def test_duplicate_reports_of_two_targets_count_once(self):
        line = diag("clippy::needless_return", "unneeded `return` statement")
        found, _ = ca.parse_messages([line, line], ROOT, "style")
        self.assertEqual(len(found), 1)

    def test_two_findings_on_one_line_count_twice(self):
        # two unwrap() calls on one line: a third one must still show up as new
        lines = [diag("clippy::unwrap_used", "m", col=5), diag("clippy::unwrap_used", "m", col=20)]
        found, _ = ca.parse_messages(lines, ROOT, "restriction")
        self.assertEqual(len(found), 2)
        merged, _ = ca.collect("cargo", ROOT, True, False, runner=FakeClippy({}, lines), env={})
        self.assertEqual(len(merged), 2)

    def test_same_message_at_two_places_counts_twice(self):
        lines = [diag("clippy::needless_return", "m", line=1), diag("clippy::needless_return", "m", line=2)]
        found, _ = ca.parse_messages(lines, ROOT, "style")
        self.assertEqual(len(found), 2)

    def test_uses_the_primary_span_and_relative_paths(self):
        msg = {
            "level": "warning",
            "message": "m",
            "code": {"code": "clippy::foo"},
            "spans": [
                {"file_name": "other.rs", "line_start": 1, "column_start": 1, "is_primary": False},
                {"file_name": str(ROOT / "crates/a/src/x.rs"), "line_start": 7, "column_start": 1, "is_primary": True},
            ],
        }
        line = json.dumps({"reason": "compiler-message", "message": msg})
        (f,), _ = ca.parse_messages([line], ROOT, "perf")
        self.assertEqual((f.file, f.line), ("crates/a/src/x.rs", 7))

    def test_compile_errors_are_reported(self):
        _, errors = ca.parse_messages([diag(None, "could not compile `x`", level="error")], ROOT, "style")
        self.assertEqual(errors, ["could not compile `x`"])

    def test_numbers_do_not_change_the_key(self):
        a = finding(message="this function has too many lines (120/100)")
        b = finding(message="this function has too many lines (131/100)", line=99)
        self.assertEqual(a.key, b.key)
        self.assertNotEqual(a.key, finding(file="crates/aurora-net/src/lib.rs").key)


class BaselineTests(unittest.TestCase):
    def baseline(self, findings, strict=False):
        return ca.baseline_document(findings, strict, "clippy 1")

    def test_moved_code_is_not_new(self):
        old = [finding(line=10)]
        moved = [finding(line=250)]
        self.assertEqual(ca.new_findings(moved, self.baseline(old), False), [])

    def test_a_second_occurrence_is_new(self):
        old = [finding(line=10)]
        now = [finding(line=10), finding(line=40)]
        fresh = ca.new_findings(now, self.baseline(old), False)
        self.assertEqual([f.line for f in fresh], [40])

    def test_a_different_file_or_lint_is_new(self):
        old = [finding()]
        now = [finding(), finding(file="crates/aurora-net/src/lib.rs"), finding(lint="clippy::other")]
        self.assertEqual(len(ca.new_findings(now, self.baseline(old), False)), 2)

    def test_fixed_findings_are_silent(self):
        self.assertEqual(ca.new_findings([], self.baseline([finding()]), False), [])

    def test_strict_findings_are_ignored_without_strict(self):
        strict = finding(lint="clippy::unwrap_used", group="restriction")
        self.assertEqual(ca.new_findings([strict], self.baseline([]), False), [])
        self.assertEqual(len(ca.new_findings([strict], self.baseline([]), True)), 1)

    def test_baseline_has_no_line_numbers(self):
        doc = self.baseline([finding(line=77)])
        self.assertNotIn("line", doc["findings"][0])
        self.assertEqual(doc["findings"][0]["count"], 1)

    def test_load_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            self.assertIsNone(ca.load_baseline(path))
            path.write_text(json.dumps(self.baseline([finding()])))
            self.assertEqual(len(ca.load_baseline(path)["findings"]), 1)
            path.write_text("{")
            with self.assertRaises(ca.ToolError):
                ca.load_baseline(path)
            path.write_text('{"version": 99}')
            with self.assertRaises(ca.ToolError):
                ca.load_baseline(path)


class PolicyTests(unittest.TestCase):
    def test_gating(self):
        self.assertTrue(ca.gating(finding(group="correctness"), False))
        self.assertTrue(ca.gating(finding(group="suspicious"), False))
        for group in ("style", "complexity", "perf", "rustc"):
            self.assertFalse(ca.gating(finding(group=group), True))
        unwrap = finding(lint="clippy::unwrap_used", group="restriction")
        self.assertFalse(ca.gating(unwrap, False))
        self.assertTrue(ca.gating(unwrap, True))
        indexing = finding(lint="clippy::indexing_slicing", group="restriction")
        self.assertFalse(ca.gating(indexing, True))

    def test_external_data_crates(self):
        self.assertTrue(ca.is_external_data("crates/aurora-net/src/udp.rs"))
        self.assertFalse(ca.is_external_data("crates/aurora-viewer/src/main.rs"))
        self.assertFalse(ca.is_external_data("crates/aurora-msg/build.rs"))
        self.assertFalse(ca.is_external_data("crates/aurora-net-extra/src/lib.rs"))


class FakeClippy:
    """Stands in for `run_clippy`: answers each pass with canned lines, keyed by the pass label.

    Labels: "gating" (correctness + suspicious together), "default" (no -W flag) or a group name.
    """

    def __init__(self, by_group, strict_lines=()):
        self.by_group = by_group
        self.strict_lines = list(strict_lines)
        self.calls = []

    def __call__(self, cargo, root, env, extra_cargo, clippy_args):
        self.calls.append((extra_cargo, clippy_args))
        if "--lib" in extra_cargo:
            return self.strict_lines, 0
        flags = [a.split("::")[1] for a in clippy_args if a.startswith("-Wclippy::")]
        label = "gating" if flags == ["correctness", "suspicious"] else flags[0] if flags else "default"
        return self.by_group.get(label, []), 0


class CollectAndMainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "Cargo.toml").write_text("[workspace]\n")
        self.baseline = self.root / "baseline.json"

    def run_main(self, fake, *args):
        argv = ["--root", str(self.root), "--baseline", str(self.baseline), *args]
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(ca, "find_cargo", return_value="cargo"),
            mock.patch.object(ca, "toolchain_env", return_value={}),
            mock.patch.object(ca, "clippy_version", return_value="clippy 1"),
            mock.patch.object(ca, "run_clippy", fake),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = ca.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_quick_mode_has_two_passes_and_gating_wins(self):
        both = diag("clippy::x", "m")  # a suspicious lint is also in the default set
        fake = FakeClippy({"gating": [both], "default": [both, diag("clippy::y", "m", line=2)]})
        found, _ = ca.collect("cargo", self.root, False, False, runner=fake, env={})
        self.assertEqual({f.lint: f.group for f in found}, {"clippy::x": ca.GATING_LABEL, "clippy::y": "default"})
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0][0], ["--all-targets"])
        self.assertIn("dead-code", fake.calls[0][1])

    def test_groups_mode_has_one_pass_per_group(self):
        fake = FakeClippy({"suspicious": [diag("clippy::x", "m")], "style": [diag("clippy::y", "m", line=2)]})
        found, _ = ca.collect("cargo", self.root, False, False, groups=True, runner=fake, env={})
        self.assertEqual({f.lint: f.group for f in found}, {"clippy::x": "suspicious", "clippy::y": "style"})
        self.assertEqual(len(fake.calls), len(ca.GROUPS))

    def test_dead_code_flag(self):
        fake = FakeClippy({})
        ca.collect("cargo", self.root, False, True, runner=fake, env={})
        self.assertNotIn("dead-code", fake.calls[0][1])

    def test_lints_allowed_in_cargo_toml_are_dropped(self):
        (self.root / "Cargo.toml").write_text('[workspace.lints.clippy]\ntoo_many_arguments = "allow"\n')
        crate = self.root / "crates" / "aurora-x"
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text('[lints.clippy]\ntype_complexity = { level = "allow" }\n')
        raw = [
            diag("clippy::too_many_arguments", "m", file="crates/aurora-y/src/a.rs"),
            diag("clippy::type_complexity", "m", file="crates/aurora-x/src/a.rs"),
            diag("clippy::type_complexity", "m", file="crates/aurora-y/src/a.rs", line=2),
        ]
        found, _ = ca.collect("cargo", self.root, False, False, runner=FakeClippy({"default": raw}), env={})
        self.assertEqual([(f.lint, f.file) for f in found], [("clippy::type_complexity", "crates/aurora-y/src/a.rs")])

    def test_rustc_lint_repeated_by_every_pass_counts_once(self):
        same = {g: [diag("unused_variables", "unused variable: `x`")] for g in ("gating", "default")}
        found, _ = ca.collect("cargo", self.root, False, False, runner=FakeClippy(same), env={})
        self.assertEqual([(f.lint, f.group) for f in found], [("unused_variables", "rustc")])

    def test_strict_keeps_only_external_data_crates(self):
        strict = [
            diag("clippy::unwrap_used", "used `unwrap()`", file="crates/aurora-net/src/a.rs"),
            diag("clippy::unwrap_used", "used `unwrap()`", file="crates/aurora-viewer/src/a.rs"),
        ]
        found, _ = ca.collect("cargo", self.root, True, False, runner=FakeClippy({}, strict), env={})
        self.assertEqual([f.file for f in found], ["crates/aurora-net/src/a.rs"])

    def test_clean_run_exits_zero_then_new_correctness_fails(self):
        old = FakeClippy({"default": [diag("clippy::y", "m")]})
        code, out, _ = self.run_main(old, "--update-baseline")
        self.assertEqual(code, ca.EXIT_OK)
        self.assertTrue(self.baseline.exists())

        code, out, _ = self.run_main(old)
        self.assertEqual(code, ca.EXIT_OK)
        self.assertIn("0 new, 0 failing", out)

        worse = FakeClippy({"default": [diag("clippy::y", "m")], "gating": [diag("clippy::bad", "wrong", line=3)]})
        code, out, _ = self.run_main(worse)
        self.assertEqual(code, ca.EXIT_NEW)
        self.assertIn("FAIL  correctness/suspicious", out)

    def test_new_style_finding_is_reported_but_passes(self):
        self.run_main(FakeClippy({}), "--update-baseline")
        code, out, _ = self.run_main(FakeClippy({"default": [diag("clippy::y", "m")]}))
        self.assertEqual(code, ca.EXIT_OK)
        self.assertIn("new   default", out)

    def test_strict_needs_a_strict_baseline(self):
        self.run_main(FakeClippy({}), "--update-baseline")
        code, _, err = self.run_main(FakeClippy({}), "--strict")
        self.assertEqual(code, ca.EXIT_TOOL)
        self.assertIn("--update-baseline --strict", err)

    def test_strict_fails_on_a_new_unwrap(self):
        self.run_main(FakeClippy({}), "--update-baseline", "--strict")
        unwrap = diag("clippy::unwrap_used", "used `unwrap()`", file="crates/aurora-llsd/src/a.rs")
        code, out, _ = self.run_main(FakeClippy({}, [unwrap]), "--strict")
        self.assertEqual(code, ca.EXIT_NEW)
        self.assertIn("clippy::unwrap_used", out)
        # the same finding without --strict is not looked at
        code, _, _ = self.run_main(FakeClippy({}, [unwrap]))
        self.assertEqual(code, ca.EXIT_OK)

    def test_missing_baseline_and_compile_errors_are_tool_errors(self):
        code, _, err = self.run_main(FakeClippy({}))
        self.assertEqual(code, ca.EXIT_TOOL)
        self.assertIn("--update-baseline", err)
        broken = FakeClippy({"default": [diag(None, "could not compile `x`", level="error")]})
        code, _, err = self.run_main(broken, "--update-baseline")
        self.assertEqual(code, ca.EXIT_TOOL)
        self.assertIn("does not compile", err)

    def test_summary_groups_by_lint(self):
        text = ca.summarise([finding(), finding(line=2), finding(lint="clippy::z", group="perf")])
        self.assertIn("style", text)
        self.assertIn("2  clippy::needless_return", text)


if __name__ == "__main__":
    unittest.main()

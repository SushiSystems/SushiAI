# test_mutate_diff.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests tools/testing/mutate_diff.py: its core on strings, its shell once on a temporary repository."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.mutate_diff import (  # noqa: E402
    Mutation,
    Result,
    apply_mutation,
    format_result,
    format_summary,
    imports_any,
    is_mutable_path,
    main,
    module_names,
    mutate_line,
    parse_changed_lines,
)

K_DIFF = (
    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    "@@ -3 +3,2 @@ def f():\n+x\n+y\n"
    "@@ -9,2 +10,0 @@\n"
    "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n"
    "diff --git a/dir/b.py b/dir/b.py\n--- a/dir/b.py\n+++ b/dir/b.py\n@@ -0,0 +1,3 @@\n"
)


class TestCore(unittest.TestCase):
    """Checks the pure functions on in-memory text."""
    K_TEST_SIZE = "small"

    def test_parse_changed_lines_reads_added_ranges_and_skips_deletions(self) -> None:
        self.assertEqual(
            parse_changed_lines(K_DIFF),
            {"a.py": frozenset({3, 4}), "dir/b.py": frozenset({1, 2, 3})},
        )

    def test_is_mutable_path_rejects_tests_and_other_languages(self) -> None:
        self.assertTrue(is_mutable_path("tools/testing/x.py"))
        self.assertFalse(is_mutable_path("tools/tests/helper.py"))
        self.assertFalse(is_mutable_path("tools/test_x.py"))
        self.assertFalse(is_mutable_path("tools/x.cpp"))

    def test_mutate_line_picks_one_mutation_per_kind(self) -> None:
        rows = (
            ("a = x == 1", "== to !="),
            ("a = x < y", "< to >="),
            ("a = x <= y", "<= to >"),
            ("a = x is None", "is to is not"),
            ("a = x not in y", "not in to in"),
            ("a = x and y", "and to or"),
            ("a = x or y", "or to and"),
            ("a = not x", "not dropped"),
            ("a = True", "True to False"),
            ("a = f(7)", "7 to 8"),
            ("a = f(0.5)", "0.5 to 1.5"),
        )
        for text, label in rows:
            with self.subTest(text=text):
                outcome = mutate_line(f"{text}\n", 1)
                self.assertIsNotNone(outcome)
                self.assertEqual(outcome[0].label, label)

    def test_mutate_line_produces_the_mutated_text(self) -> None:
        source = "def f(x):\n    if x == 1 and x != 2:\n        return x\n"
        self.assertEqual(
            mutate_line(source, 2)[1], "def f(x):\n    if x != 1 and x != 2:\n        return x\n")
        self.assertEqual(
            mutate_line(source, 3)[1], "def f(x):\n    if x == 1 and x != 2:\n        return None\n")

    def test_mutate_line_prefers_the_comparison_over_later_kinds(self) -> None:
        outcome = mutate_line("def f(x):\n    return x > 3 and not x\n", 2)
        self.assertEqual(outcome[0].label, "> to <=")

    def test_mutate_line_keeps_non_ascii_text_before_the_span(self) -> None:
        outcome = mutate_line('a = "é" if x == 1 else "ü"\n', 1)
        self.assertEqual(outcome[1], 'a = "é" if x != 1 else "ü"\n')

    def test_mutate_line_skips_what_proof_of_value_skips(self) -> None:
        source = (
            '"""Doc with 1 == 1."""\n'
            "import os\n"
            "from a import b\n"
            "K_LOGGER.warning('n=%d', 5)\n"
            "logger.info('%s', x == 1)\n"
            "values.reserve(10)\n"
            "# a == b\n"
            "\n"
        )
        for line in range(1, 9):
            with self.subTest(line=line):
                self.assertIsNone(mutate_line(source, line))

    def test_mutate_line_skips_a_return_of_none_and_a_broken_source(self) -> None:
        self.assertIsNone(mutate_line("def f():\n    return None\n", 2))
        self.assertIsNone(mutate_line("def f(:\n", 1))

    def test_mutate_line_covers_a_continuation_line_of_a_statement(self) -> None:
        source = "a = foo(\n    x == 1,\n)\n"
        self.assertEqual(mutate_line(source, 2)[0].label, "== to !=")

    def test_apply_mutation_replaces_only_the_named_span(self) -> None:
        mutation = Mutation(2, 4, 5, "9", "x")
        self.assertEqual(apply_mutation("a\nb = 1\nc\n", mutation), "a\nb = 9\nc\n")

    def test_module_names_lists_every_suffix_and_drops_init(self) -> None:
        self.assertEqual(
            module_names("tools/testing/mutate_diff.py"),
            ("tools.testing.mutate_diff", "testing.mutate_diff", "mutate_diff"),
        )
        self.assertEqual(module_names("pkg/__init__.py"), ("pkg",))

    def test_imports_any_reads_each_import_form(self) -> None:
        names = ("testing.mutate_diff", "mutate_diff")
        self.assertTrue(imports_any("import testing.mutate_diff\n", names))
        self.assertTrue(imports_any("from testing.mutate_diff import main\n", names))
        self.assertTrue(imports_any("from testing import mutate_diff\n", names))
        self.assertFalse(imports_any("from testing.other import main\n", names))
        self.assertFalse(imports_any("def broken(:\n", names))

    def test_format_lines_follow_the_output_contract(self) -> None:
        results = [
            Result("a.py", Mutation(4, 0, 1, "", "== to !="), True),
            Result("a.py", Mutation(9, 0, 1, "", "7 to 8"), False),
        ]
        self.assertEqual(format_result(results[0]), "a.py:4 == to != killed")
        self.assertEqual(format_result(results[1]), "a.py:9 7 to 8 survived")
        self.assertEqual(format_summary(results), "mutants: 2, killed: 1, survived: 1")


K_CALC = "def add(a, b):\n    return a + b\n\n\ndef untested(x):\n    return x * 2\n"
K_CALC_TEST = (
    "import sys\nimport unittest\nfrom pathlib import Path\n\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n"
    "from calc import add\n\n\n"
    "class TestAdd(unittest.TestCase):\n"
    "    def test_add_sums(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n"
)


def _git(root: Path, *arguments: str) -> None:
    """Runs a git command in a temporary repository with a fixed identity."""
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "-C", str(root), *identity, *arguments], check=True, capture_output=True)


class TestShell(unittest.TestCase):
    """Checks the command line on a temporary repository with one function and one test."""
    K_TEST_SIZE = "medium"

    def test_main_reports_one_killed_and_one_survived_mutant(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _git(root, "init", "-q")
            (root / "README.md").write_text("start\n", encoding="utf-8")
            _git(root, "add", ".")
            _git(root, "commit", "-q", "-m", "start")
            (root / "tests").mkdir()
            (root / "calc.py").write_text(K_CALC, encoding="utf-8")
            (root / "tests" / "test_calc.py").write_text(K_CALC_TEST, encoding="utf-8")
            _git(root, "add", ".")
            _git(root, "commit", "-q", "-m", "add calc")
            before = (root / "calc.py").read_bytes()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main([str(root), "--base", "HEAD~1", "--tests-dir", "tests", "--timeout", "60"])
            self.assertEqual((code, err.getvalue()), (0, ""))
            self.assertEqual(
                out.getvalue().splitlines(),
                [
                    "calc.py:2 return value to return None killed",
                    "calc.py:6 2 to 3 survived",
                    "mutants: 2, killed: 1, survived: 1",
                ],
            )
            self.assertEqual((root / "calc.py").read_bytes(), before)

    def test_main_fails_with_a_reason_outside_a_repository(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main([folder])
            self.assertNotEqual(code, 0)
            self.assertIn("mutate_diff:", err.getvalue())


if __name__ == "__main__":
    unittest.main()

# test_run_suite.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that run_suite runs every test file of a folder and reports only the failed ones."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking import run_suite  # noqa: E402

K_PASSING = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
K_FAILING = K_PASSING.replace("pass", "self.fail('boom')")
K_EMPTY = "x = 1\n"


class TestFileNamesTest(unittest.TestCase):
    """Lists the test files of a folder."""

    K_TEST_SIZE = "medium"

    def test_names_are_distinct_sorted_and_found_in_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sub").mkdir()
            for name in ("test_b.py", "test_a.py", "sub/test_a.py", "helper.py"):
                (root / name).write_text(K_EMPTY, encoding="utf-8")
            self.assertEqual(run_suite.test_file_names(root), ["test_a.py", "test_b.py"])


class RunFolderTest(unittest.TestCase):
    """Runs a folder of real test files."""

    K_TEST_SIZE = "medium"

    def run_files(self, files: dict[str, str]) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in files.items():
                (root / name).write_text(text, encoding="utf-8")
            return run_suite.run_folder(sys.executable, root)

    def test_passing_files_return_zero_and_print_nothing(self) -> None:
        code, output = self.run_files({"test_a.py": K_PASSING, "test_b.py": K_PASSING})
        self.assertEqual((code, output), (0, ""))

    def test_a_failing_file_returns_one_and_is_named_with_its_output(self) -> None:
        code, output = self.run_files({"test_a.py": K_PASSING, "test_b.py": K_FAILING})
        self.assertEqual((code, "== test_b.py" in output, "boom" in output,
                          "== test_a.py" in output), (1, True, True, False))

    def test_a_file_that_cannot_be_imported_fails(self) -> None:
        code, _ = self.run_files({"test_a.py": "raise RuntimeError('no')\n"})
        self.assertEqual(code, 1)

    def test_a_file_without_a_test_is_no_failure(self) -> None:
        code, _ = self.run_files({"test_a.py": K_EMPTY})
        self.assertEqual(code, 0)


class MainTest(unittest.TestCase):
    """Covers the command line."""

    K_TEST_SIZE = "small"

    def test_no_folder_is_a_usage_error(self) -> None:
        self.assertEqual(run_suite.main([]), 2)


if __name__ == "__main__":
    unittest.main()

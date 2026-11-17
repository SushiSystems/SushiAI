# test_tests_tree.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests tools/checking/tests_tree.py: which folders it takes for test folders."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.tests_tree import discover  # noqa: E402

K_TEST_SIZE = "medium"


def found(files: list[str]) -> list[str]:
    """Returns the files discover() takes from a tree holding the given files, as posix paths."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for relative in files:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text("x", encoding="utf-8")
        return [path.relative_to(root).as_posix() for path in discover(root).files]


class TestsTreeTest(unittest.TestCase):
    """Reads the folders named like a test folder, and the test folder of a Unity project."""

    def test_takes_a_tests_and_a_dunder_tests_folder_at_any_depth(self) -> None:
        self.assertEqual(found(["a/tests/t.py", "web/__tests__/c.ts", "src/s.py"]),
                         ["a/tests/t.py", "web/__tests__/c.ts"])

    def test_takes_the_tests_folder_of_a_unity_project_below_assets(self) -> None:
        self.assertEqual(found(["Assets/Game/Tests/Features/A/ATests.cs", "Assets/Game/Runtime/B.cs"]),
                         ["Assets/Game/Tests/Features/A/ATests.cs"])

    def test_leaves_a_capitalised_tests_folder_elsewhere_alone(self) -> None:
        self.assertEqual(found(["Assets/Game/Runtime/Tests/B.cs", "Tests/C.cs", "Assets/Tests/D.cs"]), [])

    def test_takes_a_file_once_when_test_folders_nest(self) -> None:
        self.assertEqual(found(["Assets/Game/Tests/tests/E.cs"]), ["Assets/Game/Tests/tests/E.cs"])


if __name__ == "__main__":
    unittest.main()

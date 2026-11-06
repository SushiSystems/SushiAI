# test_check_workflows.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rules of tools/ci/check_workflows.py on workflow text held in memory, and its command line on a temporary tree."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from ci.check_workflows import K_CHECKER, rule_duplicate_key  # noqa: E402

K_WORKFLOW = Path(".github/workflows/ci.yml")
K_DUPLICATE_IF = (
    "jobs:\n  a:\n    steps:\n      - uses: x\n        if: one\n        with:\n          k: v\n"
    "        if: two\n      - name: n\n        if: three\n"
)


def _found(text: str) -> list[tuple[str, int]]:
    """Returns the message and line of each duplicate-key issue a workflow text yields."""
    return [(issue.message, issue.line) for issue in rule_duplicate_key(SourceFile(K_WORKFLOW, text))]


class DuplicateKeyTest(unittest.TestCase):
    """Covers a mapping key written twice in one block mapping, and the lines that are not keys."""
    K_TEST_SIZE = "small"

    def test_reports_a_key_written_twice_in_one_step(self) -> None:
        self.assertEqual(_found(K_DUPLICATE_IF), [("key 'if' appears twice in one mapping", 8)])

    def test_the_same_key_in_separate_steps_and_jobs_is_not_a_duplicate(self) -> None:
        text = (
            "jobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: x\n        with:\n          k: v\n"
            "      - uses: y\n        with:\n          k: v\n  b:\n    runs-on: x\n"
        )
        self.assertEqual(_found(text), [])

    def test_a_block_scalar_body_and_a_comment_are_not_keys(self) -> None:
        text = (
            "jobs:\n  a:\n    steps:\n      - name: n\n        run: |\n          if: not a key\n"
            "          if: still text\n        # if: commented\n        if: real\n"
        )
        self.assertEqual(_found(text), [])


class ShellTest(unittest.TestCase):
    """Runs the command line on a tree on disk, so the folder and the file suffixes the shell reads reach the rule."""
    K_TEST_SIZE = "medium"

    def test_cli_reads_the_workflow_files_of_the_folder_only(self) -> None:
        """Reports the duplicate in a workflow file, and reads nothing from a root without the folder or from other files."""
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / "repo"
            bare = Path(folder) / "bare"
            for path, text in {
                repo / ".github" / "workflows" / "ci.yml": K_DUPLICATE_IF,
                repo / ".github" / "workflows" / "notes.txt": K_DUPLICATE_IF,
                bare / "README.md": "no workflows here\n",
            }.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [str(repo), str(bare)])
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue().splitlines(), [
                ".github/workflows/ci.yml:8: rule_duplicate_key: key 'if' appears twice in one mapping",
            ])


if __name__ == "__main__":
    unittest.main()

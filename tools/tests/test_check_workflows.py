# test_check_workflows.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/ci/check_workflows.py on workflow files built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.check_workflows import collect, rule_duplicate_key  # noqa: E402


class WorkflowsTest(unittest.TestCase):
    """Builds a repository with a workflow folder for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def found(self, text: str) -> list[tuple[str, int]]:
        """Returns the message and line of each duplicate-key issue in a workflow."""
        path = self.root / ".github" / "workflows" / "ci.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return [(issue.message, issue.line) for workflow in collect([self.root]) for issue in rule_duplicate_key(workflow)]

    def test_reports_a_key_written_twice_in_one_step(self) -> None:
        text = (
            "jobs:\n  a:\n    steps:\n      - uses: x\n        if: one\n        with:\n          k: v\n"
            "        if: two\n      - name: n\n        if: three\n"
        )
        self.assertEqual(self.found(text), [("key 'if' appears twice in one mapping", 8)])

    def test_the_same_key_in_separate_steps_and_jobs_is_not_a_duplicate(self) -> None:
        text = (
            "jobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: x\n        with:\n          k: v\n"
            "      - uses: y\n        with:\n          k: v\n  b:\n    runs-on: x\n"
        )
        self.assertEqual(self.found(text), [])

    def test_a_block_scalar_body_and_a_comment_are_not_keys(self) -> None:
        text = (
            "jobs:\n  a:\n    steps:\n      - name: n\n        run: |\n          if: not a key\n"
            "          if: still text\n        # if: commented\n        if: real\n"
        )
        self.assertEqual(self.found(text), [])

    def test_a_workflow_without_the_folder_yields_nothing(self) -> None:
        self.assertEqual(list(collect([self.root])), [])


if __name__ == "__main__":
    unittest.main()

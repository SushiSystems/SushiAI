# test_install_hooks.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the hook installer: what it asks git and reports on values in memory, and git once on a real repository."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hooks import install_hooks  # noqa: E402
from hooks.install_hooks import Outcome, hooks_path_command, outcome_of, root_of  # noqa: E402
from repo_builder import Repo  # noqa: E402

K_ROOT = Path("some") / "repo"


class InstallDecisionTest(unittest.TestCase):
    """Covers the command, the root and the report, which need no git."""

    K_TEST_SIZE = "small"

    def test_the_command_sets_the_hooks_path_of_the_given_repository(self) -> None:
        self.assertEqual(hooks_path_command(K_ROOT),
                         ["git", "-C", str(K_ROOT), "config", "core.hooksPath", "tools/hooks"])

    def test_the_root_is_the_first_argument_or_the_current_folder(self) -> None:
        current = Path("here")
        self.assertEqual(root_of(["other", "ignored"], current), Path("other"))
        self.assertEqual(root_of([], current), current)

    def test_an_accepted_path_reports_on_stdout_and_exits_clean(self) -> None:
        self.assertEqual(outcome_of(K_ROOT, True), Outcome(0, "git now runs the hooks in tools/hooks", ""))

    def test_a_refused_path_reports_on_stderr_and_fails(self) -> None:
        self.assertEqual(outcome_of(K_ROOT, False),
                         Outcome(1, "", f"cannot set core.hooksPath in {K_ROOT}; is it a git repository?"))


class InstallTest(unittest.TestCase):
    """Runs the installer against git on a real repository and on a folder that is none."""

    K_TEST_SIZE = "medium"

    def test_install_sets_the_hooks_path_of_a_repository(self) -> None:
        repo = Repo()
        self.addCleanup(repo.close)
        self.assertTrue(install_hooks.install(repo.root))
        self.assertEqual(repo.git("config", "core.hooksPath").strip(), "tools/hooks")

    def test_main_reports_the_hooks_path_it_set(self) -> None:
        repo = Repo()
        self.addCleanup(repo.close)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(install_hooks.main([str(repo.root)]), 0)
        self.assertEqual(out.getvalue(), "git now runs the hooks in tools/hooks\n")

    def test_reports_a_folder_that_is_no_repository(self) -> None:
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(errors):
            self.assertEqual(install_hooks.main([folder]), 1)
        self.assertIn("is it a git repository?", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

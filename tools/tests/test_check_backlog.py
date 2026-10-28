# test_check_backlog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the backlog checker, which runs sushicore's `backlog check`, and the bridge it loads it through."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation import sushicore_bridge  # noqa: E402
from documentation.check_backlog import K_CHECKER, Repository, rule_backlog  # noqa: E402

K_HAS_SUSHICORE = sushicore_bridge.load("sushicore.backlog", Path(__file__).resolve().parents[2]) is not None


def _repository(folder: str, files: dict[str, str]) -> Repository:
    """Writes the files below a temporary root and returns it as a repository."""
    root = Path(folder)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root)


class RegistrationTest(unittest.TestCase):
    """Covers the rule table."""

    def test_names_the_one_rule(self) -> None:
        """The rules of `backlog check` run as one."""
        self.assertEqual(sorted(K_CHECKER.rules), ["rule_backlog"])


class WithoutBacklogTest(unittest.TestCase):
    """Covers a repository with nothing to check."""

    def test_reports_nothing_without_a_store_or_a_retired_file(self) -> None:
        """A repository without a backlog is not asked for sushicore."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"docs/README.md": "# Docs\n"})
            with mock.patch.object(sushicore_bridge, "_import", return_value=None):
                self.assertEqual(list(rule_backlog(repository)), [])


class MissingSushicoreTest(unittest.TestCase):
    """Covers a repository whose checker cannot load sushicore."""

    def test_reports_once_that_the_rules_could_not_run(self) -> None:
        """One issue names the package and the version to install."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"docs/backlog/backlog.toml": "[backlog]\n"})
            with mock.patch("documentation.check_backlog.load", return_value=None):
                issues = list(rule_backlog(repository))
        self.assertEqual(len(issues), 1)
        self.assertIn("install sushicore 0.10.0", issues[0].message)


@unittest.skipUnless(K_HAS_SUSHICORE, "sushicore is not installed")
class DelegationTest(unittest.TestCase):
    """Covers the faults `backlog check` returns, read through the checker."""

    def test_reports_a_retired_file(self) -> None:
        """A repository that still holds REMAINING_WORK.md is sent to `backlog import`."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"docs/design/REMAINING_WORK.md": "# Remaining work\n"})
            messages = [issue.message for issue in rule_backlog(repository)]
        self.assertEqual(messages, ["the store in docs/backlog/ replaced this file; run `backlog import`"])

    def test_reports_nothing_for_a_store_the_tool_wrote(self) -> None:
        """A store written by the commands checks clean."""
        from sushicore.backlog import BacklogEditor, BacklogStore

        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"docs/README.md": "# Docs\n"})
            editor = BacklogEditor(BacklogStore(repository.root))
            editor.add_area("BUILD")
            editor.add_work(area="BUILD", priority="P1", title="Fix the gate", done_when="it passes")
            self.assertEqual(list(rule_backlog(repository)), [])


if __name__ == "__main__":
    unittest.main()

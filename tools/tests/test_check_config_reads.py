# test_check_config_reads.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rule of tools/provision/check_config_reads.py on scopes held in memory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provision.check_config_reads import Scope, declared_names, rule_undeclared_read  # noqa: E402

K_CONFIG = '''
class ProvisionConfig(Protocol):
    platform: str
    cmake_exe: str

    @property
    def is_windows(self) -> bool:
        """Report the platform."""

    def expand(self, value: str) -> str:
        """Expand a value."""


class Other:
    stray: str
'''


def _issues(source: str) -> list[str]:
    """Returns the messages the rule gives for one provision file holding *source*."""
    scope = Scope(declared_names(K_CONFIG), ((Path("probe.py"), source),))
    return [issue.message for issue in rule_undeclared_read(scope)]


class DeclaredNamesTest(unittest.TestCase):
    """Tests which names the protocol class declares."""

    K_TEST_SIZE = "small"

    def test_fields_properties_and_methods_are_declared_and_other_classes_are_not(self) -> None:
        """Checks that the names come from the protocol class alone."""
        self.assertEqual(declared_names(K_CONFIG), {"platform", "cmake_exe", "is_windows", "expand"})


class UndeclaredReadTest(unittest.TestCase):
    """Tests the rule that flags a config read the protocol does not declare."""

    K_TEST_SIZE = "small"

    def test_a_declared_read_is_not_reported(self) -> None:
        """Checks that a read of a declared field gives no issue."""
        self.assertEqual(_issues("x = cfg.platform\ny = cfg.expand(z)\n"), [])

    def test_an_undeclared_read_is_reported_with_its_name(self) -> None:
        """Checks that a read of a name the protocol lacks gives one issue naming it."""
        self.assertIn("mystery_exe", " ".join(_issues("x = cfg.mystery_exe\n")))


if __name__ == "__main__":
    unittest.main()

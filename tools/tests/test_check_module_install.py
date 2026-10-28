# test_check_module_install.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of the module-install checker on modules written to disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.check_module_install import K_CHECKER, collect  # noqa: E402

K_INSTALL = "# Install\n\n## Standalone\n\nx\n\n## With SushiHub\n\ny\n"


class ModuleInstallTest(unittest.TestCase):
    """Covers each rule with one failing and one passing module."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _module(self) -> None:
        """Writes the three elements of a module that follows the skill."""
        self._write("sushi-module.toml", "[module]\n")
        self._write("cli/sushistack.deps.toml", "[module]\n")
        self._write("docs/getting_started/INSTALL.md", K_INSTALL)

    def _rules(self) -> list[str]:
        """Returns the name of every rule that fires on the module, once per issue."""
        return [
            name for module in collect([self.root])
            for name, rule in K_CHECKER.rules.items() for _ in rule(module)
        ]

    def test_a_module_that_follows_the_skill_is_clean(self) -> None:
        self._module()
        self.assertEqual(self._rules(), [])

    def test_a_repository_that_is_not_a_module_is_skipped(self) -> None:
        self._write("README.md", "# Not a module\n")
        self.assertEqual(self._rules(), [])

    def test_a_missing_element(self) -> None:
        self._module()
        (self.root / "cli/sushistack.deps.toml").unlink()
        self.assertEqual(self._rules(), ["rule_elements"])

    def test_an_install_page_without_the_hub_heading(self) -> None:
        self._module()
        self._write("docs/getting_started/INSTALL.md", "# Install\n\n## Standalone\n")
        self.assertEqual(self._rules(), ["rule_install_headings"])

    def test_a_machine_path_in_a_config_file(self) -> None:
        self._module()
        self._write("CMakePresets.json", '{"CMAKE_CXX_COMPILER": "D:/Projects/stack/bin/clang++.exe"}\n')
        self.assertEqual(self._rules(), ["rule_machine_path"])

    def test_a_machine_path_in_a_comment_or_a_page_is_allowed(self) -> None:
        self._module()
        self._write("cli/config.toml", "# vcpkg_root = \"C:/Users/me/vcpkg\"\n")
        self._write("docs/guide.md", "Edit D:/Projects/data.\n")
        self.assertEqual(self._rules(), [])

    def test_a_cmake_file_that_reads_the_workspace(self) -> None:
        self._module()
        self._write("cmake/Vcpkg.cmake", 'set(VCPKG_ROOT "$ENV{SUSHISTACK_HOME}/dependencies/vcpkg")\n')
        self.assertEqual(self._rules(), ["rule_no_hub_in_code"])

    def test_a_message_that_names_a_hub_command(self) -> None:
        self._module()
        self._write("cli/pkg/doctor.py", 'FIX = "run hub install"\n')
        self.assertEqual(self._rules(), ["rule_no_hub_in_code"])

    def test_a_comment_or_a_docstring_may_name_hub(self) -> None:
        self._module()
        self._write("cli/pkg/doctor.py", '"""Written by `hub install`."""\n\n# `hub add` clones it.\nFIX = 1\n')
        self._write("cmake/Vcpkg.cmake", "# SUSHISTACK_HOME is read by the CLI.\n")
        self.assertEqual(self._rules(), [])


if __name__ == "__main__":
    unittest.main()

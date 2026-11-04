# test_check_module_install.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests each rule of the module-install checker on values, and its shell on modules written to disk."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.source_file import SourceFile  # noqa: E402
from documentation.check_module_install import (  # noqa: E402
    K_CHECKER,
    K_SETTINGS as CHECKER_SETTINGS,
    Module,
    ModuleSettings,
    collect,
    read_modules,
    rule_install_headings,
    rule_machine_path,
)

K_ROOT = Path("acme")
K_INSTALL_PAGE = "docs/getting_started/INSTALL.md"
K_INSTALL = "# Install\n\n## Standalone\n\nx\n\n## With AcmeHub\n\ny\n"
K_WORKSPACE_READ = r"\bacmehub (?:install|add)\b|ACME_HOME"
K_SETTINGS = ModuleSettings(
    manifest="acme-module.toml",
    fragment="cli/acme.deps.toml",
    install_page=K_INSTALL_PAGE,
    headings=("## Standalone", "## With AcmeHub"),
    workspace_read=K_WORKSPACE_READ,
)
K_NO_WORKSPACE = dataclasses.replace(K_SETTINGS, workspace_read="")
K_CLEAN = {
    "acme-module.toml": "[module]\n",
    "cli/acme.deps.toml": "[module]\n",
    K_INSTALL_PAGE: K_INSTALL,
}
# Each case: its name, the files below the root with their texts, the rules that must fire, the settings.
K_CASES = (
    ("a_module_that_follows_the_skill_is_clean", K_CLEAN, [], K_SETTINGS),
    ("a_missing_element", {name: text for name, text in K_CLEAN.items() if name != "cli/acme.deps.toml"},
     ["rule_elements"], K_SETTINGS),
    ("an_install_page_without_the_workspace_heading", {**K_CLEAN, K_INSTALL_PAGE: "# Install\n\n## Standalone\n"},
     ["rule_install_headings"], K_SETTINGS),
    ("a_machine_path_in_a_config_file",
     {**K_CLEAN, "CMakePresets.json": '{"CMAKE_CXX_COMPILER": "D:/Projects/stack/bin/clang++.exe"}\n'},  # agnostic: allow
     ["rule_machine_path"], K_SETTINGS),
    ("a_machine_path_in_a_comment_or_a_page_is_allowed",
     {**K_CLEAN, "cli/config.toml": '# vcpkg_root = "C:/Users/me/vcpkg"\n',  # agnostic: allow
      "docs/guide.md": "Edit D:/Projects/data.\n"},  # agnostic: allow
     [], K_SETTINGS),
    ("a_cmake_file_that_reads_the_workspace",
     {**K_CLEAN, "cmake/Vcpkg.cmake": 'set(VCPKG_ROOT "$ENV{ACME_HOME}/dependencies/vcpkg")\n'},
     ["rule_no_workspace_in_code"], K_SETTINGS),
    ("a_message_that_names_a_workspace_command",
     {**K_CLEAN, "cli/pkg/doctor.py": 'FIX = "run acmehub install"\n'},
     ["rule_no_workspace_in_code"], K_SETTINGS),
    ("a_comment_or_a_docstring_may_name_the_workspace_tool",
     {**K_CLEAN, "cli/pkg/doctor.py": '"""Written by `acmehub install`."""\n\n# `acmehub add` clones it.\nFIX = 1\n',
      "cmake/Vcpkg.cmake": "# ACME_HOME is read by the CLI.\n"},
     [], K_SETTINGS),
    ("no_workspace_pattern_means_no_workspace_rule",
     {**K_CLEAN, "cli/pkg/doctor.py": 'FIX = "run acmehub install"\n'},
     [], K_NO_WORKSPACE),
)


def _module(files: dict[str, str], config: ModuleSettings) -> Module:
    """Returns the module a shell would read from files, given as their texts below the root."""
    install = files.get(config.install_page)
    present = frozenset(name for name in (config.manifest, config.fragment, config.install_page) if name in files)
    return Module(
        root=K_ROOT,
        config=config,
        present=present,
        install_page=None if install is None else SourceFile(K_ROOT / config.install_page, install),
        sources=tuple(
            SourceFile(K_ROOT / name, text) for name, text in files.items() if name != config.install_page
        ),
    )


def _rules(module: Module) -> list[str]:
    """Returns the name of every rule that fires on the module, once per issue."""
    return [name for name, rule in K_CHECKER.rules.items() for _ in rule(module)]


def _write(root: Path, files: dict[str, str]) -> None:
    """Writes each named file below the root, creating its folders."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class RuleTest(unittest.TestCase):
    """Runs every rule on modules given as values, one case of the table at a time."""

    K_TEST_SIZE = "small"

    def test_each_case_fires_exactly_its_rules(self) -> None:
        """Fires the rules each case names, and no other, on the values the shell would read."""
        for name, files, expected, config in K_CASES:
            with self.subTest(case=name):
                self.assertEqual(_rules(_module(files, config)), expected)

    def test_an_issue_names_the_line_it_is_on(self) -> None:
        """Names line 1 for each heading the install page lacks, and the line of a machine path."""
        page = _module({**K_CLEAN, K_INSTALL_PAGE: "# Install\n"}, K_SETTINGS)
        machine = _module(
            {**K_CLEAN, "CMakePresets.json": '{\n  "p": "D:/Projects/a"\n}\n'},  # agnostic: allow
            K_SETTINGS,
        )
        self.assertEqual([issue.line for issue in rule_install_headings(page)], [1, 1])
        self.assertEqual([issue.line for issue in rule_machine_path(machine)], [2])


class ShellTest(unittest.TestCase):
    """Runs the shell on modules written to disk: each case of the table, and the two skips."""

    K_TEST_SIZE = "medium"

    def test_each_case_fires_exactly_its_rules_on_disk(self) -> None:
        """Reads each case's files through git or the folder walk, then fires the same rules."""
        for name, files, expected, config in K_CASES:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _write(root, files)
                fired = [rule for module in read_modules([root], config) for rule in _rules(module)]
                self.assertEqual(fired, expected)

    def test_a_repository_that_is_not_a_module_is_skipped(self) -> None:
        """Reads no module from a folder whose files name no manifest and no CLI package."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write(root, {"README.md": "# Not a module\n"})
            self.assertEqual(read_modules([root], K_SETTINGS), [])

    def test_a_repository_whose_settings_name_no_manifest_is_skipped(self) -> None:
        """Reads no module when the settings name no manifest, even for a folder that holds one."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write(root, K_CLEAN)
            self.assertEqual(read_modules([root], dataclasses.replace(K_SETTINGS, manifest="")), [])

    def test_only_git_tracked_files_are_read_inside_a_repository(self) -> None:
        """Reads the config file git tracks, and skips an untracked one and a tracked one missing from disk."""
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ):
            for name in [key for key in os.environ if key.startswith("GIT_")]:
                del os.environ[name]
            root = Path(folder)
            files = {**K_CLEAN, "CMakePresets.json": '{"p": "D:/Projects/a"}\n',  # agnostic: allow
                     "untracked.json": '{"p": "D:/Projects/b"}\n',  # agnostic: allow
                     "gone.json": '{"p": "D:/Projects/c"}\n'}  # agnostic: allow
            _write(root, files)
            subprocess.run(["git", "-C", folder, "init", "-q"], check=True)
            tracked = [name for name in files if name not in ("untracked.json", "gone.json")]
            subprocess.run(["git", "-C", folder, "add", "--", *tracked, "gone.json"], check=True)
            (root / "gone.json").unlink()  # tracked in the index, deleted from the folder
            fired = [issue.path.name for module in read_modules([root], K_SETTINGS)
                     for issue in rule_machine_path(module)]
            self.assertEqual(fired, ["CMakePresets.json"])

    def test_without_git_every_config_file_is_read(self) -> None:
        """Reads each config file below the folder when git cannot run, the untracked one included."""
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(subprocess, "run", side_effect=OSError):
            root = Path(folder)
            _write(root, {**K_CLEAN, "untracked.json": '{"p": "D:/Projects/b"}\n'})  # agnostic: allow
            fired = [issue.path.name for module in read_modules([root], K_SETTINGS)
                     for issue in rule_machine_path(module)]
            self.assertEqual(fired, ["untracked.json"])

    def test_collect_returns_what_read_modules_returns(self) -> None:
        """Returns the same modules as the shell under the settings the checker runs with."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write(root, K_CLEAN)
            self.assertEqual(collect([root]), read_modules([root], CHECKER_SETTINGS))


if __name__ == "__main__":
    unittest.main()

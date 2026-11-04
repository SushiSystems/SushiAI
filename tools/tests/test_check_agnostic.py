# test_check_agnostic.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/layering/check_agnostic.py on scopes held in memory, and its command line on a temporary tree."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from layering import check_agnostic  # noqa: E402
from layering.check_agnostic import (  # noqa: E402
    K_CHECKER,
    Scope,
    Terms,
    is_checked_file,
    parse_terms,
    rule_company_tool,
    rule_identity,
    rule_machine_path,
    rule_product_name,
    rule_technology_name,
    rule_terms_file,
)

K_ROOT = Path("repo")
K_TERMS = "\n".join((
    "# a comment",
    "product: Acme[A-Z][a-z]+",
    "tool: acmecli",
    "identity: Jane Doe",
    "technology: Zephyr",
    "allow: AcmeSkills",
    "",
))
K_PARSED = parse_terms(K_ROOT / "terms.txt", K_TERMS)
K_EXEMPT = ("skills/zephyr-layout",)


def _scope(files: dict[str, str], terms: Terms | None = K_PARSED, exempt: tuple[str, ...] = K_EXEMPT) -> Scope:
    """Returns the scope the shell reads for a repository that holds these text files."""
    return Scope(
        K_ROOT,
        tuple(SourceFile(K_ROOT / name, text) for name, text in files.items()),
        terms,
        exempt,
    )


def _found(rule, files: dict[str, str], **kwargs) -> list[tuple[str, int]]:
    """Returns the file name and line of each issue a rule yields for the files."""
    return [(issue.path.name, issue.line) for issue in rule(_scope(files, **kwargs))]


class ProductTermsTest(unittest.TestCase):
    """Covers the rules that match the terms file against the checked text."""
    K_TEST_SIZE = "small"

    def test_product_name_is_reported_with_its_line(self) -> None:
        self.assertEqual(_found(rule_product_name, {"skills/a/SKILL.md": "Clean line.\nUse AcmeLog here.\n"}),
                         [("SKILL.md", 2)])

    def test_allowed_name_is_not_reported(self) -> None:
        self.assertEqual(_found(rule_product_name, {"skills/a/SKILL.md": "The AcmeSkills doctrine.\n"}), [])

    def test_company_tool_and_identity_are_reported_by_their_own_rule(self) -> None:
        files = {"skills/a/SKILL.md": "Run acmecli.\nBy Jane Doe.\n"}
        self.assertEqual(_found(rule_company_tool, files), [("SKILL.md", 1)])
        self.assertEqual(_found(rule_identity, files), [("SKILL.md", 2)])

    def test_technology_name_is_reported_outside_the_exempt_folders(self) -> None:
        files = {"skills/a/SKILL.md": "Runs on Zephyr.\n",
                 "skills/zephyr-layout/SKILL.md": "Zephyr is the subject here.\n"}
        self.assertEqual(_found(rule_technology_name, files), [("SKILL.md", 1)])

    def test_exempt_folder_still_answers_to_the_other_rules(self) -> None:
        files = {"skills/zephyr-layout/SKILL.md": "Zephyr by Jane Doe.\n"}
        self.assertEqual(_found(rule_technology_name, files), [])
        self.assertEqual(_found(rule_identity, files), [("SKILL.md", 1)])

    def test_leading_hash_header_of_a_script_is_not_read(self) -> None:
        files = {"skills/a/tool.py": "# AcmeLog header\n# By Jane Doe\n'doc'\nx = 'AcmeLog'\n"}
        self.assertEqual(_found(rule_product_name, files), [("tool.py", 4)])
        self.assertEqual(_found(rule_identity, files), [])

    def test_a_line_with_the_allow_mark_is_not_read(self) -> None:
        files = {"skills/a/SKILL.md": "AcmeLog at D:/Projects/x. <!-- agnostic: allow -->\nAcmeLog again.\n"}
        self.assertEqual(_found(rule_product_name, files), [("SKILL.md", 2)])
        self.assertEqual(_found(rule_machine_path, files), [])


class MachinePathTest(unittest.TestCase):
    """Covers the rule that names a drive path or a user folder."""
    K_TEST_SIZE = "small"

    def test_machine_paths_are_reported(self) -> None:
        files = {"skills/a/SKILL.md":
                 "At D:/Projects/x.\nAt C:\\Users\\a.\nAt /home/a/b.\nAt ~/.config.\n"}  # agnostic: allow
        self.assertEqual(len(_found(rule_machine_path, files)), 3)

    def test_urls_and_prose_about_paths_are_not_machine_paths(self) -> None:
        files = {"skills/a/SKILL.md": "See https://example.org/a. No `D:/...` appears.\n"}
        self.assertEqual(_found(rule_machine_path, files), [])

    def test_an_escape_sequence_in_a_string_is_not_a_machine_path(self) -> None:
        self.assertEqual(_found(rule_machine_path, {"skills/a/tool.md": "x = 'a:\\nb: 1'\n"}), [])


class CheckedFilesTest(unittest.TestCase):
    """Covers which paths the shell reads: text files below a checked folder, but never a settings file."""
    K_TEST_SIZE = "small"

    def test_the_settings_file_is_not_read(self) -> None:
        self.assertFalse(is_checked_file("tools/settings.py", ("tools",)))

    def test_files_outside_the_checked_folders_are_not_read(self) -> None:
        self.assertFalse(is_checked_file("docs/a.md", ("skills", "claude", "tools")))

    def test_no_checked_folder_means_nothing_is_checked(self) -> None:
        self.assertFalse(is_checked_file("skills/a/SKILL.md", ()))

    def test_a_checked_text_file_is_read(self) -> None:
        self.assertTrue(is_checked_file("skills/a/SKILL.md", ("skills",)))
        self.assertFalse(is_checked_file("skills/a/image.png", ("skills",)))


class TermsFileTest(unittest.TestCase):
    """Covers a terms file that is missing, unreadable or holds a pattern that does not compile."""
    K_TEST_SIZE = "small"

    def test_missing_terms_file_is_one_issue_and_silences_the_term_rules(self) -> None:
        missing = _scope({"skills/a/SKILL.md": "AcmeLog\n"}, terms=parse_terms(K_ROOT / "terms.txt", None))
        self.assertEqual([(issue.path.name, issue.line) for issue in rule_terms_file(missing)],
                         [("terms.txt", 1)])
        self.assertEqual(list(rule_product_name(missing)), [])

    def test_unreadable_terms_line_is_reported_with_its_number(self) -> None:
        terms = parse_terms(K_ROOT / "terms.txt", "product: ok\nnot a term\n")
        self.assertEqual([(issue.path.name, issue.line) for issue in rule_terms_file(_scope({}, terms=terms))],
                         [("terms.txt", 1)])
        self.assertIn("terms.txt:2", terms.error)

    def test_a_pattern_that_does_not_compile_is_reported(self) -> None:
        self.assertIn("terms.txt:1", parse_terms(K_ROOT / "terms.txt", "product: (\n").error)


class ShellTest(unittest.TestCase):
    """Runs the command line on a tree on disk, so the files and terms the shell reads reach the rules."""
    K_TEST_SIZE = "medium"

    def test_cli_reads_the_text_files_of_the_checked_folders_only(self) -> None:
        """Reports the one checked line that names a product, and skips a file outside the folders and the settings."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in {
                "terms.txt": K_TERMS,
                "skills/a/SKILL.md": "Use AcmeLog here.\n",
                "docs/a.md": "AcmeLog outside the checked folders.\n",
                "tools/settings.py": "K_X = 'AcmeLog'\n",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            output = io.StringIO()
            with mock.patch.object(check_agnostic, "K_AGNOSTIC_PATHS", ("skills", "tools")), \
                    mock.patch.object(check_agnostic, "K_AGNOSTIC_TERMS", "terms.txt"), \
                    mock.patch.object(check_agnostic, "K_TECHNOLOGY_EXEMPT", ()), \
                    contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [folder, "--rule", "rule_product_name"])
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue().splitlines(),
                             ["skills/a/SKILL.md:1: rule_product_name: names 'AcmeLog'"])


if __name__ == "__main__":
    unittest.main()

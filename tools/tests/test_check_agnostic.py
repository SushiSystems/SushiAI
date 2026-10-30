# test_check_agnostic.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/layering/check_agnostic.py on repositories built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering import check_agnostic  # noqa: E402
from layering.check_agnostic import (  # noqa: E402
    collect,
    rule_company_tool,
    rule_identity,
    rule_machine_path,
    rule_product_name,
    rule_technology_name,
    rule_terms_file,
)

K_TERMS = "\n".join((
    "# a comment",
    "product: Acme[A-Z][a-z]+",
    "tool: acmecli",
    "identity: Jane Doe",
    "technology: Zephyr",
    "allow: AcmeSkills",
    "",
))


class AgnosticTest(unittest.TestCase):
    """Builds a repository for each test, with `skills` as its checked folder."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        patches = [mock.patch.object(check_agnostic, "K_AGNOSTIC_PATHS", ("skills",)),
                   mock.patch.object(check_agnostic, "K_AGNOSTIC_TERMS", "terms.txt"),
                   mock.patch.object(check_agnostic, "K_TECHNOLOGY_EXEMPT", ("skills/zephyr-layout",))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.write("terms.txt", K_TERMS)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, relative: str, text: str) -> None:
        """Writes a file below the repository root."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def found(self, rule) -> list[tuple[str, int]]:
        """Returns the file name and line of each issue a rule yields."""
        return [(issue.path.name, issue.line)
                for scope in collect([self.root]) for issue in rule(scope)]

    def test_product_name_is_reported_with_its_line(self) -> None:
        self.write("skills/a/SKILL.md", "Clean line.\nUse AcmeLog here.\n")
        self.assertEqual(self.found(rule_product_name), [("SKILL.md", 2)])

    def test_allowed_name_is_not_reported(self) -> None:
        self.write("skills/a/SKILL.md", "The AcmeSkills doctrine.\n")
        self.assertEqual(self.found(rule_product_name), [])

    def test_company_tool_and_identity_are_reported_by_their_own_rule(self) -> None:
        self.write("skills/a/SKILL.md", "Run acmecli.\nBy Jane Doe.\n")
        self.assertEqual(self.found(rule_company_tool), [("SKILL.md", 1)])
        self.assertEqual(self.found(rule_identity), [("SKILL.md", 2)])

    def test_machine_paths_are_reported(self) -> None:
        self.write("skills/a/SKILL.md",
                   "At D:/Projects/x.\nAt C:\\Users\\a.\nAt /home/a/b.\nAt ~/.config.\n")  # agnostic: allow
        self.assertEqual(len(self.found(rule_machine_path)), 3)

    def test_urls_and_prose_about_paths_are_not_machine_paths(self) -> None:
        self.write("skills/a/SKILL.md", "See https://example.org/a. No `D:/...` appears.\n")
        self.assertEqual(self.found(rule_machine_path), [])

    def test_technology_name_is_reported_outside_the_exempt_folders(self) -> None:
        self.write("skills/a/SKILL.md", "Runs on Zephyr.\n")
        self.write("skills/zephyr-layout/SKILL.md", "Zephyr is the subject here.\n")
        self.assertEqual(self.found(rule_technology_name), [("SKILL.md", 1)])

    def test_exempt_folder_still_answers_to_the_other_rules(self) -> None:
        self.write("skills/zephyr-layout/SKILL.md", "Zephyr by Jane Doe.\n")
        self.assertEqual(self.found(rule_technology_name), [])
        self.assertEqual(self.found(rule_identity), [("SKILL.md", 1)])

    def test_leading_hash_header_of_a_script_is_not_read(self) -> None:
        self.write("skills/a/tool.py", "# AcmeLog header\n# By Jane Doe\n'doc'\nx = 'AcmeLog'\n")
        self.assertEqual(self.found(rule_product_name), [("tool.py", 4)])
        self.assertEqual(self.found(rule_identity), [])

    def test_a_line_with_the_allow_mark_is_not_read(self) -> None:
        self.write("skills/a/SKILL.md", "AcmeLog at D:/Projects/x. <!-- agnostic: allow -->\nAcmeLog again.\n")
        self.assertEqual(self.found(rule_product_name), [("SKILL.md", 2)])
        self.assertEqual(self.found(rule_machine_path), [])

    def test_an_escape_sequence_in_a_string_is_not_a_machine_path(self) -> None:
        self.write("skills/a/tool.md", "x = 'a:\\nb: 1'\n")
        self.assertEqual(self.found(rule_machine_path), [])

    def test_the_settings_file_is_not_read(self) -> None:
        self.write("tools/settings.py", "K_X = 'AcmeLog'\n")
        with mock.patch.object(check_agnostic, "K_AGNOSTIC_PATHS", ("tools",)):
            self.assertEqual(self.found(rule_product_name), [])

    def test_files_outside_the_checked_folders_are_not_read(self) -> None:
        self.write("docs/a.md", "AcmeLog and D:/x.\n")
        self.assertEqual(self.found(rule_product_name), [])

    def test_no_checked_folder_means_nothing_is_checked(self) -> None:
        self.write("skills/a/SKILL.md", "AcmeLog\n")
        with mock.patch.object(check_agnostic, "K_AGNOSTIC_PATHS", ()):
            self.assertEqual(self.found(rule_product_name), [])

    def test_missing_terms_file_is_one_issue_and_silences_the_term_rules(self) -> None:
        self.write("skills/a/SKILL.md", "AcmeLog\n")
        (self.root / "terms.txt").unlink()
        self.assertEqual(self.found(rule_terms_file), [("terms.txt", 1)])
        self.assertEqual(self.found(rule_product_name), [])

    def test_unreadable_terms_line_is_reported_with_its_number(self) -> None:
        self.write("terms.txt", "product: ok\nnot a term\n")
        self.assertEqual(self.found(rule_terms_file), [("terms.txt", 1)])
        scope = next(iter(collect([self.root])))
        self.assertIn("terms.txt:2", scope.terms.error)

    def test_a_pattern_that_does_not_compile_is_reported(self) -> None:
        self.write("terms.txt", "product: (\n")
        scope = next(iter(collect([self.root])))
        self.assertIn("terms.txt:1", scope.terms.error)


if __name__ == "__main__":
    unittest.main()

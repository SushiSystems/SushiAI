# test_check_colour_names.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rule of tools/cli_source/check_colour_names.py on files built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli_source.check_colour_names import collect, rule_raw_colour_name  # noqa: E402


class RawColourNameTest(unittest.TestCase):
    """Builds a temporary repository with one file in the CLI package."""

    K_TEST_SIZE = "small"

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        self.package = self.root / "cli" / "sushiai"
        self.package.mkdir(parents=True)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _messages(self, text: str) -> list[str]:
        """Returns the message of each issue the rule yields for a file holding text."""
        file = self.package / "sample.py"
        file.write_text(text, encoding="utf-8")
        return [issue.message for issue in rule_raw_colour_name(file)]

    def test_reports_a_string_with_a_raw_colour_tag(self) -> None:
        self.assertEqual(len(self._messages('line = "[green]done[/green]"\n')), 1)

    def test_reports_a_closing_tag_alone(self) -> None:
        self.assertEqual(len(self._messages('line = "done[/red]"\n')), 1)

    def test_accepts_a_string_with_no_colour_tag(self) -> None:
        self.assertEqual(self._messages('line = "done"\n'), [])

    def test_ignores_a_colour_word_outside_a_tag(self) -> None:
        self.assertEqual(self._messages('line = "the green light"\n'), [])

    def test_ignores_a_colour_name_in_a_comment(self) -> None:
        self.assertEqual(self._messages('# [green] is the old spelling\nline = 1\n'), [])

    def test_collects_the_files_of_the_package(self) -> None:
        (self.package / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual([file.name for file in collect([self.root])], ["a.py"])


if __name__ == "__main__":
    unittest.main()

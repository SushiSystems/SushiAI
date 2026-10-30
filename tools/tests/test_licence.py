# test_licence.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests how the licence of a repository is read from its settings."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation import check_source_comments, licence  # noqa: E402
from licensing import write_license_block  # noqa: E402

K_SETTINGS = SimpleNamespace(
    K_LICENSE_HOLDER="Jane Doe & Acme Corp",
    K_LICENSE_COMPANY="Acme Corp",
    K_LICENSE_SITE="https://acme.example",
    K_LICENSE_OPEN_LINES=("Open row.",),
    K_LICENSE_CLOSED_LINES=("Closed row.",),
    K_LICENSE_OPEN_BOX_LINES=("Open box row.", "", "Second."),
    K_LICENSE_CLOSED_BOX_LINES=("Closed box row.",),
    K_CLOSED=False,
)


class LicenceTest(unittest.TestCase):
    """Covers reading the settings and what a repository without them gets."""

    def test_reads_every_value_from_the_settings(self) -> None:
        with mock.patch.object(licence, "settings", K_SETTINGS):
            loaded = licence.load()
        self.assertEqual(loaded.holder, "Jane Doe & Acme Corp")
        self.assertEqual(loaded.hash_lines, ("Open row.",))
        self.assertEqual(loaded.box_lines, ("Open box row.", "", "Second."))
        self.assertIn("Open box row.", loaded.own_rows)
        self.assertNotIn("", loaded.own_rows)

    def test_a_closed_repository_takes_the_closed_rows(self) -> None:
        closed = SimpleNamespace(**{**vars(K_SETTINGS), "K_CLOSED": True})
        with mock.patch.object(licence, "settings", closed):
            loaded = licence.load()
        self.assertEqual(loaded.hash_lines, ("Closed row.",))
        self.assertEqual(loaded.box_lines, ("Closed box row.",))

    def test_settings_without_a_holder_describe_no_licence(self) -> None:
        with mock.patch.object(licence, "settings", SimpleNamespace()):
            self.assertIsNone(licence.load())

    def test_the_reserved_pattern_names_the_company_only_when_there_is_one(self) -> None:
        with mock.patch.object(licence, "settings", K_SETTINGS):
            loaded = licence.load()
        self.assertTrue(loaded.reserved_line.search("Made by Acme Corp"))
        bare = SimpleNamespace(**{**vars(K_SETTINGS), "K_LICENSE_COMPANY": ""})
        with mock.patch.object(licence, "settings", bare):
            self.assertFalse(licence.load().reserved_line.search("Made by Acme Corp"))

    def test_the_checker_stops_when_no_licence_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(check_source_comments, "K_LICENCE", None):
            with self.assertRaises(SystemExit) as stopped:
                check_source_comments.collect([Path(folder)])
        self.assertIn("K_LICENSE_HOLDER", str(stopped.exception))

    def test_the_writer_stops_when_no_licence_is_configured(self) -> None:
        with mock.patch.object(write_license_block, "K_LICENCE", None):
            with self.assertRaises(write_license_block.LicenseWriterError):
                write_license_block.Header("P - https://x.example").hashed_rows("a.py", 2026)


if __name__ == "__main__":
    unittest.main()

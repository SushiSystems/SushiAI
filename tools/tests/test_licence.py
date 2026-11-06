# test_licence.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests how the licence of a repository is read from its settings."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation import check_source_comments, licence  # noqa: E402
from documentation.licence import K_UNSET_MESSAGE  # noqa: E402
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


K_TEST_SIZE = "small"


class LicenceTest(unittest.TestCase):
    """Covers reading the settings and what a repository without them gets."""

    def test_reads_every_value_from_the_settings(self) -> None:
        loaded = licence.from_module(K_SETTINGS)
        self.assertEqual(loaded.holder, "Jane Doe & Acme Corp")
        self.assertEqual(loaded.hash_lines, ("Open row.",))
        self.assertEqual(loaded.box_lines, ("Open box row.", "", "Second."))
        self.assertIn("Open box row.", loaded.own_rows)
        self.assertNotIn("", loaded.own_rows)

    def test_a_closed_repository_takes_the_closed_rows(self) -> None:
        closed = SimpleNamespace(**{**vars(K_SETTINGS), "K_CLOSED": True})
        loaded = licence.from_module(closed)
        self.assertEqual(loaded.hash_lines, ("Closed row.",))
        self.assertEqual(loaded.box_lines, ("Closed box row.",))

    def test_settings_that_do_not_set_closed_take_the_open_rows(self) -> None:
        values = {name: value for name, value in vars(K_SETTINGS).items() if name != "K_CLOSED"}
        unset = SimpleNamespace(**values)
        loaded = licence.from_module(unset)
        self.assertEqual(loaded.hash_lines, ("Open row.",))

    def test_settings_without_a_holder_describe_no_licence(self) -> None:
        self.assertIsNone(licence.from_module(SimpleNamespace()))

    def test_the_reserved_pattern_names_the_company_only_when_there_is_one(self) -> None:
        loaded = licence.from_module(K_SETTINGS)
        self.assertTrue(loaded.reserved_line.search("Made by Acme Corp"))
        bare = SimpleNamespace(**{**vars(K_SETTINGS), "K_LICENSE_COMPANY": ""})
        self.assertFalse(licence.from_module(bare).reserved_line.search("Made by Acme Corp"))

    def test_the_copyright_row_matches_the_holder_in_any_year_and_no_other(self) -> None:
        pattern = licence.from_module(K_SETTINGS).copyright_line
        rows = (
            ("Copyright (c) 2026-present Jane Doe & Acme Corp", True),
            ("Copyright (c) 1999-present Jane Doe & Acme Corp", True),
            ("Copyright (c) 2026-present Someone Else", False),
            ("Copyright (c) 2026 Jane Doe & Acme Corp", False),
            ("Copyright (c) 2026-present Jane Doe & Acme Corp Ltd", False),
        )
        for row, matches in rows:
            with self.subTest(row=row):
                self.assertEqual(pattern.search(row) is not None, matches)

    def test_the_checker_stops_when_no_licence_is_configured(self) -> None:
        with self.assertRaises(SystemExit) as stopped:
            check_source_comments.collect([Path("unused")], licence=None)
        self.assertIn("K_LICENSE_HOLDER", str(stopped.exception))

    def test_the_writer_stops_when_no_licence_is_configured(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = write_license_block.main(["unused", "--project", "P - https://x.example"], licence=None)
        self.assertEqual((code, stderr.getvalue().strip()), (2, K_UNSET_MESSAGE))


if __name__ == "__main__":
    unittest.main()

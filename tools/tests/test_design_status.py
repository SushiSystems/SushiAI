# test_design_status.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the parser and the listings of design documents, from values only."""

from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.design_status import (  # noqa: E402
    Status,
    Tree,
    body_lines,
    children,
    design_documents,
    first_content_line,
    is_under_revision,
    parse_status,
    section_body,
    section_names,
    status_documents,
    table_cells,
    work_folders,
)


def _tree(folders: set[str], files: set[str], texts: dict[str, str] | None = None) -> Tree:
    """Returns a tree from its folders, its files and the text of the files that hold text."""
    return Tree(frozenset(folders), frozenset(files), texts or {})


class FirstLineTest(unittest.TestCase):
    """Covers the line that holds a document's status."""

    K_TEST_SIZE = "small"

    def test_skips_the_title_and_blank_lines(self) -> None:
        """Returns the number and text of the first non-empty line after the title."""
        self.assertEqual(first_content_line("# Title\n\n\n**Status:** Open — x\n"), (4, "**Status:** Open — x"))

    def test_returns_an_empty_line_when_the_document_has_none(self) -> None:
        """Returns line 1 with no text when nothing follows the title."""
        self.assertEqual(first_content_line("# Title\n"), (1, ""))


class StatusFormTest(unittest.TestCase):
    """Covers each of the four forms of the status line, and what it says."""

    K_TEST_SIZE = "small"

    def test_reads_an_open_status(self) -> None:
        """Reads an open document as known and open, with no close date."""
        self.assertEqual(parse_status("# T\n\n**Status:** Open — under way\n"), Status(3, True, True, None))

    def test_reads_a_shipped_status_and_its_date_before_a_full_stop(self) -> None:
        """Reads the date of a shipped document, which may be followed by a sentence."""
        self.assertEqual(
            parse_status("# T\n**Status:** Shipped — 2026-01-02. Kept as the record.\n"),
            Status(2, True, False, datetime.date(2026, 1, 2), True),
        )

    def test_reads_a_superseded_status_by_its_successor(self) -> None:
        """Reads a superseded document's date; it is not shipped."""
        self.assertEqual(
            parse_status("# T\n**Status:** Superseded — 2026-03-04 by ECS\n"),
            Status(2, True, False, datetime.date(2026, 3, 4), False),
        )

    def test_reads_a_withdrawn_status_with_its_reason(self) -> None:
        """Reads a withdrawn document's date, with the reason after a comma."""
        self.assertEqual(
            parse_status("# T\n**Status:** Withdrawn — 2026-05-06, the plan changed\n"),
            Status(2, True, False, datetime.date(2026, 5, 6), False),
        )

    def test_rejects_a_date_that_does_not_exist(self) -> None:
        """Calls a shipped status with an impossible date unknown."""
        self.assertEqual(parse_status("# T\n**Status:** Shipped — 2026-13-45\n"), Status(2, False, False, None))

    def test_rejects_a_status_in_none_of_the_forms(self) -> None:
        """Calls any other first line unknown, and an empty document too."""
        self.assertEqual(parse_status("# T\n**Status:** done\n"), Status(2, False, False, None))
        self.assertEqual(parse_status("# T\n"), Status(1, False, False, None))


class BodyTest(unittest.TestCase):
    """Covers the lines of a document outside its fenced blocks, and its sections."""

    K_TEST_SIZE = "small"

    def test_leaves_out_a_fenced_block(self) -> None:
        """Keeps the lines outside the fence and drops those inside it, headings included."""
        text = "# T\n```\n## Inside\n```\n## Outside\n"
        self.assertEqual(body_lines(text), [(1, "# T"), (5, "## Outside")])

    def test_names_each_section_heading_with_its_line(self) -> None:
        """Lists each level-two heading, not one that sits inside a fence."""
        text = "# T\n## Problem\nx\n```\n## Not a heading\n```\n## Phases\n"
        self.assertEqual(section_names(text), [(2, "Problem"), (7, "Phases")])

    def test_returns_the_lines_under_one_section_only(self) -> None:
        """Returns the lines between the named heading and the next one."""
        text = "## Phases\n| a |\n## Other\nz\n"
        self.assertEqual(section_body(text, "Phases"), [(2, "| a |")])

    def test_splits_a_table_row_into_trimmed_cells(self) -> None:
        """Returns each cell of a row, without the outer bars or the spaces around the cells."""
        self.assertEqual(table_cells("| A0 | The mixer | built |"), ["A0", "The mixer", "built"])


class ListingTest(unittest.TestCase):
    """Covers which design documents and work folders a tree holds."""

    K_TEST_SIZE = "small"

    def test_lists_the_entries_directly_inside_a_folder_sorted(self) -> None:
        """Returns the folders and files one level down, not what lies deeper."""
        tree = _tree(
            {"docs", "docs/agent", "docs/agent/2026_10_21_X"},
            {"docs/agent/2026_10_21_X/SPEC.md", "docs/agent/README.md"},
        )
        self.assertEqual(children(tree, "docs/agent"), ["docs/agent/2026_10_21_X", "docs/agent/README.md"])

    def test_counts_a_split_topic_readme_and_the_topics_under_revision(self) -> None:
        """Lists a loose document, a split topic's README and the README of each topic under revision; not an index."""
        tree = _tree(
            {"docs", "docs/design", "docs/design/AUDIO", "docs/design/_REVISION", "docs/design/_REVISION/SOUND"},
            {
                "docs/design/ECS.md", "docs/design/README.md", "docs/design/AUDIO/README.md",
                "docs/design/_REVISION/SOUND/README.md", "docs/design/_REVISION/SOUND/MIXER.md",
            },
            {"docs/design/ECS.md": "# Ecs\n"},
        )
        self.assertEqual(
            design_documents(tree),
            ["docs/design/ECS.md", "docs/design/AUDIO/README.md", "docs/design/_REVISION/SOUND/README.md"],
        )

    def test_pairs_each_work_folder_with_its_first_status_file(self) -> None:
        """Takes SPEC.md over PLAN.md for a work folder that has both, and PLAN.md when it has no spec."""
        tree = _tree(
            {"docs", "docs/agent", "docs/agent/2026_10_21_X", "docs/agent/2026_10_22_Y"},
            {"docs/agent/2026_10_21_X/SPEC.md", "docs/agent/2026_10_21_X/PLAN.md", "docs/agent/2026_10_22_Y/PLAN.md"},
        )
        self.assertEqual(
            status_documents(tree),
            ["docs/agent/2026_10_21_X/SPEC.md", "docs/agent/2026_10_22_Y/PLAN.md"],
        )

    def test_work_folders_lists_every_entry_under_agent(self) -> None:
        """Returns each entry directly under docs/agent, a stray file included, sorted."""
        tree = _tree({"docs", "docs/agent", "docs/agent/2026_10_21_X"}, {"docs/agent/README.md"})
        self.assertEqual(work_folders(tree), ["docs/agent/2026_10_21_X", "docs/agent/README.md"])

    def test_knows_a_topic_under_revision_by_its_path(self) -> None:
        """Answers yes for a path with the revision folder in it, and no for a live topic."""
        self.assertTrue(is_under_revision("docs/design/_REVISION/SOUND/README.md"))
        self.assertFalse(is_under_revision("docs/design/ECS.md"))


if __name__ == "__main__":
    unittest.main()

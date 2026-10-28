# test_drop_retired_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the replacement of citations of documents that version control keeps and the tree does not."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.drop_retired_citations import drop_text, main, phrase, run  # noqa: E402

K_ARCHIVE = ["docs/archive"]
K_WORK = ["docs/agent/2026_09_22_ROOM_FURNISHINGS"]
K_FILE = Path("x.md")


def _drop(text: str, prefixes: list[str] | None = None) -> str:
    """Returns the text after its retired citations are replaced."""
    return drop_text(K_FILE, text, prefixes or K_ARCHIVE)[0]


class PhraseTest(unittest.TestCase):
    """Covers the words that stand for a cited file."""

    def test_names_a_spec_by_its_file(self) -> None:
        """Drops the date and the extension and adds the kind from the folder."""
        self.assertEqual(
            phrase("archive/superpowers/specs/2026-09-08-room-acoustics-design.md"),
            "the room acoustics design spec in the version history",
        )

    def test_names_a_work_file_by_its_folder(self) -> None:
        """Reads the folder of SPEC.md, PLAN.md and REPORT.md, not the file name."""
        self.assertEqual(phrase("agent/2026_09_22_ROOM_FURNISHINGS/PLAN.md"), "the room furnishings plan in the version history")

    def test_names_a_readme_by_its_folder(self) -> None:
        """Reads the folder of a README.md, which says nothing by itself."""
        self.assertEqual(phrase("design/OLD_TOPIC/README.md"), "the old topic document in the version history")

    def test_says_the_article_once(self) -> None:
        """Drops a leading THE from the file name, so the phrase does not read 'the the'."""
        self.assertEqual(phrase("design/WAVES/THE_SHORE_WINDOWS.md"), "the shore windows document in the version history")

    def test_falls_back_to_document(self) -> None:
        """Calls a file in an unknown folder a document."""
        self.assertEqual(phrase("archive/design/OLD_THING.md"), "the old thing document in the version history")


class RelativeLinkTest(unittest.TestCase):
    """Covers a link written relative to its own folder rather than to docs/."""

    def test_replaces_a_sibling_topic_link_given_the_root(self) -> None:
        """Turns a design page's link to a deleted sibling topic into its text, and leaves a live one."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            page = root / "docs/design/LIVE/README.md"
            page.parent.mkdir(parents=True)
            (root / "docs/design/OTHER").mkdir(parents=True)
            text = "See [the old topic](../OLD/README.md#phases) and [the other](../OTHER/README.md).\n"
            fixed, drops = drop_text(page, text, ["docs/design/OLD"], root)
            self.assertEqual(fixed, "See the old topic and [the other](../OTHER/README.md).\n")
            self.assertEqual(len(drops), 1)

    def test_leaves_relative_links_alone_without_the_root(self) -> None:
        """Reads no relative link when the caller gives no root, as before."""
        text = "See [the old topic](../OLD/README.md).\n"
        self.assertEqual(drop_text(Path("docs/design/LIVE/README.md"), text, ["docs/design/OLD"])[0], text)


class DropTest(unittest.TestCase):
    """Covers each form a citation takes."""

    def test_replaces_a_code_span_and_keeps_the_section(self) -> None:
        """Replaces a path in a code span, backticks included, and leaves the section mark after it."""
        text = "See `docs/archive/superpowers/specs/2026-09-08-room-acoustics-design.md` §9.3 for it.\n"
        self.assertEqual(_drop(text), "See the room acoustics design spec in the version history §9.3 for it.\n")

    def test_replaces_a_bare_path_in_a_comment(self) -> None:
        """Replaces a path in a source comment."""
        text = "/// @see docs/archive/superpowers/specs/2026-09-07-stereo-tracks-design.md section 6\n"
        self.assertEqual(_drop(text), "/// @see the stereo tracks design spec in the version history section 6\n")

    def test_keeps_the_text_of_a_link(self) -> None:
        """Turns a link into its text and drops its anchor."""
        self.assertEqual(_drop("Read [the old spec](../archive/specs/2026-01-01-a-b.md#top) first.\n"), "Read the old spec first.\n")

    def test_replaces_a_link_whose_text_is_the_path(self) -> None:
        """Gives the phrase for a link whose text is the path in a code span."""
        text = "[`docs/archive/specs/2026-01-01-a-b.md`](../archive/specs/2026-01-01-a-b.md)\n"
        self.assertEqual(_drop(text), "the a b spec in the version history\n")

    def test_reads_a_path_reached_from_a_module(self) -> None:
        """Replaces a path written with leading parent folders and docs/."""
        self.assertEqual(_drop("[x](../../docs/archive/notes/2026-02-02-n.md)\n"), "x\n")

    def test_replaces_a_retired_work_folder(self) -> None:
        """Replaces a citation of a deleted work folder's file."""
        text = "The task order is [`docs/agent/2026_09_22_ROOM_FURNISHINGS/PLAN.md`](../agent/2026_09_22_ROOM_FURNISHINGS/PLAN.md).\n"
        self.assertEqual(_drop(text, K_WORK), "The task order is the room furnishings plan in the version history.\n")

    def test_leaves_other_paths_alone(self) -> None:
        """Does not touch a citation of a live document or of a folder."""
        text = "See docs/design/ROOM.md and docs/archive/ and docs/agent/2026_10_01_OTHER/SPEC.md.\n"
        self.assertEqual(_drop(text), text)


class RunTest(unittest.TestCase):
    """Covers walking a repository and the command."""

    def _root(self, folder: str) -> Path:
        """Writes a small repository and returns its root."""
        root = Path(folder) / "alpha"
        for name, text in {
            "docs/guides/use.md": "See docs/archive/specs/2026-01-01-a-b.md.\n",
            "docs/archive/specs/2026-01-01-a-b.md": "See docs/archive/specs/2026-01-02-c.md.\n",
            "modules/core/source/x.cpp": "// docs/archive/notes/2026-02-02-n.md\n",
        }.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def test_a_dry_run_lists_and_writes_nothing(self) -> None:
        """Returns the drops and leaves every file as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            self.assertEqual(len(run(root, K_ARCHIVE, False)), 2)
            self.assertIn("docs/archive", (root / "docs/guides/use.md").read_text(encoding="utf-8"))

    def test_apply_writes_outside_the_archive_only(self) -> None:
        """Rewrites the manual and the source and leaves the archive itself as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            run(root, K_ARCHIVE, True)
            self.assertEqual((root / "docs/guides/use.md").read_text(encoding="utf-8"), "See the a b spec in the version history.\n")
            self.assertIn("docs/archive", (root / "docs/archive/specs/2026-01-01-a-b.md").read_text(encoding="utf-8"))

    def test_the_command_exits_one_for_a_dry_run_that_finds_something(self) -> None:
        """Returns 1 without --apply while citations remain, and 0 once they are replaced."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root), "--prefix", "docs/archive"]), 1)
                self.assertEqual(main([str(root), "--prefix", "docs/archive", "--apply"]), 0)
                self.assertEqual(main([str(root), "--prefix", "docs/archive"]), 0)


if __name__ == "__main__":
    unittest.main()

# test_repair_design_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the repair of citations of design documents that moved."""

from __future__ import annotations

import contextlib
import io
import posixpath
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.repair_design_citations import is_scanned, main, relative_name, repair_text, run  # noqa: E402

K_TOPIC = "# Topic\n\n**Status:** Open — under way\n"

K_TOPICS = {
    "docs/design/AUDIO/README.md": K_TOPIC,
    "docs/design/AUDIO/MIXER.md": "# Mixer\n",
    "docs/design/ECS.md": K_TOPIC,
    "docs/design/README.md": "# Design\n",
}

K_FILES = {
    **K_TOPICS,
    "docs/guides/use.md": "See `docs/design/AUDIO.md` and [the ecs](../design/ECS/README.md#phases).\n",
    "modules/core/source/mixer.cpp": "// The rule is in docs/design/ECS/README.md.\n",
}

K_KNOWN = {"AUDIO": "docs/design/AUDIO/README.md", "ECS": "docs/design/ECS.md"}
K_USE = "docs/guides/use.md"


def _present(*names: str) -> frozenset[str]:
    """Returns the names given with every folder above them, as the shell's walk would list them."""
    found: set[str] = set()
    for name in names:
        parts = name.split("/")
        found.update("/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
    return frozenset(found)


K_PRESENT = _present(
    "docs/design/AUDIO/README.md", "docs/design/AUDIO/MIXER.md", "docs/design/ECS.md", "docs/design/README.md",
    "docs/guides/use.md", "modules/core/source/mixer.cpp",
)
K_REVISED = _present(
    "docs/design/ECS.md", "docs/design/README.md", "docs/design/_REVISION/AUDIO/README.md",
    "docs/design/_REVISION/AUDIO/MIXER.md", "docs/guides/use.md",
)
K_REVISED_KNOWN = {"AUDIO": "docs/design/_REVISION/AUDIO/README.md", "ECS": "docs/design/ECS.md"}


class RelativeNameTest(unittest.TestCase):
    """Covers the path from one folder to a file, as the relative link is written."""

    K_TEST_SIZE = "small"

    def test_climbs_out_of_a_sibling_folder(self) -> None:
        """Goes up from the folder the link is written in, then down to the file."""
        self.assertEqual(relative_name("docs/design/ECS.md", "docs/guides"), "../design/ECS.md")

    def test_names_a_file_in_the_same_folder_bare(self) -> None:
        """Gives the bare name of a file that sits beside the link."""
        self.assertEqual(relative_name("docs/design/ECS.md", "docs/design"), "ECS.md")

    def test_climbs_before_it_walks_down_when_the_paths_part_at_the_first_folder(self) -> None:
        """Keeps the first folder of the target when it differs from the first folder of the start."""
        self.assertEqual(relative_name("modules/core/x.cpp", "docs/guides"), "../../modules/core/x.cpp")

    def test_gives_a_dot_for_the_folder_itself(self) -> None:
        """Gives the current folder when the target is the folder the link is written from."""
        self.assertEqual(relative_name("docs/design", "docs/design"), ".")

    def test_counts_each_folder_of_the_revision_path(self) -> None:
        """Climbs one level for each folder the link leaves, including the revision folder."""
        self.assertEqual(relative_name("docs/design/ECS.md", "docs/design/_REVISION/AUDIO"), "../../ECS.md")


class ScanTest(unittest.TestCase):
    """Covers which files a repair reads at all."""

    K_TEST_SIZE = "small"

    def test_skips_a_file_with_no_design_citation(self) -> None:
        """Passes over a file outside docs/design that never names design/."""
        self.assertFalse(is_scanned("modules/core/source/x.cpp", "int x;\n", None))

    def test_reads_a_file_under_design_and_a_file_that_names_design(self) -> None:
        """Reads a design document whatever its text, and any file that names design/."""
        self.assertTrue(is_scanned("docs/design/ECS.md", "# Ecs\n", None))
        self.assertTrue(is_scanned("docs/guides/use.md", "See design/ECS.md.\n", None))

    def test_reads_every_file_when_a_move_is_given(self) -> None:
        """Reads any file once a move is named, because a link may point at the moved file."""
        self.assertTrue(is_scanned("docs/guides/use.md", "[m](../design/AUDIO/MIXER.md)\n", {"a": "b"}))


class RepairTextTest(unittest.TestCase):
    """Covers what the repair rewrites and what it leaves, as text in and text out."""

    K_TEST_SIZE = "small"

    def test_points_a_file_citation_at_the_folder_that_replaced_it(self) -> None:
        """Rewrites a citation of AUDIO.md to AUDIO/README.md."""
        fixed, repairs = repair_text(K_USE, "See `docs/design/AUDIO.md`.\n", K_PRESENT, K_KNOWN)
        self.assertEqual(fixed, "See `docs/design/AUDIO/README.md`.\n")
        self.assertEqual([(item.old, item.new) for item in repairs], [("docs/design/AUDIO.md", "docs/design/AUDIO/README.md")])

    def test_points_a_folder_citation_at_the_file_that_replaced_it(self) -> None:
        """Rewrites ECS/README.md to ECS.md in a link, keeping its anchor, and in a source comment."""
        link, repairs = repair_text(K_USE, "See [the ecs](../design/ECS/README.md#phases).\n", K_PRESENT, K_KNOWN)
        self.assertEqual(link, "See [the ecs](../design/ECS.md#phases).\n")
        self.assertEqual([(item.old, item.new) for item in repairs], [("../design/ECS/README.md", "../design/ECS.md")])
        comment, repairs = repair_text(
            "modules/core/source/mixer.cpp", "// The rule is in docs/design/ECS/README.md.\n", K_PRESENT, K_KNOWN,
        )
        self.assertEqual(comment, "// The rule is in docs/design/ECS.md.\n")
        self.assertEqual(len(repairs), 1)

    def test_reports_a_page_the_topic_does_not_have(self) -> None:
        """Reports a citation of a missing page and does not rewrite it."""
        fixed, repairs = repair_text(K_USE, "See docs/design/AUDIO/GONE.md.\n", K_PRESENT, K_KNOWN)
        self.assertEqual(fixed, "See docs/design/AUDIO/GONE.md.\n")
        self.assertEqual([(item.old, item.new) for item in repairs], [("docs/design/AUDIO/GONE.md", None)])

    def test_reports_a_topic_that_does_not_exist(self) -> None:
        """Reports a citation of a topic nobody wrote, and leaves it as it was."""
        fixed, repairs = repair_text(K_USE, "See `docs/design/VIDEO.md`.\n", K_PRESENT, K_KNOWN)
        self.assertEqual(fixed, "See `docs/design/VIDEO.md`.\n")
        self.assertEqual([item.new for item in repairs], [None])

    def test_leaves_a_citation_that_resolves_alone(self) -> None:
        """Changes no citation that already points at a file that exists."""
        self.assertEqual(repair_text(K_USE, "See docs/design/AUDIO/MIXER.md.\n", K_PRESENT, K_KNOWN), ("See docs/design/AUDIO/MIXER.md.\n", []))

    def test_repairs_a_path_that_starts_at_design(self) -> None:
        """Rewrites `design/AUDIO.md` and `../design/AUDIO.md` in prose, keeping the form the author used."""
        fixed, _ = repair_text(
            K_USE, "Read `design/AUDIO.md`, then ../design/AUDIO.md and redesign/AUDIO.md.\n", K_PRESENT, K_KNOWN,
        )
        self.assertEqual(fixed, "Read `design/AUDIO/README.md`, then ../design/AUDIO/README.md and redesign/AUDIO.md.\n")

    def test_rewrites_a_path_and_a_link_to_the_new_file(self) -> None:
        """Rewrites a path citation and a link to a file the caller moved, keeping the link's anchor."""
        moves = {"docs/design/AUDIO/MIXER.md": "docs/design/SOUND/MIXER.md"}
        fixed, _ = repair_text(
            K_USE, "See docs/design/AUDIO/MIXER.md and [m](../design/AUDIO/MIXER.md#gain).\n", K_PRESENT, K_KNOWN, moves,
        )
        self.assertEqual(fixed, "See docs/design/SOUND/MIXER.md and [m](../design/SOUND/MIXER.md#gain).\n")

    def test_leaves_other_files_of_the_topic_alone(self) -> None:
        """Does not touch a citation of a file that was not named in the move."""
        moves = {"docs/design/AUDIO/MIXER.md": "docs/design/SOUND/MIXER.md"}
        self.assertEqual(repair_text(K_USE, "See docs/design/AUDIO/README.md.\n", K_PRESENT, K_KNOWN, moves)[1], [])

    def test_points_a_citation_at_the_topic_under_revision(self) -> None:
        """Rewrites a page, a folder and a README citation to the topic's place under the revision folder."""
        fixed, _ = repair_text(
            K_USE,
            "See docs/design/AUDIO/MIXER.md, docs/design/AUDIO and [audio](../design/AUDIO/README.md).\n",
            K_REVISED,
            K_REVISED_KNOWN,
        )
        self.assertEqual(
            fixed,
            "See docs/design/_REVISION/AUDIO/MIXER.md, docs/design/_REVISION/AUDIO and "
            "[audio](../design/_REVISION/AUDIO/README.md).\n",
        )

    def test_reads_the_links_of_a_moved_page_from_its_former_place(self) -> None:
        """Rewrites the links a page wrote before it moved and leaves the link inside its own topic."""
        fixed, _ = repair_text(
            "docs/design/_REVISION/AUDIO/README.md",
            "See [ecs](../ECS.md), [use](../../guides/use.md) and [mixer](MIXER.md).\n",
            K_REVISED,
            K_REVISED_KNOWN,
        )
        self.assertEqual(fixed, "See [ecs](../../ECS.md), [use](../../../guides/use.md) and [mixer](MIXER.md).\n")

    def test_reads_the_links_of_a_page_that_left_the_revision_folder(self) -> None:
        """Rewrites a link written from the revision folder once the topic is back in docs/design."""
        fixed, _ = repair_text("docs/design/AUDIO/MIXER.md", "See [ecs](../../ECS.md).\n", K_PRESENT, K_KNOWN)
        self.assertEqual(fixed, "See [ecs](../ECS.md).\n")


class RunTest(unittest.TestCase):
    """Covers the disk: the files written, the archive and the fixtures left alone, the command and its exit code."""

    K_TEST_SIZE = "medium"

    def _root(self, folder: str, files: dict[str, str] | None = None) -> Path:
        """Writes the given files, or the default ones, under a root and returns it."""
        root = Path(folder) / "alpha"
        for name, text in (K_FILES if files is None else files).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def test_apply_writes_the_repaired_citations_to_disk(self) -> None:
        """Rewrites a file citation, a folder citation and a source comment in the files themselves."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            repairs = run(root, True)
            self.assertIn("docs/design/AUDIO/README.md", (root / "docs/guides/use.md").read_text(encoding="utf-8"))
            self.assertIn("[the ecs](../design/ECS.md#phases)", (root / "docs/guides/use.md").read_text(encoding="utf-8"))
            self.assertIn("docs/design/ECS.md.", (root / "modules/core/source/mixer.cpp").read_text(encoding="utf-8"))
            self.assertTrue(any(item.old == "docs/design/AUDIO.md" for item in repairs))

    def test_writes_nothing_without_apply(self) -> None:
        """Lists the repairs and leaves every file as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            before = (root / "docs/guides/use.md").read_text(encoding="utf-8")
            self.assertTrue(run(root, False))
            self.assertEqual((root / "docs/guides/use.md").read_text(encoding="utf-8"), before)

    def test_leaves_the_archive_alone(self) -> None:
        """Reads no archive file and changes no citation that already points at a file."""
        files = {**K_TOPICS, "docs/archive/old.md": "See docs/design/AUDIO.md.\n", "docs/guides/use.md": "See docs/design/AUDIO/MIXER.md.\n"}
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder, files)
            self.assertEqual(run(root, False), [])
            self.assertEqual(run(root, True), [])
            self.assertEqual((root / "docs/archive/old.md").read_text(encoding="utf-8"), "See docs/design/AUDIO.md.\n")
            self.assertEqual((root / "docs/guides/use.md").read_text(encoding="utf-8"), "See docs/design/AUDIO/MIXER.md.\n")

    def test_leaves_the_fixtures_of_the_checker_tests_alone(self) -> None:
        """Reports and writes nothing for a citation inside tools/tests."""
        files = {**K_TOPICS, "tools/tests/test_x.py": 'TEXT = "docs/design/AUDIO.md"\n'}
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder, files)
            self.assertEqual(run(root, True), [])
            self.assertIn("docs/design/AUDIO.md", (root / "tools/tests/test_x.py").read_text(encoding="utf-8"))

    def test_the_command_reads_each_move_pair(self) -> None:
        """Takes --move twice and applies both pairs."""
        files = {**K_TOPICS, "docs/guides/use.md": "docs/design/AUDIO/MIXER.md docs/design/AUDIO/README.md\n"}
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder, files)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(root), "--apply", "--move", "docs/design/AUDIO/MIXER.md", "docs/design/S/M.md", "--move", "docs/design/AUDIO/README.md", "docs/design/S/R.md"])
            self.assertEqual(code, 0)
            self.assertEqual((root / "docs/guides/use.md").read_text(encoding="utf-8"), "docs/design/S/M.md docs/design/S/R.md\n")

    def test_exits_one_only_for_a_citation_it_cannot_place(self) -> None:
        """Returns 0 after a plain repair and 1 while a citation is unplaced."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._root(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root), "--apply"]), 0)
                (root / "docs/guides/other.md").write_text("docs/design/GHOST.md\n", encoding="utf-8")
                self.assertEqual(main([str(root)]), 1)


if __name__ == "__main__":
    unittest.main()

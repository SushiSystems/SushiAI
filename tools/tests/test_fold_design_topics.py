# test_fold_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests giving each loose design document a folder of its own."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.design_status import Tree, read_tree  # noqa: E402
from documentation.fold_design_topics import fold, folded_name, loose_documents, main, plan, relink  # noqa: E402

K_FILES = {
    "docs/design/README.md": "# Design\n\nSee [audio](AUDIO.md).\n",
    "docs/design/REMAINING_WORK.md": "# Remaining work\n",
    "docs/design/AUDIO.md": "# Audio\n\n**Status:** Open — x\n\nSee [ecs](ECS.md#phases), [module](../../modules/audio/README.md) and [web](https://x.io/a.md).\n",
    "docs/design/ECS.md": "# Ecs\n\nBack to [audio](AUDIO.md).\n",
    "docs/design/SPLIT/README.md": "# Split\n",
    "docs/guides/use.md": "Read docs/design/AUDIO.md and [it](../design/AUDIO.md).\n",
    "modules/audio/README.md": "# Audio\n",
}

K_LOOSE = {
    "docs/design/AUDIO.md": "docs/design/AUDIO/README.md",
    "docs/design/ECS.md": "docs/design/ECS/README.md",
}


def _root(folder: str) -> Path:
    """Writes the files under a root and returns it."""
    root = Path(folder) / "alpha"
    for name, text in K_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _other_spelling(test: unittest.TestCase, folder: Path) -> Path:
    """Returns another name of *folder*: its 8.3 short name on Windows, a symbolic link elsewhere."""
    if os.name == "nt":
        import ctypes
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(folder), buffer, len(buffer)) and buffer.value != str(folder):
            return Path(buffer.value)
        test.skipTest("the volume makes no short names")
    link = folder.parent / "linked"
    try:
        os.symlink(folder, link, target_is_directory=True)
    except OSError as error:
        test.skipTest(f"no symbolic link here: {error}")
    return link


class FoldPlanTest(unittest.TestCase):
    """Covers the folder each loose document is folded into."""

    K_TEST_SIZE = "small"

    def test_names_the_readme_of_a_folder_named_after_the_document(self) -> None:
        """Puts the document's name, without its extension, in front of README.md."""
        self.assertEqual(folded_name("docs/design/AUDIO.md"), "docs/design/AUDIO/README.md")

    def test_keeps_the_dots_inside_the_name(self) -> None:
        """Strips only the final extension, as Path.with_suffix does."""
        self.assertEqual(folded_name("docs/design/A.B.md"), "docs/design/A.B/README.md")

    def test_plans_one_move_per_document(self) -> None:
        """Returns the moves for the loose documents, old to new."""
        self.assertEqual(plan(["docs/design/AUDIO.md", "docs/design/ECS.md"]), K_LOOSE)


class LooseDocumentsTest(unittest.TestCase):
    """Covers which files count as loose, read from the values of a tree."""

    K_TEST_SIZE = "small"

    def test_skips_the_indexes_and_the_folders(self) -> None:
        """Lists only the topic documents that sit directly in docs/design."""
        tree = Tree(
            folders=frozenset({"docs", "docs/design", "docs/design/SPLIT"}),
            files=frozenset({
                "docs/design/README.md", "docs/design/REMAINING_WORK.md", "docs/design/AUDIO.md",
                "docs/design/ECS.md", "docs/design/SPLIT/README.md",
            }),
            texts={},
        )
        self.assertEqual(loose_documents(tree), ["docs/design/AUDIO.md", "docs/design/ECS.md"])


class RelinkTest(unittest.TestCase):
    """Covers the links out of a moved file, as names."""

    K_TEST_SIZE = "small"

    def test_keeps_a_link_to_the_same_file_and_leaves_external_ones(self) -> None:
        """Adds a level to a relative link, keeps its anchor and does not touch a web link."""
        text = "[a](../../x/R.md#top) [b](https://x.io/a.md) [c](#here)\n"
        self.assertEqual(relink(text, "docs/design", "docs/design/AUDIO"), "[a](../../../x/R.md#top) [b](https://x.io/a.md) [c](#here)\n")

    def test_a_link_into_the_moved_folder_stays_within_it(self) -> None:
        """Rewrites a link to a sibling file so that it still reaches the same file from the new folder."""
        self.assertEqual(relink("[e](ECS.md)\n", "docs/design", "docs/design/AUDIO"), "[e](../ECS.md)\n")


class FoldTest(unittest.TestCase):
    """Covers the move on disk and what it rewrites."""

    K_TEST_SIZE = "medium"

    def test_a_dry_run_changes_nothing(self) -> None:
        """Returns the moves and leaves the files where they are."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            self.assertEqual(fold(root, False), K_LOOSE)
            self.assertTrue((root / "docs/design/AUDIO.md").is_file())

    def test_apply_moves_each_document_into_its_folder(self) -> None:
        """Leaves no loose topic document and keeps the indexes."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            fold(root, True)
            self.assertEqual(loose_documents(read_tree(root)), [])
            self.assertTrue((root / "docs/design/AUDIO/README.md").is_file())
            self.assertTrue((root / "docs/design/README.md").is_file())

    def test_apply_rewrites_links_out_of_the_moved_file_and_into_it(self) -> None:
        """Repoints the moved file's links, its siblings' links and a citation elsewhere."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            fold(root, True)
            audio = (root / "docs/design/AUDIO/README.md").read_text(encoding="utf-8")
            self.assertIn("[ecs](../ECS/README.md#phases)", audio)
            self.assertIn("[module](../../../modules/audio/README.md)", audio)
            self.assertIn("[web](https://x.io/a.md)", audio)
            self.assertIn("[audio](../AUDIO/README.md)", (root / "docs/design/ECS/README.md").read_text(encoding="utf-8"))
            self.assertIn("[audio](AUDIO/README.md)", (root / "docs/design/README.md").read_text(encoding="utf-8"))
            self.assertEqual(
                (root / "docs/guides/use.md").read_text(encoding="utf-8"),
                "Read docs/design/AUDIO/README.md and [it](../design/AUDIO/README.md).\n",
            )

    def test_a_folder_reached_through_a_short_name_keeps_its_links_relative(self) -> None:
        """Folds a repository named by its other spelling and writes the link from the folder it was asked to fold."""
        with tempfile.TemporaryDirectory() as folder:
            real = Path(folder) / "a folder with a long name"
            (real / "docs/design").mkdir(parents=True)
            (real / "x").mkdir()
            (real / "x/R.md").write_text("# R\n", encoding="utf-8")
            (real / "docs/design/AUDIO.md").write_text("[a](../../x/R.md)\n", encoding="utf-8")
            fold(_other_spelling(self, real), True)
            self.assertEqual((real / "docs/design/AUDIO/README.md").read_text(encoding="utf-8"), "[a](../../../x/R.md)\n")

    def test_the_command_exits_one_only_for_a_dry_run_with_work(self) -> None:
        """Returns 1 for a dry run that finds a loose document and 0 after folding."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root)]), 1)
                self.assertEqual(main([str(root), "--apply"]), 0)
                self.assertEqual(main([str(root)]), 0)


if __name__ == "__main__":
    unittest.main()

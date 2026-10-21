# test_fold_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests giving each loose design document a folder of its own."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.fold_design_topics import fold, loose_documents, main, relink  # noqa: E402

K_FILES = {
    "docs/design/README.md": "# Design\n\nSee [audio](AUDIO.md).\n",
    "docs/design/REMAINING_WORK.md": "# Remaining work\n",
    "docs/design/AUDIO.md": "# Audio\n\n**Status:** Open — x\n\nSee [ecs](ECS.md#phases), [module](../../modules/audio/README.md) and [web](https://x.io/a.md).\n",
    "docs/design/ECS.md": "# Ecs\n\nBack to [audio](AUDIO.md).\n",
    "docs/design/SPLIT/README.md": "# Split\n",
    "docs/guides/use.md": "Read docs/design/AUDIO.md and [it](../design/AUDIO.md).\n",
    "modules/audio/README.md": "# Audio\n",
}


def _root(folder: str) -> Path:
    """Writes the files under a root and returns it."""
    root = Path(folder) / "alpha"
    for name, text in K_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class LooseDocumentsTest(unittest.TestCase):
    """Covers which files count as loose."""

    def test_skips_the_indexes_and_the_folders(self) -> None:
        """Lists only the topic documents that sit directly in docs/design."""
        with tempfile.TemporaryDirectory() as folder:
            names = [path.name for path in loose_documents(_root(folder))]
            self.assertEqual(names, ["AUDIO.md", "ECS.md"])


class RelinkTest(unittest.TestCase):
    """Covers the links out of a moved file."""

    def test_keeps_a_link_to_the_same_file_and_leaves_external_ones(self) -> None:
        """Adds a level to a relative link, keeps its anchor and does not touch a web link."""
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "docs/design"
            text = "[a](../../x/R.md#top) [b](https://x.io/a.md) [c](#here)\n"
            self.assertEqual(relink(text, old, old / "AUDIO"), "[a](../../../x/R.md#top) [b](https://x.io/a.md) [c](#here)\n")


class FoldTest(unittest.TestCase):
    """Covers the move and what it rewrites."""

    def test_a_dry_run_changes_nothing(self) -> None:
        """Returns the moves and leaves the files where they are."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            self.assertEqual(fold(root, False), {"docs/design/AUDIO.md": "docs/design/AUDIO/README.md", "docs/design/ECS.md": "docs/design/ECS/README.md"})
            self.assertTrue((root / "docs/design/AUDIO.md").is_file())

    def test_apply_moves_each_document_into_its_folder(self) -> None:
        """Leaves no loose topic document and keeps the indexes."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            fold(root, True)
            self.assertEqual(loose_documents(root), [])
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

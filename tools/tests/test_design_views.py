# test_design_views.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the generated views of the design topics and the generator that writes them."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.design_status import Tree  # noqa: E402
from documentation.design_views import (  # noqa: E402
    expected_blocks,
    insert_block,
    read_block,
    render_module_designs,
    render_topic_map,
    render_topic_work,
    write_block,
)
from documentation.generate_design_views import main  # noqa: E402

K_TOPIC = """# Audio

**Status:** Open — phase A1 is building
**About:** The sound the engine makes.
**Modules:** modules/domain/audio

## Problem

None.

## Decision

None.

## Design

None.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| A0 | The mixer | built |
| A1 | The spatializer | building |

## Open questions

None.
"""

K_OPEN_WORK = """# Spatializer

**Status:** Open — under way
**Implements:** AUDIO#A1

## Problem

None.
"""

K_CLOSED_WORK = K_OPEN_WORK.replace("Open — under way", "Shipped — 2026-10-01")

K_FILES = {
    "docs/design/README.md": "# Design\n\nThe topics.\n",
    "docs/design/AUDIO.md": K_TOPIC,
    "docs/agent/2026_10_20_SPATIALIZER/SPEC.md": K_OPEN_WORK,
    "docs/agent/2026_09_01_MIXER/SPEC.md": K_CLOSED_WORK.replace("AUDIO#A1", "AUDIO#A0"),
    "modules/domain/audio/README.md": "# Audio\n\nThe module.\n",
}


def _tree(files: dict[str, str]) -> Tree:
    """Returns the folders, files and texts of a set of files, as the shell would read them."""
    folders = {"docs"}
    for name in files:
        folders.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
    return Tree(frozenset(folders), frozenset(files), files)


def _write(folder: str, files: dict[str, str]) -> Path:
    """Writes the given files under a root and returns the root."""
    root = Path(folder) / "alpha"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class RenderTest(unittest.TestCase):
    """Covers the text of each view, rendered from the texts of a tree."""
    K_TEST_SIZE = "small"

    def test_renders_the_topic_map(self) -> None:
        """Lists each topic with its link, state and About line."""
        self.assertEqual(
            render_topic_map(_tree(K_FILES), "docs", "docs/design/README.md").split("\n")[2],
            "| [AUDIO](AUDIO.md) | Open | The sound the engine makes. |",
        )

    def test_marks_a_topic_under_revision(self) -> None:
        """Lists a topic in the revision folder with its link there and the state Revision."""
        tree = _tree({**K_FILES, "docs/design/_REVISION/VIDEO/README.md": K_TOPIC})
        self.assertIn(
            "| [VIDEO](_REVISION/VIDEO/README.md) | Revision | The sound the engine makes. |",
            render_topic_map(tree, "docs", "docs/design/README.md").split("\n"),
        )

    def test_says_when_a_topic_has_no_about_line(self) -> None:
        """Writes a fixed word for a topic that has not described itself."""
        tree = _tree({"docs/design/AUDIO.md": K_TOPIC.replace("**About:** The sound the engine makes.\n", "")})
        self.assertIn("No description.", render_topic_map(tree, "docs", "docs/design/README.md"))

    def test_lists_the_topics_of_a_module(self) -> None:
        """Lists the topics whose Modules line names the module, as links from its README."""
        body = render_module_designs(_tree(K_FILES), "docs", "modules/domain/audio", "modules/domain/audio/README.md")
        self.assertEqual(body, "- [AUDIO](../../../docs/design/AUDIO.md): The sound the engine makes.")

    def test_lists_only_open_work_that_names_the_topic(self) -> None:
        """Lists an open folder with its phases and leaves a closed one out."""
        body = render_topic_work(_tree(K_FILES), "docs", "AUDIO", "docs/design/AUDIO.md")
        self.assertEqual(body, "- [2026_10_20_SPATIALIZER](../agent/2026_10_20_SPATIALIZER/SPEC.md): A1")

    def test_says_when_no_work_is_open(self) -> None:
        """Writes a fixed sentence when no open folder names the topic."""
        tree = _tree({key: value for key, value in K_FILES.items() if "SPATIALIZER" not in key})
        self.assertEqual(render_topic_work(tree, "docs", "AUDIO", "docs/design/AUDIO.md"), "No open work.")

    def test_names_the_topic_map_the_module_and_the_work_blocks(self) -> None:
        """Names the topic map, the module block and the work block of a tree."""
        names = sorted(block for _, block, _ in expected_blocks(_tree(K_FILES)))
        self.assertEqual(names, ["designs", "topics", "work"])


class BlockTest(unittest.TestCase):
    """Covers reading and writing between markers, on the text of a file."""
    K_TEST_SIZE = "small"

    def test_returns_none_without_markers(self) -> None:
        """Reads no block from a text that has no markers."""
        self.assertIsNone(read_block("# Design\n", "topics"))

    def test_writes_between_the_markers_and_keeps_the_rest(self) -> None:
        """Replaces what is between the markers and nothing else, and does nothing again on the same body."""
        text = "head\n<!-- topics:begin -->\nold\n<!-- topics:end -->\ntail\n"
        written = write_block(text, "topics", "new one\nnew two")
        self.assertEqual(written, "head\n<!-- topics:begin -->\nnew one\nnew two\n<!-- topics:end -->\ntail\n")
        self.assertEqual(write_block(written, "topics", "new one\nnew two"), written)

    def test_leaves_a_text_without_markers_as_it_was(self) -> None:
        """Returns the text itself when the text carries no such block."""
        self.assertEqual(write_block("plain\n", "topics", "new"), "plain\n")

    def test_inserts_empty_blocks(self) -> None:
        """Adds a block under a heading, and into the Phases section of a topic, once."""
        index = K_FILES["docs/design/README.md"]
        added = insert_block(index, "topics", "## Topics")
        self.assertIsNotNone(added)
        self.assertIsNone(insert_block(added, "topics", "## Topics"))
        topic = insert_block(K_TOPIC, "work", None)
        self.assertIsNotNone(topic)
        self.assertLess(topic.index("### Work"), topic.index("## Open questions"))
        self.assertGreater(topic.index("### Work"), topic.index("| A1 |"))


class GeneratorTest(unittest.TestCase):
    """Covers the command that writes the views to the files on disk."""
    K_TEST_SIZE = "medium"

    def test_initialises_then_fills_every_block(self) -> None:
        """Adds the markers and writes the three views."""
        with tempfile.TemporaryDirectory() as folder:
            root = _write(folder, K_FILES)
            self.assertEqual(main([str(root), "--init"]), 0)
            self.assertIn("[AUDIO](AUDIO.md)", read_block((root / "docs/design/README.md").read_text(encoding="utf-8"), "topics") or "")
            self.assertIn("[AUDIO]", read_block((root / "modules/domain/audio/README.md").read_text(encoding="utf-8"), "designs") or "")
            self.assertIn("SPATIALIZER", read_block((root / "docs/design/AUDIO.md").read_text(encoding="utf-8"), "work") or "")

    def test_check_reports_a_stale_block_and_writes_nothing(self) -> None:
        """Exits 1 for a stale block and leaves the file as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = _write(folder, K_FILES)
            main([str(root), "--init"])
            path = root / "docs/design/README.md"
            path.write_text(path.read_text(encoding="utf-8").replace("The sound the engine makes.", "stale text"), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(main([str(root), "--check"]), 1)
            self.assertIn("the block 'topics' is stale", printed.getvalue())
            self.assertIn("stale text", path.read_text(encoding="utf-8"))
            self.assertEqual(main([str(root)]), 0)
            self.assertEqual(main([str(root), "--check"]), 0)

    def test_leaves_a_file_without_markers_alone(self) -> None:
        """Does not write into a README that has no markers unless asked to initialise."""
        with tempfile.TemporaryDirectory() as folder:
            root = _write(folder, K_FILES)
            before = (root / "docs/design/README.md").read_text(encoding="utf-8")
            main([str(root)])
            self.assertEqual((root / "docs/design/README.md").read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _root(folder: str, files: dict[str, str] | None = None) -> Path:
    """Writes the given files, or the default ones, under a root and returns it."""
    root = Path(folder) / "alpha"
    for name, text in (K_FILES if files is None else files).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class RenderTest(unittest.TestCase):
    """Covers the text of each view."""

    def test_renders_the_topic_map(self) -> None:
        """Lists each topic with its link, state and About line."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            index = root / "docs/design/README.md"
            self.assertEqual(
                render_topic_map(root / "docs", index).split("\n")[2],
                "| [AUDIO](AUDIO.md) | Open | The sound the engine makes. |",
            )

    def test_says_when_a_topic_has_no_about_line(self) -> None:
        """Writes a fixed word for a topic that has not described itself."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder, {"docs/design/AUDIO.md": K_TOPIC.replace("**About:** The sound the engine makes.\n", "")})
            self.assertIn("No description.", render_topic_map(root / "docs", root / "docs/design/README.md"))

    def test_lists_the_topics_of_a_module(self) -> None:
        """Lists the topics whose Modules line names the module, as links from its README."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            readme = root / "modules/domain/audio/README.md"
            body = render_module_designs(root / "docs", "modules/domain/audio", readme)
            self.assertEqual(body, "- [AUDIO](../../../docs/design/AUDIO.md): The sound the engine makes.")

    def test_lists_only_open_work_that_names_the_topic(self) -> None:
        """Lists an open folder with its phases and leaves a closed one out."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            body = render_topic_work(root / "docs", "AUDIO", root / "docs/design/AUDIO.md")
            self.assertEqual(body, "- [2026_10_20_SPATIALIZER](../agent/2026_10_20_SPATIALIZER/SPEC.md): A1")

    def test_says_when_no_work_is_open(self) -> None:
        """Writes a fixed sentence when no open folder names the topic."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder, {key: value for key, value in K_FILES.items() if "SPATIALIZER" not in key})
            self.assertEqual(render_topic_work(root / "docs", "AUDIO", root / "docs/design/AUDIO.md"), "No open work.")


class BlockTest(unittest.TestCase):
    """Covers reading and writing between markers."""

    def test_returns_none_without_markers(self) -> None:
        """Reads no block from a file that has no markers."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(read_block(_root(folder) / "docs/design/README.md", "topics"))

    def test_writes_between_the_markers_and_keeps_the_rest(self) -> None:
        """Replaces what is between the markers and nothing else."""
        with tempfile.TemporaryDirectory() as folder:
            path = _root(folder, {"x.md": "head\n<!-- topics:begin -->\nold\n<!-- topics:end -->\ntail\n"}) / "x.md"
            self.assertTrue(write_block(path, "topics", "new one\nnew two"))
            self.assertEqual(path.read_text(encoding="utf-8"), "head\n<!-- topics:begin -->\nnew one\nnew two\n<!-- topics:end -->\ntail\n")
            self.assertFalse(write_block(path, "topics", "new one\nnew two"))

    def test_inserts_empty_blocks(self) -> None:
        """Adds a block under a heading, and into the Phases section of a topic."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            self.assertTrue(insert_block(root / "docs/design/README.md", "topics", "## Topics"))
            self.assertFalse(insert_block(root / "docs/design/README.md", "topics", "## Topics"))
            self.assertTrue(insert_block(root / "docs/design/AUDIO.md", "work", None))
            text = (root / "docs/design/AUDIO.md").read_text(encoding="utf-8")
            self.assertLess(text.index("### Work"), text.index("## Open questions"))
            self.assertGreater(text.index("### Work"), text.index("| A1 |"))


class GeneratorTest(unittest.TestCase):
    """Covers the command that writes the views."""

    def test_initialises_then_fills_every_block(self) -> None:
        """Adds the markers and writes the three views."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
            self.assertEqual(main([str(root), "--init"]), 0)
            self.assertIn("[AUDIO](AUDIO.md)", read_block(root / "docs/design/README.md", "topics") or "")
            self.assertIn("[AUDIO]", read_block(root / "modules/domain/audio/README.md", "designs") or "")
            self.assertIn("SPATIALIZER", read_block(root / "docs/design/AUDIO.md", "work") or "")

    def test_check_reports_a_stale_block_and_writes_nothing(self) -> None:
        """Exits 1 for a stale block and leaves the file as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = _root(folder)
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
            root = _root(folder)
            before = (root / "docs/design/README.md").read_text(encoding="utf-8")
            main([str(root)])
            self.assertEqual((root / "docs/design/README.md").read_text(encoding="utf-8"), before)

    def test_lists_a_block_for_each_place_it_can_live(self) -> None:
        """Names the topic map, the module block and the work block."""
        with tempfile.TemporaryDirectory() as folder:
            names = sorted(name for _, name, _ in expected_blocks(_root(folder)))
            self.assertEqual(names, ["designs", "topics", "work"])


if __name__ == "__main__":
    unittest.main()

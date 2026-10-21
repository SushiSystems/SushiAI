# test_check_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of the design topics checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_design_topics import (  # noqa: E402
    K_CHECKER,
    Repository,
    rule_archive_not_cited,
    rule_designs_resolve,
    rule_folder_ceiling,
    rule_glossary_entry,
    rule_implements_resolve,
    rule_modules_exist,
    rule_modules_required,
    rule_topic_folder,
    rule_views_current,
    rule_views_present,
)

K_TOPIC = """# Audio

**Status:** Open — phase A1 is building
**Modules:** modules/domain/audio, modules/domain/dsp

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

K_WORK = """# Spatializer

**Status:** Open — under way
**Implements:** AUDIO#A1, AUDIO#A0
**Designs:** AUDIO

## Problem

None.
"""

K_FILES = {
    "docs/design/AUDIO.md": K_TOPIC,
    "docs/agent/2026_10_21_SPATIALIZER/SPEC.md": K_WORK,
    "modules/domain/audio/README.md": "# Audio\n",
    "modules/domain/dsp/README.md": "# Dsp\n",
}


def _repository(folder: str, files: dict[str, str], is_required: bool = False, ceiling: int = 0, untracked: bool = False) -> Repository:
    """Writes the given files under a root and returns its repository."""
    root = Path(folder) / "alpha"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root, is_required, ceiling, untracked)


def _messages(issues: Iterable[Issue]) -> list[str]:
    """Returns the message of each issue."""
    return [issue.message for issue in issues]


def _check(rule, overrides: dict[str, str] | None = None, is_required: bool = False, ceiling: int = 0, untracked: bool = False) -> list[str]:
    """Returns the messages a rule gives for the valid files with some of them replaced."""
    with tempfile.TemporaryDirectory() as folder:
        files = {**K_FILES, **(overrides or {})}
        return _messages(rule(_repository(folder, files, is_required, ceiling, untracked)))


class RegistrationTest(unittest.TestCase):
    """Checks the rule table."""

    def test_names_every_rule(self) -> None:
        """Holds the nine rules under their names."""
        self.assertEqual(
            sorted(K_CHECKER.rules),
            [
                "rule_archive_not_cited",
                "rule_designs_resolve",
                "rule_folder_ceiling",
                "rule_glossary_entry",
                "rule_implements_resolve",
                "rule_modules_exist",
                "rule_modules_required",
                "rule_topic_folder",
                "rule_views_current",
                "rule_views_present",
            ],
        )

    def test_reports_nothing_for_valid_files_with_every_rule_on(self) -> None:
        """Runs every rule over the valid files, the topic in its folder, with the strict settings on."""
        files = {key: value for key, value in K_FILES.items() if key != "docs/design/AUDIO.md"}
        files["docs/design/AUDIO/README.md"] = K_TOPIC
        files["docs/reference/GLOSSARY.md"] = "# Glossary\n\n**audio**: the sound of the engine.\n"
        for name, rule in K_CHECKER.rules.items():
            with tempfile.TemporaryDirectory() as folder:
                self.assertEqual(_messages(rule(_repository(folder, files, True, 10, True))), [], name)


class TopicFolderTest(unittest.TestCase):
    """Covers the rule that a topic is a folder."""

    def test_is_silent_when_not_asked(self) -> None:
        """Reports nothing for a loose document while the repository does not ask for topics."""
        self.assertEqual(_check(rule_topic_folder), [])

    def test_reports_a_loose_document_when_asked(self) -> None:
        """Reports the document that sits directly in docs/design, and not the indexes."""
        files = {"docs/design/README.md": "# Design\n", "docs/design/REMAINING_WORK.md": "# Work\n"}
        messages = _check(rule_topic_folder, files, True)
        self.assertEqual(messages, ["a topic is a folder: move it to AUDIO/README.md with fold_design_topics.py"])


class ModulesExistTest(unittest.TestCase):
    """Covers the Modules line."""

    def test_reports_a_module_that_is_not_there(self) -> None:
        """Reports a listed module that has no folder."""
        text = K_TOPIC.replace("modules/domain/dsp", "modules/domain/missing")
        self.assertEqual(_check(rule_modules_exist, {"docs/design/AUDIO.md": text}), ["the module 'modules/domain/missing' is not in the repository"])

    def test_accepts_none_and_a_missing_line(self) -> None:
        """Accepts a topic that says none and one without the line."""
        none = K_TOPIC.replace("modules/domain/audio, modules/domain/dsp", "none")
        absent = K_TOPIC.replace("**Modules:** modules/domain/audio, modules/domain/dsp\n", "")
        self.assertEqual(_check(rule_modules_exist, {"docs/design/AUDIO.md": none}), [])
        self.assertEqual(_check(rule_modules_exist, {"docs/design/AUDIO.md": absent}), [])

    def test_reads_code_spans(self) -> None:
        """Reads a module written in a code span."""
        text = K_TOPIC.replace("modules/domain/audio", "`modules/domain/audio`")
        self.assertEqual(_check(rule_modules_exist, {"docs/design/AUDIO.md": text}), [])


class ModulesRequiredTest(unittest.TestCase):
    """Covers the opt-in rule."""

    ABSENT = K_TOPIC.replace("**Modules:** modules/domain/audio, modules/domain/dsp\n", "")

    def test_is_silent_when_not_asked(self) -> None:
        """Reports nothing while the repository does not ask for the line."""
        self.assertEqual(_check(rule_modules_required, {"docs/design/AUDIO.md": self.ABSENT}), [])

    def test_reports_a_topic_without_the_line_when_asked(self) -> None:
        """Reports a topic with no Modules line once the repository asks for it."""
        self.assertEqual(len(_check(rule_modules_required, {"docs/design/AUDIO.md": self.ABSENT}, True)), 1)


class ImplementsTest(unittest.TestCase):
    """Covers the links from work to phases."""

    def test_reports_an_unknown_topic(self) -> None:
        """Reports a work folder that implements a topic nobody wrote."""
        text = K_WORK.replace("AUDIO#A1, AUDIO#A0", "VIDEO#V1")
        messages = _check(rule_implements_resolve, {"docs/agent/2026_10_21_SPATIALIZER/SPEC.md": text})
        self.assertEqual(messages, ["'VIDEO' is not a design topic"])

    def test_reports_an_unknown_phase(self) -> None:
        """Reports a phase id the topic's table does not hold."""
        text = K_WORK.replace("AUDIO#A1, AUDIO#A0", "AUDIO#A9")
        messages = _check(rule_implements_resolve, {"docs/agent/2026_10_21_SPATIALIZER/SPEC.md": text})
        self.assertEqual(messages, ["'AUDIO' has no phase 'A9'"])

    def test_reports_a_topic_without_a_phase(self) -> None:
        """Asks for the TOPIC#PHASE form."""
        text = K_WORK.replace("AUDIO#A1, AUDIO#A0", "AUDIO")
        messages = _check(rule_implements_resolve, {"docs/agent/2026_10_21_SPATIALIZER/SPEC.md": text})
        self.assertEqual(messages, ["'AUDIO' names a topic and no phase; write TOPIC#PHASE"])

    def test_reads_the_readme_of_a_split_topic(self) -> None:
        """Finds the phases of a split topic in its README."""
        overrides = {"docs/design/AUDIO/README.md": K_TOPIC}
        with tempfile.TemporaryDirectory() as folder:
            files = {key: value for key, value in K_FILES.items() if key != "docs/design/AUDIO.md"}
            files.update(overrides)
            self.assertEqual(_messages(rule_implements_resolve(_repository(folder, files))), [])

    def test_leaves_work_without_the_line_alone(self) -> None:
        """Does not ask an audit to implement a phase."""
        text = K_WORK.replace("**Implements:** AUDIO#A1, AUDIO#A0\n", "")
        self.assertEqual(_check(rule_implements_resolve, {"docs/agent/2026_10_21_SPATIALIZER/SPEC.md": text}), [])


class DesignsTest(unittest.TestCase):
    """Covers the links from work to whole topics."""

    def test_reports_an_unknown_topic(self) -> None:
        """Reports a Designs item that is not a topic."""
        text = K_WORK.replace("**Designs:** AUDIO", "**Designs:** AUDIO, VIDEO")
        messages = _check(rule_designs_resolve, {"docs/agent/2026_10_21_SPATIALIZER/SPEC.md": text})
        self.assertEqual(messages, ["'VIDEO' is not a design topic"])


class GlossaryTest(unittest.TestCase):
    """Covers the glossary term of each topic."""

    def test_is_silent_when_not_asked(self) -> None:
        """Reports nothing while the repository does not ask for glossary entries."""
        self.assertEqual(_check(rule_glossary_entry), [])

    def test_reports_a_topic_the_glossary_lacks(self) -> None:
        """Reports a topic with no term when the glossary exists and does not name it."""
        glossary = {"docs/reference/GLOSSARY.md": "# Glossary\n\n**mixer**: the sum of the sources.\n"}
        self.assertEqual(_check(rule_glossary_entry, glossary, True), ["the glossary has no entry for the topic 'AUDIO'"])

    def test_reports_every_topic_when_there_is_no_glossary(self) -> None:
        """Reports a topic when the repository has no glossary at all."""
        self.assertEqual(len(_check(rule_glossary_entry, None, True)), 1)

    def test_reads_a_table_cell_and_a_heading_with_separators(self) -> None:
        """Takes a table first cell, and a topic name written with underscores."""
        topic = {"docs/design/SPATIAL_AUDIO.md": K_TOPIC}
        cell = {"docs/reference/GLOSSARY.md": "| Word | Meaning |\n| --- | --- |\n| Audio | sound |\n| Spatial audio | sound in space |\n"}
        self.assertEqual(_check(rule_glossary_entry, {**topic, **cell}, True), [])

    def test_does_not_match_inside_a_longer_word(self) -> None:
        """Does not take 'audiology' for 'audio'."""
        glossary = {"docs/reference/GLOSSARY.md": "# Glossary\n\n**audiology**: the study of hearing.\n"}
        self.assertEqual(len(_check(rule_glossary_entry, glossary, True)), 1)


class FolderCeilingTest(unittest.TestCase):
    """Covers the entry limit of docs/design and docs/agent."""

    def test_is_off_at_zero(self) -> None:
        """Reports nothing when the ceiling is zero."""
        self.assertEqual(_check(rule_folder_ceiling, None, False, 0), [])

    def test_reports_a_folder_over_the_ceiling(self) -> None:
        """Reports the folder that holds more entries than the ceiling."""
        extra = {f"docs/agent/2026_10_2{index}_W/SPEC.md": "# W\n" for index in range(3)}
        messages = _check(rule_folder_ceiling, extra, False, 3)
        self.assertEqual(messages, ["4 entries; the ceiling is 3"])


class ArchiveCitationTest(unittest.TestCase):
    """Covers the ban on citing an archive the repository does not track."""

    CITING = {"docs/guides/use.md": "See docs/archive/design/OLD.md and [x](../archive/y.md).\nPlain line.\n"}

    def test_is_silent_while_the_archive_is_tracked(self) -> None:
        """Reports nothing for a repository that still tracks its archive."""
        self.assertEqual(_check(rule_archive_not_cited, self.CITING), [])

    def test_reports_each_line_that_cites_the_archive(self) -> None:
        """Reports the one line that cites the archive by path and by link."""
        messages = _check(rule_archive_not_cited, self.CITING, untracked=True)
        self.assertEqual(messages, ["the archive is not tracked; cite the topic that holds the decision"])

    def test_does_not_read_the_archive_itself(self) -> None:
        """Does not report a citation inside the archive."""
        files = {"docs/archive/old.md": "See docs/archive/other.md.\n"}
        self.assertEqual(_check(rule_archive_not_cited, files, untracked=True), [])


class ViewsTest(unittest.TestCase):
    """Covers the generated blocks."""

    BLOCKS = {"docs/design/README.md": "# Design\n\n<!-- topics:begin -->\nstale\n<!-- topics:end -->\n"}

    def test_reports_a_stale_block(self) -> None:
        """Reports a block whose text is not what the metadata gives."""
        messages = _check(rule_views_current, self.BLOCKS)
        self.assertEqual(messages, ["the generated block 'topics' is stale; run generate_design_views.py"])

    def test_leaves_a_file_without_markers_alone(self) -> None:
        """Reports nothing while no file carries a marker."""
        self.assertEqual(_check(rule_views_current, {"docs/design/README.md": "# Design\n"}), [])

    def test_asks_for_the_topic_map_when_topics_are_required(self) -> None:
        """Reports a design index with no topic map once the repository asks for topics."""
        overrides = {"docs/design/README.md": "# Design\n"}
        self.assertEqual(_check(rule_views_present, overrides), [])
        self.assertEqual(len(_check(rule_views_present, overrides, True)), 1)


if __name__ == "__main__":
    unittest.main()

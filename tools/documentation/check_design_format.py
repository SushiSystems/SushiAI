# check_design_format.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the design documents of a repository against the design format of the documentation skill.

Usage: python tools/documentation/check_design_format.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from documentation.design_status import design_documents, first_content_line  # noqa: E402

K_SECTIONS = ("Problem", "Decision", "Design", "Phases", "Open questions")
K_PHASES = "Phases"
K_PHASE_COLUMNS = ("Id", "Phase", "State")
K_PHASE_STATES = frozenset({"planned", "building", "built", "withdrawn"})
K_NO_PHASES = "None."
K_STATUS_CEILING = 160
K_FENCE = re.compile(r"^(`{3,}|~{3,})")
K_SECTION = re.compile(r"^## (\S.*?)\s*$")
K_TABLE_ROW = re.compile(r"^\|(.*)\|\s*$")
K_TABLE_RULE = re.compile(r"^\|[\s:|-]+\|\s*$")
K_HASH = re.compile(r"(?<![\w/.-])[0-9a-f]{7,40}(?![\w/-])")
K_EXECUTION_VERB = r"(?:built|landed|merged|shipped|completed|implemented|reviewed|finished|done)"
K_ISO_DATE = r"\d{4}-\d{2}-\d{2}"
K_EXECUTION_RECORD = re.compile(
    rf"\b{K_EXECUTION_VERB}\b[^.\n]{{0,60}}{K_ISO_DATE}|{K_ISO_DATE}[^.\n]{{0,40}}\b{K_EXECUTION_VERB}\b",
    re.IGNORECASE,
)
K_HASH_DIGIT = re.compile(r"\d")
K_HASH_LETTER = re.compile(r"[a-f]")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of one repository whose design documents are checked."""

    root: Path

    def documents(self) -> list[Path]:
        """Returns each design document that exists, counting a split topic's README as its document."""
        return [path for path in design_documents(self.root / "docs") if path.is_file()]


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree."""
    return [Repository(path) for path in paths if (path / "docs").is_dir()]


def _lines(path: Path) -> list[tuple[int, str]]:
    """Returns the number and text of each line of a document outside fenced blocks."""
    kept: list[tuple[int, str]] = []
    fence = ""
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        opener = K_FENCE.match(line.lstrip())
        if fence:
            fence = "" if opener and opener.group(1).startswith(fence) else fence
        elif opener:
            fence = opener.group(1)
        else:
            kept.append((number, line))
    return kept


def _sections(path: Path) -> list[tuple[int, str]]:
    """Returns the number and name of each `##` heading of a document."""
    found = ((number, K_SECTION.match(line)) for number, line in _lines(path))
    return [(number, match.group(1)) for number, match in found if match]


def _section_body(path: Path, name: str) -> list[tuple[int, str]]:
    """Returns the lines under the named `##` heading, up to the next one."""
    body: list[tuple[int, str]] = []
    inside = False
    for number, line in _lines(path):
        heading = K_SECTION.match(line)
        if heading:
            inside = heading.group(1) == name
        elif inside:
            body.append((number, line))
    return body


def _cells(line: str) -> list[str]:
    """Returns the trimmed cells of a table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def rule_sections(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each section that is not one of the five, each one missing, and an order that is wrong."""
    for path in repository.documents():
        found = _sections(path)
        names = [name for _, name in found]
        for number, name in found:
            if name not in K_SECTIONS:
                yield Issue(path, number, f"'## {name}' is not one of the sections: {', '.join(K_SECTIONS)}")
        for name in K_SECTIONS:
            if name not in names:
                yield Issue(path, 1, f"the section '## {name}' is missing")
        known = [name for name in names if name in K_SECTIONS]
        if len(set(known)) == len(K_SECTIONS) and known != list(K_SECTIONS):
            yield Issue(path, 1, f"the sections are in the order {', '.join(K_SECTIONS)}")


def rule_status_length(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each status line longer than the ceiling."""
    for path in repository.documents():
        number, line = first_content_line(path)
        if len(line) > K_STATUS_CEILING:
            yield Issue(path, number, f"status is {len(line)} characters; the ceiling is {K_STATUS_CEILING}")


def rule_phase_table(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a Phases section without its table, with other columns, or with an unknown state."""
    for path in repository.documents():
        heading = next((number for number, name in _sections(path) if name == K_PHASES), None)
        if heading is None:
            continue
        body = _section_body(path, K_PHASES)
        rows = [(number, line) for number, line in body if K_TABLE_ROW.match(line)]
        if not rows:
            if not any(line.strip() == K_NO_PHASES for _, line in body):
                yield Issue(path, heading, f"'## {K_PHASES}' holds a table of {', '.join(K_PHASE_COLUMNS)}, or '{K_NO_PHASES}'")
            continue
        number, header = rows[0]
        if tuple(_cells(header)) != K_PHASE_COLUMNS:
            yield Issue(path, number, f"the phase table has the columns {', '.join(K_PHASE_COLUMNS)}")
            continue
        for number, line in rows[1:]:
            if K_TABLE_RULE.match(line):
                continue
            cells = _cells(line)
            if len(cells) != len(K_PHASE_COLUMNS) or cells[-1].strip("`") not in K_PHASE_STATES:
                yield Issue(path, number, f"a phase state is one of {', '.join(sorted(K_PHASE_STATES))}")


def rule_no_commit_hashes(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each commit hash in a design document, in a code span or bare."""
    for path in repository.documents():
        for number, line in _lines(path):
            for match in K_HASH.finditer(line):
                if K_HASH_DIGIT.search(match.group()) and K_HASH_LETTER.search(match.group()):
                    yield Issue(path, number, "a commit hash belongs in the work folder's REPORT.md")


def rule_no_execution_record(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each line outside the status that dates when work was built, merged or reviewed."""
    for path in repository.documents():
        status_line, _ = first_content_line(path)
        for number, line in _lines(path):
            if number != status_line and K_EXECUTION_RECORD.search(line):
                yield Issue(path, number, "when work was done belongs in the work folder's REPORT.md")


K_CHECKER = Checker(
    description="Checks the design documents against the design format of the documentation skill.",
    collect=collect,
    rules={
        "rule_sections": rule_sections,
        "rule_status_length": rule_status_length,
        "rule_phase_table": rule_phase_table,
        "rule_no_commit_hashes": rule_no_commit_hashes,
        "rule_no_execution_record": rule_no_execution_record,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# design_status.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Finds the design documents and work folders of a docs/ tree and reads their status lines."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

K_DESIGN_INDEXES = frozenset({"README.md", "REMAINING_WORK.md"})
K_REVISION_FOLDER = "_REVISION"
K_WORK_FILES_IN_ORDER = ("SPEC.md", "PLAN.md", "REPORT.md")
K_FENCE = re.compile(r"^(`{3,}|~{3,})")
K_SECTION = re.compile(r"^## (\S.*?)\s*$")
K_TABLE_ROW = re.compile(r"^\|(.*)\|\s*$")
K_TABLE_RULE = re.compile(r"^\|[\s:|-]+\|\s*$")
K_STATUS_FORMS = re.compile(
    r"^\*\*Status:\*\* (?:(?P<open>Open) — \S"
    r"|Shipped — (?P<shipped>\d{4}-\d{2}-\d{2})(?:$|\. \S)"
    r"|Superseded — (?P<superseded>\d{4}-\d{2}-\d{2}) by \S"
    r"|Withdrawn — (?P<withdrawn>\d{4}-\d{2}-\d{2}), \S)"
)


@dataclass(frozen=True, slots=True)
class Status:
    """Holds the status line of one document and what its form says."""

    line: int
    is_known: bool
    is_open: bool
    closed_on: datetime.date | None
    is_shipped: bool = False


def design_documents(docs: Path) -> list[Path]:
    """Returns every design document, counting a split topic's README as its document and the topics under revision."""
    design = docs / "design"
    if not design.is_dir():
        return []
    revision = design / K_REVISION_FOLDER
    documents = [path for path in sorted(design.glob("*.md")) if path.name not in K_DESIGN_INDEXES]
    documents += [folder / "README.md" for folder in sorted(design.iterdir()) if folder.is_dir() and folder != revision]
    if revision.is_dir():
        documents += [folder / "README.md" for folder in sorted(revision.iterdir()) if folder.is_dir()]
    return documents


def is_under_revision(path: Path) -> bool:
    """Returns whether a design document sits in the folder of topics set aside for revision."""
    return K_REVISION_FOLDER in path.parts


def work_folders(docs: Path) -> list[Path]:
    """Returns every entry directly under docs/agent/."""
    agent = docs / "agent"
    return sorted(agent.iterdir()) if agent.is_dir() else []


def status_documents(docs: Path) -> list[Path]:
    """Returns each design document that exists and the first file of each work folder."""
    documents = [path for path in design_documents(docs) if path.is_file()]
    for folder in work_folders(docs):
        first = next((folder / name for name in K_WORK_FILES_IN_ORDER if (folder / name).is_file()), None)
        if first is not None:
            documents.append(first)
    return documents


def first_content_line(path: Path) -> tuple[int, str]:
    """Returns the number and text of the first non-empty line after the title."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines[1:], start=2):
        if line.strip():
            return index, line
    return 1, ""


def read_status(path: Path) -> Status:
    """Returns the status line of a document, unknown when it is in none of the four forms."""
    number, line = first_content_line(path)
    match = K_STATUS_FORMS.match(line)
    if match is None:
        return Status(number, False, False, None)
    if match["open"]:
        return Status(number, True, True, None)
    try:
        closed_on = datetime.date.fromisoformat(match["shipped"] or match["superseded"] or match["withdrawn"])
    except ValueError:
        return Status(number, False, False, None)
    return Status(number, True, False, closed_on, bool(match["shipped"]))


def body_lines(path: Path) -> list[tuple[int, str]]:
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


def section_names(path: Path) -> list[tuple[int, str]]:
    """Returns the number and name of each `##` heading of a document."""
    found = ((number, K_SECTION.match(line)) for number, line in body_lines(path))
    return [(number, match.group(1)) for number, match in found if match]


def section_body(path: Path, name: str) -> list[tuple[int, str]]:
    """Returns the lines under the named `##` heading, up to the next one."""
    body: list[tuple[int, str]] = []
    inside = False
    for number, line in body_lines(path):
        heading = K_SECTION.match(line)
        if heading:
            inside = heading.group(1) == name
        elif inside:
            body.append((number, line))
    return body


def table_cells(line: str) -> list[str]:
    """Returns the trimmed cells of a table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]

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
K_WORK_FILES_IN_ORDER = ("SPEC.md", "PLAN.md", "REPORT.md")
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


def design_documents(docs: Path) -> list[Path]:
    """Returns every design document, counting a split topic's README as its document."""
    design = docs / "design"
    if not design.is_dir():
        return []
    documents = [path for path in sorted(design.glob("*.md")) if path.name not in K_DESIGN_INDEXES]
    documents += [folder / "README.md" for folder in sorted(design.iterdir()) if folder.is_dir()]
    return documents


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
    return Status(number, True, False, closed_on)

# design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads the topic of a design document: its name, its metadata lines, its phase ids and the work that names it."""

from __future__ import annotations

import re
from pathlib import Path

from documentation.design_status import (
    K_DESIGN_INDEXES,
    K_TABLE_RULE,
    K_TABLE_ROW,
    K_WORK_FILES_IN_ORDER,
    design_documents,
    first_content_line,
    section_body,
    table_cells,
    work_folders,
)

K_FIELD = re.compile(r"^\*\*(?P<name>[A-Za-z][A-Za-z ]*):\*\*\s*(?P<value>.*?)\s*$")
K_NONE = "none"
K_STATUS_FIELD = "Status"
K_ABOUT_FIELD = "About"
K_MODULES_FIELD = "Modules"
K_IMPLEMENTS_FIELD = "Implements"
K_DESIGNS_FIELD = "Designs"
K_PHASES_SECTION = "Phases"
K_REFERENCE_SEPARATOR = "#"
K_VALUE_SEPARATOR = ","


def topic_name(path: Path) -> str:
    """Returns the topic a design document belongs to: its folder name for a split topic, its stem otherwise."""
    return path.parent.name if path.name == "README.md" else path.stem


def topics(docs: Path) -> dict[str, Path]:
    """Returns each topic of a docs/ tree by name, with the document that holds it."""
    return {topic_name(path): path for path in design_documents(docs) if path.is_file()}


def metadata(path: Path) -> dict[str, tuple[int, str]]:
    """Returns each `**Field:** value` line that follows the status line, by field name with its line number."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    status_line, _ = first_content_line(path)
    found: dict[str, tuple[int, str]] = {}
    for number, line in enumerate(lines[status_line:], start=status_line + 1):
        if not line.strip():
            continue
        match = K_FIELD.match(line)
        if match is None:
            break
        if match["name"] != K_STATUS_FIELD:
            found[match["name"]] = (number, match["value"])
    return found


def split_values(value: str) -> list[str]:
    """Returns the comma separated items of a metadata value, without code span marks."""
    items = (item.strip().strip("`").strip() for item in value.split(K_VALUE_SEPARATOR))
    return [item for item in items if item and item.lower() != K_NONE]


def phase_ids(path: Path) -> set[str]:
    """Returns the id of each row of the phase table of a design document."""
    found: set[str] = set()
    rows = [line for _, line in section_body(path, K_PHASES_SECTION) if K_TABLE_ROW.match(line)]
    for line in rows[1:]:
        if not K_TABLE_RULE.match(line):
            found.add(table_cells(line)[0].strip("`* "))
    return found


def loose_documents(root: Path) -> list[Path]:
    """Returns each design document that sits directly in docs/design and is not an index."""
    design = root / "docs" / "design"
    if not design.is_dir():
        return []
    return [path for path in sorted(design.glob("*.md")) if path.name not in K_DESIGN_INDEXES]


def work_documents(docs: Path) -> list[Path]:
    """Returns the first file of each work folder, the one that carries its status and metadata."""
    found: list[Path] = []
    for folder in work_folders(docs):
        first = next((folder / name for name in K_WORK_FILES_IN_ORDER if (folder / name).is_file()), None)
        if first is not None:
            found.append(first)
    return found


def split_reference(item: str) -> tuple[str, str]:
    """Returns the topic and the phase of a `TOPIC#PHASE` item; the phase is empty when there is none."""
    topic, _, phase = item.partition(K_REFERENCE_SEPARATOR)
    return topic.strip(), phase.strip()

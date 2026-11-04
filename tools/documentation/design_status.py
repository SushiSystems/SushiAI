# design_status.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Parses the status, sections and table rows of design documents and lists them from values; reads the tree once."""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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


@dataclass(frozen=True, slots=True)
class Tree:
    """Holds a tree as values: its folders and files as root-relative POSIX names, and the text of each file read."""

    folders: frozenset[str]
    files: frozenset[str]
    texts: Mapping[str, str]


def children(tree: Tree, folder: str) -> list[str]:
    """Returns the root-relative name of each folder and file directly inside a folder, sorted."""
    prefix = f"{folder}/"
    return sorted(
        name for name in (*tree.folders, *tree.files)
        if name.startswith(prefix) and "/" not in name.removeprefix(prefix)
    )


def design_documents(tree: Tree, docs: str = "docs") -> list[str]:
    """Returns every design document, a split topic's README counting as its document; topics under revision too."""
    design = f"{docs}/design"
    revision = f"{design}/{K_REVISION_FOLDER}"
    entries = children(tree, design)
    documents = [
        name for name in entries
        if PurePosixPath(name).name not in K_DESIGN_INDEXES
        and (name in tree.texts or (name in tree.folders and name.endswith(".md")))
    ]
    documents += [f"{folder}/README.md" for folder in entries if folder in tree.folders and folder != revision]
    if revision in tree.folders:
        documents += [f"{folder}/README.md" for folder in children(tree, revision) if folder in tree.folders]
    return documents


def status_documents(tree: Tree, docs: str = "docs") -> list[str]:
    """Returns each design document that exists and the first status file of each work folder."""
    documents = [name for name in design_documents(tree, docs) if name in tree.files]
    for folder in children(tree, f"{docs}/agent"):
        first = next(
            (f"{folder}/{name}" for name in K_WORK_FILES_IN_ORDER if f"{folder}/{name}" in tree.files),
            None,
        )
        if first is not None:
            documents.append(first)
    return documents


def work_folders(tree: Tree, docs: str = "docs") -> list[str]:
    """Returns every entry directly under docs/agent/, sorted."""
    return children(tree, f"{docs}/agent")


def is_under_revision(name: str) -> bool:
    """Returns whether a design document sits in the folder of topics set aside for revision."""
    return K_REVISION_FOLDER in name.split("/")


def first_content_line(text: str) -> tuple[int, str]:
    """Returns the number and text of the first non-empty line after the title."""
    for index, line in enumerate(text.splitlines()[1:], start=2):
        if line.strip():
            return index, line
    return 1, ""


def parse_status(text: str) -> Status:
    """Returns the status of a document's text, unknown when it is in none of the four forms."""
    number, line = first_content_line(text)
    match = K_STATUS_FORMS.match(line)
    if match is None:
        return Status(number, False, False, None)
    if match["open"]:
        return Status(number, True, True, None)
    closed_text = match["shipped"] or match["superseded"] or match["withdrawn"]
    try:
        closed_on = datetime.date.fromisoformat(closed_text)
    except ValueError:
        return Status(number, False, False, None)
    return Status(number, True, False, closed_on, bool(match["shipped"]))


def body_lines(text: str) -> list[tuple[int, str]]:
    """Returns the number and text of each line of a document outside fenced blocks."""
    kept: list[tuple[int, str]] = []
    fence = ""
    for number, line in enumerate(text.splitlines(), start=1):
        opener = K_FENCE.match(line.lstrip())
        if fence:
            fence = "" if opener and opener.group(1).startswith(fence) else fence
        elif opener:
            fence = opener.group(1)
        else:
            kept.append((number, line))
    return kept


def section_names(text: str) -> list[tuple[int, str]]:
    """Returns the number and name of each `##` heading of a document."""
    found = ((number, K_SECTION.match(line)) for number, line in body_lines(text))
    return [(number, match.group(1)) for number, match in found if match]


def section_body(text: str, name: str) -> list[tuple[int, str]]:
    """Returns the lines under the named `##` heading, up to the next one."""
    body: list[tuple[int, str]] = []
    inside = False
    for number, line in body_lines(text):
        heading = K_SECTION.match(line)
        if heading:
            inside = heading.group(1) == name
        elif inside:
            body.append((number, line))
    return body


def table_cells(line: str) -> list[str]:
    """Returns the trimmed cells of a table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_tree(root: Path) -> Tree:
    """Reads the docs/ tree and the module READMEs of a repository into a Tree; the only reader of the disk here."""
    folders = {"docs"}
    files: set[str] = set()
    for path in (root / "docs").rglob("*"):
        (folders if path.is_dir() else files).add(path.relative_to(root).as_posix())
    modules = root / "modules"
    if modules.is_dir():
        files.update(
            readme.relative_to(root).as_posix() for readme in modules.glob("*/*/README.md") if readme.is_file()
        )
    wanted = [name for name in sorted(files) if (name.startswith("docs/") and name.endswith(".md")) or name.endswith("/README.md")]
    texts = _read_texts(root, wanted)
    return Tree(frozenset(folders), frozenset(files), texts)


def _read_texts(root: Path, names: Iterable[str]) -> dict[str, str]:
    """Returns the text of each named file, read as UTF-8 with its byte order mark removed."""
    return {name: (root / name).read_text(encoding="utf-8-sig") for name in names}

# design_views.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Renders the generated views of the design topics: the topic map, a Designs block per module and a work block per topic."""

from __future__ import annotations

import os
import re
from pathlib import Path

from documentation.design_status import K_STATUS_FORMS, first_content_line, is_under_revision, section_names
from documentation.design_topics import (
    K_ABOUT_FIELD,
    K_IMPLEMENTS_FIELD,
    K_MODULES_FIELD,
    metadata,
    split_reference,
    split_values,
    topics,
    work_documents,
)

K_TOPICS_BLOCK = "topics"
K_DESIGNS_BLOCK = "designs"
K_WORK_BLOCK = "work"
K_NO_WORK = "No open work."
K_NO_ABOUT = "No description."
K_STATE_WORDS = {"open": "Open", "shipped": "Shipped", "superseded": "Superseded", "withdrawn": "Withdrawn"}
K_REVISION_STATE = "Revision"
K_TITLE = re.compile(r"^# (.+?)\s*$")


def markers(name: str) -> tuple[str, str]:
    """Returns the begin and end marker lines of a generated block."""
    return f"<!-- {name}:begin -->", f"<!-- {name}:end -->"


def read_block(path: Path, name: str) -> str | None:
    """Returns the lines between the markers of a block in a file, or None when the file has no such block."""
    begin, end = markers(name)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if begin not in lines or end not in lines:
        return None
    first, last = lines.index(begin), lines.index(end)
    return "\n".join(lines[first + 1 : last]) if first < last else None


def write_block(path: Path, name: str, body: str) -> bool:
    """Replaces the lines between the markers of a block; returns whether the file changed."""
    begin, end = markers(name)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.split("\n")
    if begin not in lines or end not in lines:
        return False
    first, last = lines.index(begin), lines.index(end)
    updated = "\n".join([*lines[: first + 1], *body.split("\n"), *lines[last:]])
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8", newline="")
    return True


def _state(path: Path) -> str:
    """Returns the state word a status line carries, such as Open or Shipped, or Revision for a topic set aside."""
    if is_under_revision(path):
        return K_REVISION_STATE
    _, line = first_content_line(path)
    match = K_STATUS_FORMS.match(line)
    if match is None:
        return "Unknown"
    return K_STATE_WORDS[next(name for name in K_STATE_WORDS if match.groupdict().get(name))]


def _about(path: Path) -> str:
    """Returns the About line of a topic, or a fixed word when it has none."""
    found = metadata(path).get(K_ABOUT_FIELD)
    return found[1] if found and found[1] else K_NO_ABOUT


def _link(source: Path, target: Path) -> str:
    """Returns the relative link target from a file to another, with forward slashes."""
    return os.path.relpath(target, source.parent).replace(os.sep, "/")


def _title(path: Path) -> str:
    """Returns the title of a work document, or its folder name when it has none."""
    first = path.read_text(encoding="utf-8-sig").splitlines()[:1]
    match = K_TITLE.match(first[0]) if first else None
    return match.group(1) if match else path.parent.name


def render_topic_map(docs: Path, target: Path) -> str:
    """Returns the table of every topic as it reads in the given file."""
    rows = ["| Topic | State | About |", "| --- | --- | --- |"]
    for name, path in sorted(topics(docs).items()):
        rows.append(f"| [{name}]({_link(target, path)}) | {_state(path)} | {_about(path)} |")
    return "\n".join(rows)


def render_module_designs(docs: Path, module: str, target: Path) -> str:
    """Returns the list of the topics whose Modules line names a module, as links from the given file."""
    items = []
    for name, path in sorted(topics(docs).items()):
        found = metadata(path).get(K_MODULES_FIELD)
        listed = [item.rstrip("/") for item in split_values(found[1])] if found else []
        if module in listed:
            items.append(f"- [{name}]({_link(target, path)}): {_about(path)}")
    return "\n".join(items) if items else ""


def modules_with_designs(docs: Path) -> list[str]:
    """Returns each module path that at least one topic names, in order."""
    found: set[str] = set()
    for path in topics(docs).values():
        line = metadata(path).get(K_MODULES_FIELD)
        if line:
            found.update(item.rstrip("/") for item in split_values(line[1]))
    return sorted(found)


def render_topic_work(docs: Path, topic: str, target: Path) -> str:
    """Returns the work folders that carry out a phase of a topic, with the phases each one names."""
    items = []
    for path in work_documents(docs):
        found = metadata(path).get(K_IMPLEMENTS_FIELD)
        phases = [phase for item in split_values(found[1]) for name, phase in [split_reference(item)] if name == topic] if found else []
        if phases and _state(path) == "Open":
            items.append(f"- [{path.parent.name}]({_link(target, path)}): {', '.join(phases)}")
    return "\n".join(items) if items else K_NO_WORK


def insert_block(path: Path, name: str, heading: str | None) -> bool:
    """Adds an empty block to a file, under its own heading or at the end of the Phases section; returns whether it changed."""
    begin, end = markers(name)
    text = path.read_text(encoding="utf-8-sig")
    if begin in text.splitlines():
        return False
    block = f"{begin}\n{end}\n"
    if heading is not None:
        updated = text.rstrip("\n") + f"\n\n{heading}\n\n{block}"
    else:
        names = section_names(path)
        phases = next((index for index, (_, section) in enumerate(names) if section == "Phases"), None)
        if phases is None:
            return False
        lines = text.split("\n")
        following = names[phases + 1][0] - 1 if phases + 1 < len(names) else len(lines)
        lines[following:following] = ["### Work", "", begin, end, ""]
        updated = "\n".join(lines)
    path.write_text(updated, encoding="utf-8", newline="")
    return True


def expected_blocks(root: Path) -> list[tuple[Path, str, str]]:
    """Returns each generated block a repository can hold, as its file, its name and the text it should carry."""
    docs = root / "docs"
    found: list[tuple[Path, str, str]] = []
    index = docs / "design" / "README.md"
    if index.is_file():
        found.append((index, K_TOPICS_BLOCK, render_topic_map(docs, index)))
    for module in modules_with_designs(docs):
        readme = root / module / "README.md"
        if readme.is_file():
            found.append((readme, K_DESIGNS_BLOCK, render_module_designs(docs, module, readme)))
    for name, path in topics(docs).items():
        found.append((path, K_WORK_BLOCK, render_topic_work(docs, name, path)))
    return found

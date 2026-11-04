# design_views.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Renders the generated views of the design topics from their texts: the topic map, a Designs block per module and a work block per topic."""

from __future__ import annotations

import posixpath
import re

from documentation.design_status import (
    K_STATUS_FORMS,
    Tree,
    first_content_line,
    is_under_revision,
    section_names,
)
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
K_PHASES_SECTION = "Phases"
K_WORK_HEADING = "### Work"


def markers(name: str) -> tuple[str, str]:
    """Returns the begin and end marker lines of a generated block."""
    return f"<!-- {name}:begin -->", f"<!-- {name}:end -->"


def read_block(text: str, name: str) -> str | None:
    """Returns the lines between the markers of a block in a text, or None when the text has no such block."""
    begin, end = markers(name)
    lines = text.splitlines()
    if begin not in lines or end not in lines:
        return None
    first, last = lines.index(begin), lines.index(end)
    return "\n".join(lines[first + 1 : last]) if first < last else None


def write_block(text: str, name: str, body: str) -> str:
    """Returns the text with the lines between the markers of a block replaced; the text itself when it has no such block."""
    begin, end = markers(name)
    lines = text.split("\n")
    if begin not in lines or end not in lines:
        return text
    first, last = lines.index(begin), lines.index(end)
    return "\n".join([*lines[: first + 1], *body.split("\n"), *lines[last:]])


def insert_block(text: str, name: str, heading: str | None) -> str | None:
    """Returns the text with an empty block added, under its own heading or at the end of the Phases section.

    Returns None when the block is already there, or when no heading is given and the text has no Phases section.
    """
    begin, end = markers(name)
    if begin in text.splitlines():
        return None
    block = f"{begin}\n{end}\n"
    if heading is not None:
        return text.rstrip("\n") + f"\n\n{heading}\n\n{block}"
    names = section_names(text)
    phases = next((index for index, (_, section) in enumerate(names) if section == K_PHASES_SECTION), None)
    if phases is None:
        return None
    lines = text.split("\n")
    following = names[phases + 1][0] - 1 if phases + 1 < len(names) else len(lines)
    lines[following:following] = [K_WORK_HEADING, "", begin, end, ""]
    return "\n".join(lines)


def state_of(name: str, text: str) -> str:
    """Returns the state word a status line carries, such as Open or Shipped, or Revision for a topic set aside."""
    if is_under_revision(name):
        return K_REVISION_STATE
    _, line = first_content_line(text)
    match = K_STATUS_FORMS.match(line)
    if match is None:
        return "Unknown"
    return K_STATE_WORDS[next(word for word in K_STATE_WORDS if match.groupdict().get(word))]


def _about(text: str) -> str:
    """Returns the About line of a topic, or a fixed word when it has none."""
    found = metadata(text).get(K_ABOUT_FIELD)
    return found[1] if found and found[1] else K_NO_ABOUT


def _link(source: str, target: str) -> str:
    """Returns the relative link target from a file to another, with forward slashes."""
    return posixpath.relpath(target, posixpath.dirname(source))


def render_topic_map(tree: Tree, docs: str, target: str) -> str:
    """Returns the table of every topic as it reads in the given file."""
    rows = ["| Topic | State | About |", "| --- | --- | --- |"]
    for name, path in sorted(topics(tree, docs).items()):
        rows.append(f"| [{name}]({_link(target, path)}) | {state_of(path, tree.texts[path])} | {_about(tree.texts[path])} |")
    return "\n".join(rows)


def render_module_designs(tree: Tree, docs: str, module: str, target: str) -> str:
    """Returns the list of the topics whose Modules line names a module, as links from the given file."""
    items = []
    for name, path in sorted(topics(tree, docs).items()):
        found = metadata(tree.texts[path]).get(K_MODULES_FIELD)
        listed = [item.rstrip("/") for item in split_values(found[1])] if found else []
        if module in listed:
            items.append(f"- [{name}]({_link(target, path)}): {_about(tree.texts[path])}")
    return "\n".join(items) if items else ""


def modules_with_designs(tree: Tree, docs: str = "docs") -> list[str]:
    """Returns each module path that at least one topic names, in order."""
    found: set[str] = set()
    for path in topics(tree, docs).values():
        line = metadata(tree.texts[path]).get(K_MODULES_FIELD)
        if line:
            found.update(item.rstrip("/") for item in split_values(line[1]))
    return sorted(found)


def render_topic_work(tree: Tree, docs: str, topic: str, target: str) -> str:
    """Returns the work folders that carry out a phase of a topic, with the phases each one names."""
    items = []
    for path in work_documents(tree, docs):
        text = tree.texts[path]
        found = metadata(text).get(K_IMPLEMENTS_FIELD)
        phases = [phase for item in split_values(found[1]) for name, phase in [split_reference(item)] if name == topic] if found else []
        if phases and state_of(path, text) == "Open":
            items.append(f"- [{posixpath.basename(posixpath.dirname(path))}]({_link(target, path)}): {', '.join(phases)}")
    return "\n".join(items) if items else K_NO_WORK


def expected_blocks(tree: Tree, docs: str = "docs") -> list[tuple[str, str, str]]:
    """Returns each generated block a repository can hold, as its file, its name and the text it should carry."""
    found: list[tuple[str, str, str]] = []
    index = f"{docs}/design/README.md"
    if index in tree.files:
        found.append((index, K_TOPICS_BLOCK, render_topic_map(tree, docs, index)))
    for module in modules_with_designs(tree, docs):
        readme = f"{module}/README.md"
        if readme in tree.files:
            found.append((readme, K_DESIGNS_BLOCK, render_module_designs(tree, docs, module, readme)))
    for name, path in topics(tree, docs).items():
        found.append((path, K_WORK_BLOCK, render_topic_work(tree, docs, name, path)))
    return found

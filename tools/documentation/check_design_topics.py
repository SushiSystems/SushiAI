# check_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that each design topic names real modules, that work folders name real topics and phases, and the topic limits.

Usage: python tools/documentation/check_design_topics.py [repository root] [--report] [--rule NAME]

`settings.py` may set K_DESIGN_TOPICS (require the Modules line and a glossary entry per topic) and
K_FOLDER_CEILING (entries allowed in docs/design and docs/agent; 0 turns the rule off).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from documentation.citation_files import archive_citations  # noqa: E402
from documentation.design_topics import (  # noqa: E402
    K_DESIGNS_FIELD,
    K_IMPLEMENTS_FIELD,
    K_MODULES_FIELD,
    loose_documents,
    metadata,
    phase_ids,
    split_reference,
    split_values,
    topics,
    work_documents,
)
from documentation.design_views import K_TOPICS_BLOCK, expected_blocks, read_block  # noqa: E402

K_GLOSSARY = Path("docs/reference/GLOSSARY.md")
K_TERM_LINE = re.compile(r"^(\|\s*`?|\*\*|#+\s)")
K_NAME_SEPARATOR = re.compile(r"[_\s-]+")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of one repository and the two settings that decide how strict the topic rules are."""

    root: Path
    is_required: bool = False
    ceiling: int = 0
    is_archive_untracked: bool = False

    @property
    def docs(self) -> Path:
        """Returns the docs/ folder of the repository."""
        return self.root / "docs"


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree, with the topic settings of this repository."""
    required = bool(getattr(settings, "K_DESIGN_TOPICS", False))
    ceiling = int(getattr(settings, "K_FOLDER_CEILING", 0))
    untracked = bool(getattr(settings, "K_ARCHIVE_UNTRACKED", False))
    return [Repository(path, required, ceiling, untracked) for path in paths if (path / "docs").is_dir()]


def _normal(name: str) -> str:
    """Returns a name in lower case with its separators as single spaces."""
    return K_NAME_SEPARATOR.sub(" ", name).strip().lower()


def rule_topic_folder(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a design document that sits loose in docs/design, when the repository asks for topics."""
    if repository.is_required:
        for path in loose_documents(repository.root):
            yield Issue(path, 1, f"a topic is a folder: move it to {path.stem}/README.md with fold_design_topics.py")


def rule_modules_exist(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each module path of a Modules line that is not in the repository."""
    for path in topics(repository.docs).values():
        found = metadata(path).get(K_MODULES_FIELD)
        if found is None:
            continue
        number, value = found
        for item in split_values(value):
            if not (repository.root / item).exists():
                yield Issue(path, number, f"the module '{item}' is not in the repository")


def rule_modules_required(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a topic without a Modules line, when the repository asks for one on every topic."""
    if not repository.is_required:
        return
    for path in topics(repository.docs).values():
        if K_MODULES_FIELD not in metadata(path):
            yield Issue(path, 1, f"a topic names the modules it governs on a '**{K_MODULES_FIELD}:**' line, or says none")


def rule_implements_resolve(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for an Implements item whose topic or phase does not exist."""
    known = topics(repository.docs)
    for path in work_documents(repository.docs):
        found = metadata(path).get(K_IMPLEMENTS_FIELD)
        if found is None:
            continue
        number, value = found
        for item in split_values(value):
            name, phase = split_reference(item)
            if name not in known:
                yield Issue(path, number, f"'{name}' is not a design topic")
            elif not phase:
                yield Issue(path, number, f"'{item}' names a topic and no phase; write TOPIC#PHASE")
            elif phase not in phase_ids(known[name]):
                yield Issue(path, number, f"'{name}' has no phase '{phase}'")


def rule_designs_resolve(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a Designs item that is not a design topic."""
    known = topics(repository.docs)
    for path in work_documents(repository.docs):
        found = metadata(path).get(K_DESIGNS_FIELD)
        if found is None:
            continue
        number, value = found
        for item in split_values(value):
            if item not in known:
                yield Issue(path, number, f"'{item}' is not a design topic")


def rule_glossary_entry(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a topic that the glossary does not name as a term, when the repository asks for it."""
    if not repository.is_required:
        return
    glossary = repository.root / K_GLOSSARY
    terms = ""
    if glossary.is_file():
        lines = glossary.read_text(encoding="utf-8-sig").splitlines()
        terms = "\n".join(_normal(line) for line in lines if K_TERM_LINE.match(line))
    for name, path in topics(repository.docs).items():
        if not re.search(rf"(?<![a-z0-9]){re.escape(_normal(name))}(?![a-z0-9])", terms):
            yield Issue(path, 1, f"the glossary has no entry for the topic '{name}'")


def rule_folder_ceiling(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a docs/design or docs/agent folder that holds more entries than the ceiling."""
    if repository.ceiling <= 0:
        return
    for name in ("design", "agent"):
        folder = repository.docs / name
        if folder.is_dir():
            count = sum(1 for _ in folder.iterdir())
            if count > repository.ceiling:
                yield Issue(folder, 1, f"{count} entries; the ceiling is {repository.ceiling}")


def rule_archive_not_cited(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each line that cites the archive, once the repository no longer tracks it."""
    if not repository.is_archive_untracked:
        return
    for path, number in archive_citations(repository.root):
        yield Issue(path, number, "the archive is not tracked; cite the topic that holds the decision")


def rule_views_current(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a generated block whose text is not what the metadata says it should be."""
    for path, name, body in expected_blocks(repository.root):
        current = read_block(path, name)
        if current is not None and current != body:
            yield Issue(path, 1, f"the generated block '{name}' is stale; run generate_design_views.py")


def rule_views_present(repository: Repository) -> Iterator[Issue]:
    """Yields an issue when the repository asks for topics and its design index lacks the topic map block."""
    index = repository.docs / "design" / "README.md"
    if repository.is_required and index.is_file() and read_block(index, K_TOPICS_BLOCK) is None:
        yield Issue(index, 1, "the topic map is generated; run generate_design_views.py --init")


K_CHECKER = Checker(
    description="Checks the topic metadata of design documents, the phases work folders name and the folder limits.",
    collect=collect,
    rules={
        "rule_topic_folder": rule_topic_folder,
        "rule_modules_exist": rule_modules_exist,
        "rule_modules_required": rule_modules_required,
        "rule_implements_resolve": rule_implements_resolve,
        "rule_designs_resolve": rule_designs_resolve,
        "rule_glossary_entry": rule_glossary_entry,
        "rule_folder_ceiling": rule_folder_ceiling,
        "rule_archive_not_cited": rule_archive_not_cited,
        "rule_views_current": rule_views_current,
        "rule_views_present": rule_views_present,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

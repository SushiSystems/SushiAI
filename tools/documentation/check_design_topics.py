# check_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks the topic metadata of design documents, each module owned by one topic, the generated views and the limits.

Usage: python tools/documentation/check_design_topics.py [repository root] [--report] [--rule NAME]

Topic, module and work folder rules run the tooling package's `design check`. settings.py may set
K_DESIGN_TOPICS (a glossary entry per topic and the topic map) and K_FOLDER_CEILING (0 is off).
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
from documentation.design_status import Tree, children, read_tree  # noqa: E402
from documentation.design_topics import topics  # noqa: E402
from documentation.design_views import K_TOPICS_BLOCK, expected_blocks, read_block  # noqa: E402
from documentation.tooling_bridge import design_rule  # noqa: E402

K_GLOSSARY = "docs/reference/GLOSSARY.md"
K_DESIGN_INDEX = "docs/design/README.md"
K_TERM_LINE = re.compile(r"^(\|\s*`?|\*\*|#+\s)")
K_NAME_SEPARATOR = re.compile(r"[_\s-]+")
K_DELEGATED = ("topic_folder", "modules_required", "modules_exist", "module_owner", "work_references")
K_CEILED_FOLDERS = ("docs/design", "docs/agent")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the values one repository is checked from: its root, its tree, its glossary and its topic settings.

    The shell reads the tree, the glossary and the archive citations in collect(); the rules take them as given.
    """

    root: Path
    tree: Tree
    glossary: str | None = None
    is_required: bool = False
    ceiling: int = 0
    is_archive_untracked: bool = False
    archive_citations: tuple[tuple[str, int], ...] = ()


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree, read into values with the topic settings of this repository."""
    required = bool(getattr(settings, "K_DESIGN_TOPICS", False))
    ceiling = int(getattr(settings, "K_FOLDER_CEILING", 0))
    untracked = bool(getattr(settings, "K_ARCHIVE_UNTRACKED", False))
    return [
        Repository(
            root=path,
            tree=read_tree(path),
            glossary=_read_glossary(path),
            is_required=required,
            ceiling=ceiling,
            is_archive_untracked=untracked,
            archive_citations=_read_archive_citations(path) if untracked else (),
        )
        for path in paths if (path / "docs").is_dir()
    ]


def _normal(name: str) -> str:
    """Returns a name in lower case with its separators as single spaces."""
    return K_NAME_SEPARATOR.sub(" ", name).strip().lower()


def rule_glossary_entry(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a topic that the glossary does not name as a term, when the repository asks for it."""
    if not repository.is_required:
        return
    terms = ""
    if repository.glossary is not None:
        terms = "\n".join(_normal(line) for line in repository.glossary.splitlines() if K_TERM_LINE.match(line))
    for name, path in topics(repository.tree).items():
        if not re.search(rf"(?<![a-z0-9]){re.escape(_normal(name))}(?![a-z0-9])", terms):
            yield Issue(repository.root / path, 1, f"the glossary has no entry for the topic '{name}'")


def rule_folder_ceiling(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a docs/design or docs/agent folder that holds more entries than the ceiling."""
    if repository.ceiling <= 0:
        return
    for folder in K_CEILED_FOLDERS:
        if folder in repository.tree.folders:
            count = len(children(repository.tree, folder))
            if count > repository.ceiling:
                yield Issue(repository.root / folder, 1, f"{count} entries; the ceiling is {repository.ceiling}")


def rule_archive_not_cited(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each line that cites the archive, once the repository no longer tracks it."""
    if not repository.is_archive_untracked:
        return
    for name, number in repository.archive_citations:
        yield Issue(repository.root / name, number, "the archive is not tracked; cite the topic that holds the decision")


def rule_views_current(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a generated block whose text is not what the metadata says it should be."""
    for name, block, body in expected_blocks(repository.tree):
        current = read_block(repository.tree.texts[name], block)
        if current is not None and current != body:
            yield Issue(repository.root / name, 1, f"the generated block '{block}' is stale; run generate_design_views.py")


def rule_views_present(repository: Repository) -> Iterator[Issue]:
    """Yields an issue when the repository asks for topics and its design index lacks the topic map block."""
    if not repository.is_required or K_DESIGN_INDEX not in repository.tree.files:
        return
    if read_block(repository.tree.texts[K_DESIGN_INDEX], K_TOPICS_BLOCK) is None:
        yield Issue(repository.root / K_DESIGN_INDEX, 1, "the topic map is generated; run generate_design_views.py --init")


def _read_glossary(root: Path) -> str | None:
    """Returns the text of the glossary, or None when the repository has none."""
    glossary = root / K_GLOSSARY
    return glossary.read_text(encoding="utf-8-sig") if glossary.is_file() else None


def _read_archive_citations(root: Path) -> tuple[tuple[str, int], ...]:
    """Returns each line that cites the archive, as a root-relative name and a line number."""
    return tuple((path.relative_to(root).as_posix(), number) for path, number in archive_citations(root))


K_CHECKER = Checker(
    description="Checks the topic metadata of design documents, the phases work folders name and the folder limits.",
    collect=collect,
    rules={
        **{f"rule_{name}": design_rule(name, name == K_DELEGATED[0]) for name in K_DELEGATED},
        "rule_glossary_entry": rule_glossary_entry,
        "rule_folder_ceiling": rule_folder_ceiling,
        "rule_archive_not_cited": rule_archive_not_cited,
        "rule_views_current": rule_views_current,
        "rule_views_present": rule_views_present,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# check_docs_facts.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks the facts a manual keeps in one home, and that every quote of one holds its text exactly.

Usage: python tools/documentation/check_docs_facts.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402
from documentation import facts  # noqa: E402
from documentation.facts import Page  # noqa: E402

K_SKIPPED = frozenset({".git", "third_party", "node_modules", "build"})
K_APPLY = "python tools/documentation/facts_cli.py apply"


@final
@dataclass(frozen=True, slots=True)
class Manual:
    """Holds the root of one repository, the text of its pages and the facts those pages carry."""

    root: Path
    pages: tuple[Page, ...]
    sources: tuple[SourceFile, ...]

    def homes(self) -> dict[str, tuple[Page, facts.Block]]:
        """Returns the first home of each fact id, with the page that holds it."""
        found: dict[str, tuple[Page, facts.Block]] = {}
        for page in self.pages:
            for block in page.facts:
                found.setdefault(block.id, (page, block))
        return found


def build_manual(root: Path, files: Iterable[SourceFile]) -> Manual:
    """Returns the manual of a repository from its markdown files, each given with its text."""
    sources = tuple(files)
    pages = tuple(
        facts.parse(source.path.relative_to(root).as_posix(), source.text) for source in sources
    )
    return Manual(root, pages, sources)


def is_checked_page(root: Path, path: Path) -> bool:
    """Returns whether a page lies outside every folder the manual skips."""
    return not K_SKIPPED.intersection(path.relative_to(root).parts)


def rule_markers(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each marker that is unpaired, nested or misnamed."""
    for page in manual.pages:
        for line, message in page.faults:
            yield Issue(manual.root / page.path, line, message)


def rule_unique_ids(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each fact id that two homes claim."""
    seen: dict[str, str] = {}
    for page in manual.pages:
        for block in page.facts:
            if block.id in seen:
                yield Issue(
                    manual.root / page.path, block.first,
                    f"fact {block.id} already has its home in {seen[block.id]}",
                )
            else:
                seen[block.id] = page.path


def rule_cited_fact_exists(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each quote or follow mark whose fact has no home."""
    homes = manual.homes()
    for page in manual.pages:
        for block in page.quotes:
            if block.id not in homes:
                yield Issue(manual.root / page.path, block.first,
                            f"quote of {block.id}, which has no home")
        for fact_id, line in page.follows:
            if fact_id not in homes:
                yield Issue(manual.root / page.path, line, f"follows {fact_id}, which has no home")


def rule_quote_current(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each quote that does not hold the text of its fact exactly."""
    homes = manual.homes()
    for page in manual.pages:
        for block in page.quotes:
            home = homes.get(block.id)
            if home is not None and home[1].body != block.body:
                yield Issue(
                    manual.root / page.path, block.first,
                    f"quote of {block.id} differs from its home in {home[0].path}; run {K_APPLY}",
                )


def _read_pages(root: Path) -> tuple[SourceFile, ...]:
    """Reads the pages under a root that git does not ignore and no skipped folder holds."""
    candidates = [path for path in sorted(root.rglob("*.md")) if is_checked_page(root, path)]
    return read_files(drop_ignored(root, candidates), lambda _: True)


def load(root: Path) -> Manual:
    """Returns the manual of a repository, read from its disk."""
    return build_manual(root, _read_pages(root))


def collect(paths: list[Path]) -> Iterable[Manual]:
    """Returns the manual of each given repository root."""
    return [load(path) for path in paths if path.is_dir()]


K_CHECKER = Checker(
    description="Check the facts a manual keeps in one home and the quotes of them.",
    collect=collect,
    rules={
        "rule_markers": rule_markers,
        "rule_unique_ids": rule_unique_ids,
        "rule_cited_fact_exists": rule_cited_fact_exists,
        "rule_quote_current": rule_quote_current,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

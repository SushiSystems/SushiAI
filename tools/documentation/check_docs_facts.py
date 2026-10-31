# check_docs_facts.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the facts a manual keeps in one home, and that every quote of one holds its text exactly.

Usage: python tools/documentation/check_docs_facts.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
from documentation import facts  # noqa: E402
from documentation.facts import Page  # noqa: E402

K_SKIPPED = frozenset({".git", "third_party", "node_modules", "build"})
K_APPLY = "python tools/documentation/facts_cli.py apply"


@dataclass(frozen=True, slots=True)
class Manual:
    """Holds the root of one repository and the facts its markdown pages carry."""

    root: Path
    pages: tuple[Page, ...]

    def homes(self) -> dict[str, tuple[Page, facts.Block]]:
        """Returns the first home of each fact id, with the page that holds it."""
        found: dict[str, tuple[Page, facts.Block]] = {}
        for page in self.pages:
            for block in page.facts:
                found.setdefault(block.id, (page, block))
        return found


def load(root: Path) -> Manual:
    """Returns the pages of a repository that git does not ignore, parsed for facts."""
    candidates = [path for path in sorted(root.rglob("*.md")) if not K_SKIPPED.intersection(path.relative_to(root).parts)]
    pages = tuple(facts.parse(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace"))
                  for path in drop_ignored(root, candidates))
    return Manual(root, pages)


def collect(paths: list[Path]) -> Iterable[Manual]:
    """Returns the manual of each given repository root."""
    return [load(path) for path in paths if path.is_dir()]


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
                yield Issue(manual.root / page.path, block.first, f"fact {block.id} already has its home in {seen[block.id]}")
            else:
                seen[block.id] = page.path


def rule_cited_fact_exists(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each quote or follow mark whose fact has no home."""
    homes = manual.homes()
    for page in manual.pages:
        for block in page.quotes:
            if block.id not in homes:
                yield Issue(manual.root / page.path, block.first, f"quote of {block.id}, which has no home")
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
                yield Issue(manual.root / page.path, block.first,
                            f"quote of {block.id} differs from its home in {home[0].path}; run {K_APPLY}")


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

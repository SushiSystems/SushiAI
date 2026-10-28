# check_backlog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks docs/backlog/ against the backlog format of the documentation skill, through sushicore.

Usage: python tools/documentation/check_backlog.py [repository root] [--report] [--rule NAME]

The rules are those of `backlog check` in sushicore.backlog, the tool that writes the store.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from documentation.sushicore_bridge import load, missing  # noqa: E402

K_BACKLOG = Path("docs/backlog")
K_RETIRED = (Path("docs/design/REMAINING_WORK.md"), Path("docs/reference/KNOWN_ISSUES.md"))


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of one repository whose backlog is checked."""

    root: Path

    def has_backlog(self) -> bool:
        """Returns whether the repository holds a store or a retired file to check."""
        return (self.root / K_BACKLOG).is_dir() or any((self.root / path).is_file() for path in K_RETIRED)


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree."""
    return [Repository(path) for path in paths if (path / "docs").is_dir()]


def rule_backlog(repository: Repository) -> Iterator[Issue]:
    """Yields each fault `backlog check` finds, or one issue when sushicore cannot be loaded."""
    if not repository.has_backlog():
        return
    backlog = load("sushicore.backlog", repository.root)
    if backlog is None:
        yield missing(repository.root / "docs", "sushicore.backlog")
        return
    design = load("sushicore.design", repository.root)
    store = backlog.BacklogStore(repository.root)
    open_designs = design.DesignStore(repository.root).open_documents if design else (lambda: [])
    for fault in backlog.check_backlog(store, open_designs):
        yield Issue(fault.path, fault.line, fault.message)


K_CHECKER = Checker(
    description="Checks the backlog and the known issues in docs/backlog/ through sushicore.backlog.",
    collect=collect,
    rules={"rule_backlog": rule_backlog},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

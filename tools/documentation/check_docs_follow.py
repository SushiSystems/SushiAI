# check_docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that each commit after K_DOCS_FOLLOW_SINCE updates the pages that describe what it changes.

Usage: python tools/documentation/check_docs_follow.py [repository root] [--report] [--rule NAME]
A page that falls behind needs a `Docs: <page> unchanged (<reason>)` line in the message.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from commits.history import Commit, commits_since, git  # noqa: E402
from documentation import docs_follow  # noqa: E402


def collect(paths: list[Path]) -> Iterable[Commit]:
    """Returns each commit after the base below each given repository root, oldest first."""
    return [commit for root in paths if root.is_dir() and docs_follow.K_SINCE
            for commit in commits_since(root, docs_follow.K_SINCE)]


def rule_pages_follow(commit: Commit) -> Iterator[Issue]:
    """Yields an issue for each page that describes what a commit changes and the commit leaves alone."""
    listing = git(commit.root, "diff-tree", "-r", "--name-status", "-M", "--no-commit-id", commit.parent, commit.sha)
    changes = docs_follow.parse_changes(listing or "")
    for problem in docs_follow.follow_problems(commit.root, changes, commit.message, commit.parent, commit.sha):
        yield Issue(commit.root / f"commit-{commit.sha[:9]}", 1, problem)


K_CHECKER = Checker(
    description="Check that each commit updates the pages that describe what it changes.",
    collect=collect,
    rules={"rule_pages_follow": rule_pages_follow},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

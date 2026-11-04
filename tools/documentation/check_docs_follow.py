# check_docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that each commit after K_DOCS_FOLLOW_SINCE updates the pages that describe its changes.

Usage: python tools/documentation/check_docs_follow.py [repository root] [--report] [--rule NAME]
A page that falls behind needs a `Docs: <page> unchanged (<reason>)` line in the message.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from commits.history import Commit, commits_since, git  # noqa: E402
from documentation import docs_follow  # noqa: E402
from documentation.docs_follow import Follow  # noqa: E402


@dataclass(frozen=True, slots=True)
class CommitFollow:
    """Holds one commit to check as values: the root it lies in, its hash and what it presents."""

    root: Path
    sha: str
    follow: Follow


def rule_pages_follow(subject: CommitFollow) -> Iterator[Issue]:
    """Yields an issue for each page that describes a changed path of the commit and is left alone."""
    for problem in docs_follow.page_problems(subject.follow):
        yield Issue(subject.root / f"commit-{subject.sha[:9]}", 1, problem)


def collect(paths: list[Path]) -> list[CommitFollow]:
    """Returns each commit after the base below each given root, oldest first, read into values."""
    base = docs_follow.K_SINCE
    return [
        CommitFollow(
            commit.root,
            commit.sha,
            docs_follow.read_follow(
                commit.root, docs_follow.parse_changes(_read_listing(commit)), commit.message,
                commit.parent, commit.sha,
            ),
        )
        for root in paths if root.is_dir() and base
        for commit in commits_since(root, base)
    ]


def _read_listing(commit: Commit) -> str:
    """Returns the status listing of the paths a commit changes, from git; empty when git fails."""
    return git(commit.root, "diff-tree", "-r", "--name-status", "-M", "--no-commit-id",
               commit.parent, commit.sha) or ""


K_CHECKER = Checker(
    description="Check that each commit updates the pages that describe what it changes.",
    collect=collect,
    rules={"rule_pages_follow": rule_pages_follow},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

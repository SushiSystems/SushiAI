# check_backlog_trailer.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that each commit after K_TRAILER_SINCE ends with a `Backlog:` line, from `commits`.

Usage: python tools/commits/check_backlog_trailer.py [repository root] [--report] [--rule NAME]
The line is `Backlog: <ID>[, <ID>]` or `Backlog: none (<reason>)`; an ID is open in the parent or the commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402

K_SINCE: str = getattr(settings, "K_TRAILER_SINCE", "")
K_ID_PATTERN: str = getattr(settings, "K_BACKLOG_ID_PATTERN", r"[A-Z]{2}-[0-9]+")
K_BACKLOG_FOLDER = "docs/backlog"
K_TRAILER = re.compile(r"^Backlog:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
K_NONE = re.compile(r"^none \(\s*(\S.*?)\s*\)$")


@dataclass(frozen=True, slots=True)
class Commit:
    """Holds one commit to check: where it is, its hash, its parent and its message."""

    root: Path
    sha: str
    parent: str
    message: str


def _git(root: Path, *arguments: str) -> str | None:
    """Returns the output of a git command, or None when it fails."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _ids_in(text: str) -> set[str]:
    """Returns the IDs that open the rows of the tables in one backlog file."""
    row = re.compile(rf"^\|\s*({K_ID_PATTERN})\s*\|", re.MULTILINE)
    return set(row.findall(text))


def open_ids(root: Path, revision: str | None) -> frozenset[str]:
    """Returns the IDs of the entries in the backlog at a revision, or on disk when revision is None."""
    ids: set[str] = set()
    if revision is None:
        folder = root / K_BACKLOG_FOLDER
        for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
            ids |= _ids_in(path.read_text(encoding="utf-8", errors="replace"))
        return frozenset(ids)
    listing = _git(root, "ls-tree", "-r", "--name-only", revision, "--", K_BACKLOG_FOLDER)
    for name in (listing or "").splitlines():
        if name.endswith(".md"):
            ids |= _ids_in(_git(root, "show", f"{revision}:{name}") or "")
    return frozenset(ids)


def trailer_value(message: str) -> str | None:
    """Returns the text after the last `Backlog:` of a message, or None when it has none."""
    found = K_TRAILER.findall(message)
    return found[-1] if found else None


def trailer_problems(message: str, known: frozenset[str]) -> list[str]:
    """Returns what is wrong with the `Backlog:` line of a message; empty when it is right."""
    value = trailer_value(message)
    if value is None:
        return ["the message has no `Backlog:` line; write `Backlog: <ID>` or `Backlog: none (<reason>)`"]
    if value.startswith("none"):
        return [] if K_NONE.match(value) else ["`Backlog: none` needs its reason in brackets: `none (<reason>)`"]
    ids = [item.strip() for item in value.split(",")]
    problems = [f"`Backlog:` names {item!r}, which is not a backlog ID" for item in ids
                if not re.fullmatch(K_ID_PATTERN, item)]
    problems += [f"`Backlog:` names {item}, which is not an entry open at the parent or in this commit"
                 for item in ids if re.fullmatch(K_ID_PATTERN, item) and item not in known]
    return problems


def collect(paths: list[Path]) -> Iterable[Commit]:
    """Returns each commit after K_TRAILER_SINCE below each given repository root, oldest first."""
    commits: list[Commit] = []
    for root in (path for path in paths if path.is_dir() and K_SINCE):
        listing = _git(root, "rev-list", "--no-merges", "--reverse", f"{K_SINCE}..HEAD")
        if listing is None:
            raise SystemExit(f"cannot read the history after {K_SINCE}; fetch the full history")
        for sha in listing.split():
            parent = _git(root, "rev-parse", f"{sha}^")
            message = _git(root, "log", "-1", "--format=%B", sha)
            if parent is not None and message is not None:
                commits.append(Commit(root, sha, parent.strip(), message))
    return commits


def _issue(commit: Commit, text: str) -> Issue:
    """Returns an issue placed on the commit, named by its short hash."""
    return Issue(commit.root / f"commit-{commit.sha[:9]}", 1, text)


def rule_trailer(commit: Commit) -> Iterator[Issue]:
    """Yields an issue for each fault of the `Backlog:` line of a commit."""
    known = open_ids(commit.root, commit.parent) | open_ids(commit.root, commit.sha)
    for problem in trailer_problems(commit.message, known):
        yield _issue(commit, problem)


K_CHECKER = Checker(
    description="Check that each commit names the backlog entry it serves.",
    collect=collect,
    rules={"rule_trailer": rule_trailer},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

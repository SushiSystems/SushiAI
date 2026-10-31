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

import functools
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from commits.history import Commit, commits_since, git  # noqa: E402

K_SINCE: str = getattr(settings, "K_TRAILER_SINCE", "")
K_ID_PATTERN: str = getattr(settings, "K_BACKLOG_ID_PATTERN", r"[A-Z]{2}-[0-9]+")
K_BACKLOG_FOLDER = "docs/backlog"
K_TRAILER = re.compile(r"^Backlog:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
K_NONE = re.compile(r"^none \(\s*(\S.*?)\s*\)$")


def _ids_in(text: str) -> set[str]:
    """Returns the IDs that open the rows of the tables in one backlog file."""
    row = re.compile(rf"^\|\s*({K_ID_PATTERN})\s*\|", re.MULTILINE)
    return set(row.findall(text))


@functools.lru_cache(maxsize=None)
def _ids_of_blob(root: Path, blob: str) -> frozenset[str]:
    """Returns the IDs in one backlog file, named by its object hash, which never changes its content."""
    return frozenset(_ids_in(git(root, "cat-file", "blob", blob) or ""))


@functools.lru_cache(maxsize=None)
def _ids_of_tree(root: Path, tree: str) -> frozenset[str]:
    """Returns the IDs in the backlog folder named by its tree hash; most commits leave the folder alone."""
    ids: set[str] = set()
    for line in (git(root, "ls-tree", "-r", tree) or "").splitlines():
        info, _, name = line.partition("\t")
        if name.endswith(".md"):
            ids |= _ids_of_blob(root, info.split()[2])
    return frozenset(ids)


@functools.lru_cache(maxsize=None)
def _backlog_tree(root: Path, revision: str) -> str:
    """Returns the tree hash of the backlog folder at a revision, or an empty string when it has none."""
    return (git(root, "rev-parse", f"{revision}:{K_BACKLOG_FOLDER}") or "").strip()


def open_ids(root: Path, revision: str | None) -> frozenset[str]:
    """Returns the IDs of the entries in the backlog at a revision, or on disk when revision is None.

    A revision is read through the hashes of the backlog folder and its files, so a run over many
    commits reads each version of a file once.
    """
    if revision is None:
        ids: set[str] = set()
        folder = root / K_BACKLOG_FOLDER
        for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
            ids |= _ids_in(path.read_text(encoding="utf-8", errors="replace"))
        return frozenset(ids)
    tree = _backlog_tree(root, revision)
    return _ids_of_tree(root, tree) if tree else frozenset()


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
    return [commit for root in paths if root.is_dir() and K_SINCE for commit in commits_since(root, K_SINCE)]


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

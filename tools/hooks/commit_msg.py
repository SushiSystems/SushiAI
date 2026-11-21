# commit_msg.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Refuses a commit message that does not name the backlog entry it serves.

Usage: python tools/hooks/commit_msg.py MESSAGE_FILE

Git runs it through the commit-msg hook: the `Backlog:` line, the pages and the test, each when its setting is on.
An amend is judged against the parent of the commit it replaces, since HEAD is that commit.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import check_backlog_trailer, tests_trailer  # noqa: E402
from commits.history import git  # noqa: E402
from documentation import docs_follow  # noqa: E402
from documentation.docs_follow import Follow  # noqa: E402

K_EXIT_CLEAN = 0
K_EXIT_REFUSED = 1
K_COMMENT = "#"
K_MERGE = "Merge "
K_PARENT = "HEAD"
K_AMENDED_PARENT = "HEAD^"
K_AMEND_MIN_AGE_S = 2


def strip_comments(text: str) -> str:
    """Returns a message file's text without git's comment lines."""
    return "\n".join(line for line in text.splitlines() if not line.startswith(K_COMMENT))


def is_merge(message: str) -> bool:
    """Returns whether a message is a merge commit's, which the hook leaves alone."""
    return message.startswith(K_MERGE)


def is_amend(root: Path, environ: Mapping[str, str], now: float | None = None) -> bool:
    """Returns whether the commit being made replaces HEAD, which git marks by giving it HEAD's author date.

    A new commit made in the same second as HEAD carries the same date, so a HEAD younger than
    `K_AMEND_MIN_AGE_S` seconds is taken for a parent and judged the stricter way.
    """
    given = environ.get("GIT_AUTHOR_DATE", "").lstrip("@")
    head = (git(root, "log", "-1", "--format=%ad", "--date=raw", K_PARENT) or "").strip()
    if not given or given != head:
        return False
    return (time.time() if now is None else now) - int(head.split()[0]) > K_AMEND_MIN_AGE_S


def commit_problems(
    message: str, known: frozenset[str] | None, id_pattern: str, follow: Follow | None,
    siblings: check_backlog_trailer.SiblingLookup | None = None,
) -> list[str]:
    """Returns what is wrong with a commit being made; empty when the hook accepts it.

    *known* is the IDs open at HEAD or on disk, and None when the `Backlog:` line is not required.
    *follow* is the staged change as the page rules read it, and None when pages need not follow.
    *siblings* looks up an entry cited as `<repository> <ID>`; without it such a citation is taken on trust.
    """
    problems: list[str] = []
    if known is not None:
        problems += check_backlog_trailer.trailer_problems(message, known, id_pattern, siblings)
    if follow is not None:
        problems += docs_follow.page_problems(follow)
    return problems


def refusal_lines(problems: list[str]) -> list[str]:
    """Returns the line printed for each problem."""
    return [f"commit refused: {problem}" for problem in problems]


def _read_known(root: Path, parent: str) -> frozenset[str] | None:
    """Returns the IDs open at *parent* or on disk, or None when no base is set for the `Backlog:` line."""
    if not check_backlog_trailer.K_SINCE:
        return None
    return check_backlog_trailer.open_ids(root, parent) | check_backlog_trailer.open_ids(root, None)


def _read_follow(root: Path, message: str, parent: str) -> Follow | None:
    """Returns the staged change against *parent* as the page rules read it, or None when no base is set for the pages."""
    if not docs_follow.K_SINCE:
        return None
    listing = git(root, "diff", "--cached", "--name-status", "-M", parent) or ""
    return docs_follow.read_follow(root, docs_follow.parse_changes(listing), message, parent, None)


def main(argv: list[str] | None = None) -> int:
    """Returns 1 and prints each fault when the message in the given file breaks the rule."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: commit_msg.py MESSAGE_FILE", file=sys.stderr)
        return K_EXIT_REFUSED
    message = strip_comments(Path(arguments[0]).read_text(encoding="utf-8", errors="replace"))
    if is_merge(message):
        return K_EXIT_CLEAN
    root = Path.cwd()
    parent = K_AMENDED_PARENT if is_amend(root, os.environ) else K_PARENT
    problems = commit_problems(
        message, _read_known(root, parent), check_backlog_trailer.K_ID_PATTERN,
        _read_follow(root, message, parent), check_backlog_trailer.sibling_lookup(root))
    if tests_trailer.K_TESTS_LINE:
        staged = (git(root, "diff", "--cached", "--name-only") or "").splitlines()
        problems += tests_trailer.tests_problems(message, staged)
    for line in refusal_lines(problems):
        print(line, file=sys.stderr)
    return K_EXIT_REFUSED if problems else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

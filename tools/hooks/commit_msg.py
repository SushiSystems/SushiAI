# commit_msg.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Refuses a commit message that does not name the backlog entry it serves.

Usage: python tools/hooks/commit_msg.py MESSAGE_FILE

Git runs it through the commit-msg hook. It checks the `Backlog:` line when K_TRAILER_SINCE is set and
the pages that follow the change when K_DOCS_FOLLOW_SINCE is set; a merge commit is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import check_backlog_trailer  # noqa: E402
from commits.history import git  # noqa: E402
from documentation import docs_follow  # noqa: E402
from documentation.docs_follow import Follow  # noqa: E402

K_EXIT_CLEAN = 0
K_EXIT_REFUSED = 1
K_COMMENT = "#"
K_MERGE = "Merge "


def strip_comments(text: str) -> str:
    """Returns a message file's text without git's comment lines."""
    return "\n".join(line for line in text.splitlines() if not line.startswith(K_COMMENT))


def is_merge(message: str) -> bool:
    """Returns whether a message is a merge commit's, which the hook leaves alone."""
    return message.startswith(K_MERGE)


def commit_problems(
    message: str, known: frozenset[str] | None, id_pattern: str, follow: Follow | None,
) -> list[str]:
    """Returns what is wrong with a commit being made; empty when the hook accepts it.

    *known* is the IDs open at HEAD or on disk, and None when the `Backlog:` line is not required.
    *follow* is the staged change as the page rules read it, and None when pages need not follow.
    """
    problems: list[str] = []
    if known is not None:
        problems += check_backlog_trailer.trailer_problems(message, known, id_pattern)
    if follow is not None:
        problems += docs_follow.page_problems(follow)
    return problems


def refusal_lines(problems: list[str]) -> list[str]:
    """Returns the line printed for each problem."""
    return [f"commit refused: {problem}" for problem in problems]


def _read_known(root: Path) -> frozenset[str] | None:
    """Returns the IDs open at HEAD or on disk, or None when no base is set for the `Backlog:` line."""
    if not check_backlog_trailer.K_SINCE:
        return None
    return check_backlog_trailer.open_ids(root, "HEAD") | check_backlog_trailer.open_ids(root, None)


def _read_follow(root: Path, message: str) -> Follow | None:
    """Returns the staged change as the page rules read it, or None when no base is set for the pages."""
    if not docs_follow.K_SINCE:
        return None
    listing = git(root, "diff", "--cached", "--name-status", "-M") or ""
    return docs_follow.read_follow(root, docs_follow.parse_changes(listing), message, "HEAD", None)


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
    problems = commit_problems(
        message, _read_known(root), check_backlog_trailer.K_ID_PATTERN, _read_follow(root, message))
    for line in refusal_lines(problems):
        print(line, file=sys.stderr)
    return K_EXIT_REFUSED if problems else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

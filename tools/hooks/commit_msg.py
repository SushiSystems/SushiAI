# commit_msg.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Refuses a commit message that does not name the backlog entry it serves.

Usage: python tools/hooks/commit_msg.py MESSAGE_FILE

Git runs it through the commit-msg hook. A repository whose settings set no K_TRAILER_SINCE is not
checked, and a merge commit is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import check_backlog_trailer  # noqa: E402

K_EXIT_CLEAN = 0
K_EXIT_REFUSED = 1
K_COMMENT = "#"
K_MERGE = "Merge "


def _message(path: Path) -> str:
    """Returns the message in a file with git's comment lines removed."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(line for line in lines if not line.startswith(K_COMMENT))


def main(argv: list[str] | None = None) -> int:
    """Returns 1 and prints each fault when the message in the given file breaks the rule."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: commit_msg.py MESSAGE_FILE", file=sys.stderr)
        return K_EXIT_REFUSED
    if not check_backlog_trailer.K_SINCE:
        return K_EXIT_CLEAN
    message = _message(Path(arguments[0]))
    if message.startswith(K_MERGE):
        return K_EXIT_CLEAN
    root = Path.cwd()
    known = check_backlog_trailer.open_ids(root, "HEAD") | check_backlog_trailer.open_ids(root, None)
    problems = check_backlog_trailer.trailer_problems(message, known)
    for problem in problems:
        print(f"commit refused: {problem}", file=sys.stderr)
    return K_EXIT_REFUSED if problems else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

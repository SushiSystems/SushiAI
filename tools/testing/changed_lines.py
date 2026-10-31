# changed_lines.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads which lines of which files a working tree adds or edits after a commit.

The root is the top of the repository, since git names the paths in a diff from there.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits.history import git  # noqa: E402

K_FILE = re.compile(r"^\+\+\+ b/(.+?)\t?$")
K_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
K_QUIET = ("-c", "core.quotepath=off")


def changed_lines(root: Path, since: str) -> dict[Path, frozenset[int]]:
    """Returns, for each file the working tree adds or edits after *since*, the lines it adds or edits.

    A file git does not track yet counts every one of its lines. A deleted file is left out.
    """
    diff = git(root, *K_QUIET, "diff", "-U0", "--no-color", "--no-ext-diff", since, "--")
    if diff is None:
        raise SystemExit(f"cannot read the changes after {since}; fetch the full history")
    lines: dict[Path, set[int]] = {}
    current: Path | None = None
    for text in diff.splitlines():
        if text.startswith("+++ "):
            file = K_FILE.match(text)
            current = root / file.group(1) if file else None
            continue
        hunk = K_HUNK.match(text)
        if hunk and current is not None:
            first = int(hunk.group(1))
            count = 1 if hunk.group(2) is None else int(hunk.group(2))
            lines.setdefault(current, set()).update(range(first, first + count))
    for path in _untracked(root):
        lines[path] = set(range(1, _line_count(path) + 1))
    return {path: frozenset(numbers) for path, numbers in lines.items()}


def _untracked(root: Path) -> list[Path]:
    """Returns the files git does not track and does not ignore."""
    listing = git(root, *K_QUIET, "ls-files", "--others", "--exclude-standard", "-z")
    return [root / name for name in (listing or "").split("\0") if name]


def _line_count(path: Path) -> int:
    """Returns the number of lines in a file, or 0 when it cannot be read as text."""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0

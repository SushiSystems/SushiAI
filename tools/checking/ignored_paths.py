# ignored_paths.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the paths git ignores, so a checker reads only what the repository owns."""

from __future__ import annotations

import subprocess
from pathlib import Path

K_EXIT_SOME_IGNORED = 0
K_EXIT_NONE_IGNORED = 1


def _git(root: Path, arguments: list[str], stdin: str = "") -> str | None:
    """Returns the output of a git command, or None outside a repository or without git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            input=stdin, capture_output=True, text=True, encoding="utf-8",
        )
    except OSError:
        return None
    if completed.returncode not in (K_EXIT_SOME_IGNORED, K_EXIT_NONE_IGNORED):
        return None
    return completed.stdout


def drop_ignored(root: Path, paths: list[Path]) -> list[Path]:
    """Returns the paths git does not ignore below root; all of them outside a repository."""
    if not paths:
        return paths
    output = _git(root, ["check-ignore", "-z", "--stdin"],
                  "\0".join(str(path.resolve()) for path in paths) + "\0")
    if output is None:
        return paths
    ignored = {Path(entry).resolve() for entry in output.split("\0") if entry}
    return [path for path in paths if path.resolve() not in ignored]


def ignored_folders(root: Path) -> frozenset[Path]:
    """Returns every folder below root that git ignores whole; none outside a repository."""
    output = _git(root, ["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"])
    if output is None:
        return frozenset()
    return frozenset(
        (root / entry).resolve() for entry in output.split("\0") if entry.endswith("/")
    )

# history.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads the commits after a base from a repository, for the checkers that judge each commit."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Commit:
    """Holds one commit to check: where it is, its hash, its parent and its message."""

    root: Path
    sha: str
    parent: str
    message: str


def git(root: Path, *arguments: str) -> str | None:
    """Returns the output of a git command, or None when it fails."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


K_FIELD = "\x1f"
K_RECORD = "\x1e"


def commits_since(root: Path, since: str) -> list[Commit]:
    """Returns each non-merge commit after *since* below HEAD, oldest first; stops when git cannot read them.

    One git call reads them all, so the cost does not grow with the number of commits. A commit with
    no parent, the first of the history, is left out.
    """
    listing = git(root, "log", "--no-merges", "--reverse",
                  f"--format=%H{K_FIELD}%P{K_FIELD}%B{K_RECORD}", f"{since}..HEAD")
    if listing is None:
        raise SystemExit(f"cannot read the history after {since}; fetch the full history")
    commits: list[Commit] = []
    for record in listing.split(K_RECORD):
        if not record.strip():
            continue
        sha, parent, message = record.lstrip("\n").split(K_FIELD, 2)
        if parent.strip():
            commits.append(Commit(root, sha.strip(), parent.strip(), message.rstrip("\n") + "\n"))
    return commits


def blob_texts(root: Path, hashes: Sequence[str]) -> dict[str, str]:
    """Returns the text of each object hash in one git call; an object git does not hold is left out."""
    if not hashes:
        return {}
    try:
        completed = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                                   input=("\n".join(hashes) + "\n").encode(), capture_output=True)
    except OSError:
        return {}
    data, position, texts = completed.stdout, 0, {}
    while position < len(data):
        end = data.index(b"\n", position)
        header = data[position:end].split()
        position = end + 1
        if len(header) == 3:
            size = int(header[2])
            texts[header[0].decode()] = data[position:position + size].decode("utf-8", errors="replace")
            position += size + 1
    return texts

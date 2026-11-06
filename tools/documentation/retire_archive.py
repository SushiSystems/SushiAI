# retire_archive.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Stops tracking docs/archive: tags the history that holds it, unstages it and ignores it, once nothing cites it.

Usage: python tools/documentation/retire_archive.py [repository root] [--apply]

Without `--apply` nothing is changed. With it, the changes are staged and left for the orchestrator to commit.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import archive_citations  # noqa: E402

K_ARCHIVE = "docs/archive"
K_IGNORE_LINE = "docs/archive/"
K_TAG_PREFIX = "docs-archive-"
K_EXIT_CLEAN = 0
K_EXIT_BLOCKED = 1


@dataclass(frozen=True, slots=True)
class Plan:
    """Holds what the command prints, the exit code, and whether the shell tags, unstages and ignores the archive."""

    lines: tuple[str, ...]
    after: tuple[str, ...]
    code: int
    acts: bool
    tag: str


def tag_name(day: datetime.date) -> str:
    """Returns the name of the tag that marks the history holding the archive on a day."""
    return f"{K_TAG_PREFIX}{day.isoformat()}"


def with_ignore_line(text: str) -> str | None:
    """Returns the text of a .gitignore with the archive's line added, or None when it already names the archive."""
    lines = text.splitlines()
    if K_IGNORE_LINE in lines:
        return None
    return "\n".join([*lines, K_IGNORE_LINE]) + "\n"


def plan(tracked: int, citations: Sequence[tuple[str, int]], day: datetime.date, apply: bool) -> Plan:
    """Returns what retiring the archive prints and does, given the tracked files and the citations that remain."""
    if not tracked:
        return Plan(("the archive holds no tracked file; nothing to retire",), (), K_EXIT_CLEAN, False, "")
    cited = tuple(f"{name}:{number}: cites the archive; repoint it to the topic that holds the decision" for name, number in citations)
    if citations:
        return Plan(cited, (), K_EXIT_BLOCKED, False, "")
    tag = tag_name(day)
    summary = f"{tracked} tracked files; tag {tag}, unstage docs/archive, ignore it" + ("" if apply else " (dry run)")
    if not apply:
        return Plan((summary,), (), K_EXIT_CLEAN, False, tag)
    return Plan((summary,), ("staged; review `git diff --cached --stat` and commit",), K_EXIT_CLEAN, True, tag)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Runs a git command in the repository and returns its result."""
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, encoding="utf-8")


def tracked_files(root: Path) -> list[str]:
    """Returns each tracked file below docs/archive."""
    return [line for line in _git(root, "ls-files", "--", K_ARCHIVE).stdout.splitlines() if line]


def _ensure_ignored(root: Path) -> None:
    """Adds the archive to .gitignore when it is not there, and stages the file."""
    ignore = root / ".gitignore"
    text = ignore.read_text(encoding="utf-8") if ignore.is_file() else ""
    updated = with_ignore_line(text)
    if updated is not None:
        ignore.write_text(updated, encoding="utf-8", newline="")
    _git(root, "add", "--", ".gitignore")


def main(argv: list[str] | None = None) -> int:
    """Reports what retiring the archive would do, or does it; returns 1 while a citation of the archive remains."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--apply", action="store_true", help="tag, unstage and ignore the archive")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = arguments.root.resolve()
    files = tracked_files(root)
    citations = [(path.relative_to(root).as_posix(), number) for path, number in archive_citations(root)] if files else []
    decision = plan(len(files), citations, datetime.date.today(), arguments.apply)
    for line in decision.lines:
        print(line)
    if not decision.acts:
        return decision.code
    if _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{decision.tag}").returncode != 0:
        _git(root, "tag", decision.tag, "HEAD")
    _git(root, "rm", "-r", "--cached", "-q", "--", K_ARCHIVE)
    _ensure_ignored(root)
    for line in decision.after:
        print(line)
    return decision.code


if __name__ == "__main__":
    sys.exit(main())

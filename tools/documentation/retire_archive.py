# retire_archive.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Stops tracking docs/archive: tags the history that holds it, unstages it and ignores it, once nothing cites it.

Usage: python tools/documentation/retire_archive.py [repository root] [--apply]

Without `--apply` nothing is changed. With it, the changes are staged and left for the orchestrator to commit.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import archive_citations  # noqa: E402

K_ARCHIVE = "docs/archive"
K_IGNORE_LINE = "docs/archive/"
K_TAG_PREFIX = "docs-archive-"
K_EXIT_CLEAN = 0
K_EXIT_BLOCKED = 1


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Runs a git command in the repository and returns its result."""
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, encoding="utf-8")


def tracked_files(root: Path) -> list[str]:
    """Returns each tracked file below docs/archive."""
    return [line for line in _git(root, "ls-files", "--", K_ARCHIVE).stdout.splitlines() if line]


def _ensure_ignored(root: Path) -> None:
    """Adds the archive to .gitignore when it is not there, and stages the file."""
    ignore = root / ".gitignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.is_file() else []
    if K_IGNORE_LINE not in lines:
        ignore.write_text("\n".join([*lines, K_IGNORE_LINE]) + "\n", encoding="utf-8", newline="")
    _git(root, "add", "--", ".gitignore")


def main(argv: list[str] | None = None) -> int:
    """Reports what retiring the archive would do, or does it; returns 1 while a citation of the archive remains."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--apply", action="store_true", help="tag, unstage and ignore the archive")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = arguments.root.resolve()
    files = tracked_files(root)
    if not files:
        print("the archive holds no tracked file; nothing to retire")
        return K_EXIT_CLEAN
    citations = archive_citations(root)
    for path, number in citations:
        print(f"{path.relative_to(root).as_posix()}:{number}: cites the archive; repoint it to the topic that holds the decision")
    if citations:
        return K_EXIT_BLOCKED
    tag = f"{K_TAG_PREFIX}{datetime.date.today().isoformat()}"
    print(f"{len(files)} tracked files; tag {tag}, unstage docs/archive, ignore it" + ("" if arguments.apply else " (dry run)"))
    if not arguments.apply:
        return K_EXIT_CLEAN
    if _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode != 0:
        _git(root, "tag", tag, "HEAD")
    _git(root, "rm", "-r", "--cached", "-q", "--", K_ARCHIVE)
    _ensure_ignored(root)
    print("staged; review `git diff --cached --stat` and commit")
    return K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

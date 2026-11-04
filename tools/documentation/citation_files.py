# citation_files.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Decides which files of a repository can cite a document: the manual, the source and the build files."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from checking.ignored_paths import drop_ignored

K_EXTENSIONS = frozenset({".md", ".cpp", ".hpp", ".h", ".py", ".ts", ".glsl", ".comp", ".toml", ".json", ".txt", ".cmake"})
K_SKIPPED_FOLDERS = frozenset({".git", "third_party", "node_modules", "build"})
K_ARCHIVE = "docs/archive"
K_SHARED_TOOLS = tuple(
    f"tools/{name}" for name in ("checking", "documentation", "layering", "licensing", "logs", "routing", "testing", "tests")
)
K_ARCHIVE_CITATION = re.compile(r"(?:docs/|\.\./)archive/")


def is_citable(name: str) -> bool:
    """Returns whether a root-relative file name may hold a citation: a known extension, outside the skipped folders, the archive and the shared tools."""
    path = PurePosixPath(name)
    if path.suffix not in K_EXTENSIONS or K_SKIPPED_FOLDERS.intersection(path.parts[:-1]):
        return False
    return not name.startswith((f"{K_ARCHIVE}/", *(f"{folder}/" for folder in K_SHARED_TOOLS)))


def archive_citation_lines(text: str) -> list[int]:
    """Returns the number of each line of a text that cites the archive."""
    return [number for number, line in enumerate(text.splitlines(), start=1) if K_ARCHIVE_CITATION.search(line)]


def citing_files(root: Path) -> list[Path]:
    """Returns each file below root that may hold citations: not ignored, not vendored, not in the archive.

    The shared tools and their tests name paths and write fake citations as fixtures, so they are left out.
    """
    found = _walk(root)
    return drop_ignored(root, [path for path in found if is_citable(_relative(root, path))])


def archive_citations(root: Path) -> list[tuple[Path, int]]:
    """Returns the file and line of each citation of the archive outside it."""
    found: list[tuple[Path, int]] = []
    for path in citing_files(root):
        found += [(path, number) for number in archive_citation_lines(_read_text(path))]
    return found


def _relative(root: Path, path: Path) -> str:
    """Returns a path below the root as a root-relative POSIX name."""
    return path.relative_to(root).as_posix()


def _walk(root: Path) -> list[Path]:
    """Returns every file below root, leaving out the vendored and build folders the walk prunes."""
    found: list[Path] = []
    for folder, names, files in os.walk(root):
        names[:] = [name for name in names if name not in K_SKIPPED_FOLDERS]
        found += [Path(folder) / name for name in files]
    return found


def _read_text(path: Path) -> str:
    """Returns the text of a file, with the bytes it cannot decode replaced."""
    return path.read_text(encoding="utf-8", errors="replace")

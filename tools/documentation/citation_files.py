# citation_files.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Lists the files of a repository that can cite a document: the manual, the source and the build files."""

from __future__ import annotations

import os
import re
from pathlib import Path

from checking.ignored_paths import drop_ignored

K_EXTENSIONS = frozenset({".md", ".cpp", ".hpp", ".h", ".py", ".ts", ".glsl", ".comp", ".toml", ".json", ".txt", ".cmake"})
K_SKIPPED_FOLDERS = frozenset({".git", "third_party", "node_modules", "build"})
K_ARCHIVE = Path("docs") / "archive"
K_SHARED_TOOLS = tuple(
    Path("tools") / name for name in ("checking", "documentation", "layering", "licensing", "logs", "routing", "testing", "tests")
)
K_ARCHIVE_CITATION = re.compile(r"(?:docs/|\.\./)archive/")


def citing_files(root: Path) -> list[Path]:
    """Returns each file below root that may hold citations: not ignored, not vendored, not in the archive.

    The shared tools and their tests name paths and write fake citations as fixtures, so they are left out.
    """
    found: list[Path] = []
    for folder, names, files in os.walk(root):
        names[:] = [name for name in names if name not in K_SKIPPED_FOLDERS]
        found += [Path(folder) / name for name in files if Path(name).suffix in K_EXTENSIONS]
    skipped = [(root / K_ARCHIVE).resolve(), *((root / folder).resolve() for folder in K_SHARED_TOOLS)]
    return [path for path in drop_ignored(root, found) if not any(folder in path.resolve().parents for folder in skipped)]


def archive_citations(root: Path) -> list[tuple[Path, int]]:
    """Returns the file and line of each citation of the archive outside it."""
    found: list[tuple[Path, int]] = []
    for path in citing_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        found += [(path, number) for number, line in enumerate(text.splitlines(), start=1) if K_ARCHIVE_CITATION.search(line)]
    return found

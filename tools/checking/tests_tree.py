# tests_tree.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Finds the test files a repository holds, for the checkers that read them."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from checking.ignored_paths import drop_ignored, ignored_folders

K_TEST_FOLDERS = frozenset({"tests", "__tests__"})
K_SKIPPED_FOLDERS = frozenset({".git", "node_modules"})


@dataclass(frozen=True, slots=True)
class TestTree:
    """Holds the test files of one repository that git tracks."""

    root: Path
    files: tuple[Path, ...]

    def of(self, is_kind: Callable[[Path], bool]) -> Iterator[Path]:
        """Yields the files whose path the predicate accepts."""
        return (path for path in self.files if is_kind(path))


def _test_folders(root: Path) -> Iterator[Path]:
    """Yields every test folder below a repository, skipping ignored and nested ones."""
    ignored = ignored_folders(root)
    for current, names, _ in os.walk(root):
        names[:] = sorted(
            name for name in names
            if name not in K_SKIPPED_FOLDERS and (Path(current) / name).resolve() not in ignored
        )
        found = [name for name in names if name in K_TEST_FOLDERS]
        for name in found:
            yield Path(current) / name
        names[:] = [name for name in names if name not in found]


def discover(root: Path) -> TestTree:
    """Returns the files under every test folder of a repository, at any depth."""
    found = sorted(
        path for folder in _test_folders(root) for path in folder.rglob("*") if path.is_file()
    )
    return TestTree(root, tuple(drop_ignored(root, found)))

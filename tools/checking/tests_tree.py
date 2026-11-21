# tests_tree.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Finds the test files a repository holds, for the checkers that read them.

A test folder is any folder named `tests` or `__tests__`, and `Assets/<Project>/Tests` of a Unity project.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from checking.ignored_paths import drop_ignored, walk_files

K_TEST_FOLDERS = frozenset({"tests", "__tests__"})
K_UNITY_TEST_FOLDER = "Tests"
K_UNITY_ASSETS_FOLDER = "Assets"
K_SKIPPED_FOLDERS = frozenset({".git", "node_modules"})


@dataclass(frozen=True, slots=True)
class TestTree:
    """Holds the test files of one repository that git tracks."""

    root: Path
    files: tuple[Path, ...]

    def of(self, is_kind: Callable[[Path], bool]) -> Iterator[Path]:
        """Yields the files whose path the predicate accepts."""
        return (path for path in self.files if is_kind(path))


def _is_test_folder(parent: Path, name: str) -> bool:
    """Returns whether a folder is a test folder: named like one, or `Tests` of a Unity project, `Assets/<Project>/Tests`."""
    return name in K_TEST_FOLDERS or (
        name == K_UNITY_TEST_FOLDER and parent.parent.name == K_UNITY_ASSETS_FOLDER)


def _under_test_folder(root: Path, path: Path) -> bool:
    """Returns whether a file lies below a test folder that no skipped folder precedes."""
    current = root
    for name in path.relative_to(root).parts[:-1]:
        if name in K_SKIPPED_FOLDERS:
            return False
        if _is_test_folder(current, name):
            return True
        current = current / name
    return False


def discover(root: Path) -> TestTree:
    """Returns the files under every test folder of a repository, at any depth."""
    found = [path for path in walk_files(root) if _under_test_folder(root, path)]
    return TestTree(root, tuple(drop_ignored(root, found)))

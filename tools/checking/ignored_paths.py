# ignored_paths.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Names the paths git ignores, so a checker reads only what the repository owns."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
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
    """Returns the paths git does not ignore below root; all of them outside a repository.

    A path below an ignored folder is dropped by the folder, so a nested clone inside it, which makes
    ``git check-ignore`` fail, cannot put the whole tree back.
    """
    if not paths:
        return paths
    folders = ignored_folders(root)
    if folders:
        paths = [path for path in paths if not folders.intersection(path.resolve().parents)]
    output = _git(root, ["check-ignore", "-z", "--stdin"],
                  "\0".join(str(path.resolve()) for path in paths) + "\0")
    if output is None:
        return paths
    ignored = {Path(entry).resolve() for entry in output.split("\0") if entry}
    return [path for path in paths if path.resolve() not in ignored]


def walk_files(root: Path) -> list[Path]:
    """Returns every file below root, sorted, without entering a folder git ignores whole.

    Pruning in place keeps a large ignored folder, such as a vendored dependency tree, from being listed
    only to be dropped afterwards. Files git ignores by their own name are still left to ``drop_ignored``.
    """
    ignored = ignored_folders(root)
    found: list[Path] = []
    for folder, names, files in os.walk(root):
        names[:] = [name for name in names if not ignored or (Path(folder) / name).resolve() not in ignored]
        found.extend(Path(folder) / name for name in files)
    return sorted(found)


def folders_of(files: Iterable[Path], root: Path) -> set[Path]:
    """Returns every folder strictly below root that holds one of the files, directly or deeper."""
    folders: set[Path] = set()
    for path in files:
        folders.update(root / parent for parent in path.relative_to(root).parents if parent != Path("."))
    return folders


def ignored_folders(root: Path) -> frozenset[Path]:
    """Returns every folder below root that git ignores whole; none outside a repository.

    Git lists an ignored folder with a trailing slash, but a folder link differently: a Windows
    junction is a folder to git and keeps the slash, a POSIX symlink is a file to git and loses it.
    An entry without the slash that opens as a folder is therefore a link to one and counts too.
    """
    output = _git(root, ["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"])
    if output is None:
        return frozenset()
    return frozenset(
        (root / entry).resolve() for entry in output.split("\0")
        if entry and (entry.endswith("/") or (root / entry).is_dir())
    )

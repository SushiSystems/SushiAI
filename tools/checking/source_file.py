# source_file.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Holds a file's path and text as values, for checker rules that read no disk."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Holds one file's path, as the caller names it, and its text."""

    path: Path
    text: str


def line_of(text: str, offset: int) -> int:
    """Returns the line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def read_files(paths: Iterable[Path], wanted: Callable[[Path], bool]) -> tuple[SourceFile, ...]:
    """Reads each path the predicate accepts as UTF-8, replacing bytes that do not decode."""
    return tuple(
        SourceFile(path, path.read_text(encoding="utf-8", errors="replace"))
        for path in paths
        if wanted(path)
    )

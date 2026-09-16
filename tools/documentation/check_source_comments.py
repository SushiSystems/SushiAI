# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Checks C++, GLSL, TypeScript and Python source against the source-comments skill.

Usage: python tools/documentation/check_source_comments.py [paths...] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_C_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".glsl", ".comp", ".vert", ".frag", ".ts", ".tsx"})
K_PYTHON_SUFFIXES = frozenset({".py"})
K_SKIPPED_FOLDERS = frozenset({".git", "third_party", "build", "__pycache__", "node_modules"})
K_FILE_HEADER_CEILING = 6
K_BLOCK_CEILING = 8
K_SEPARATOR = re.compile(r"[-=*_~#]{5,}")
K_HISTORY = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b|\b(previously|formerly|fixed in)\b", re.IGNORECASE)
K_DATE_TAG = re.compile(r"@date\b")
K_LICENSE_LINE = re.compile(r"^/\*.*\*/\s*$")


@dataclass(frozen=True, slots=True)
class CommentLine:
    """Holds one source line that lies inside a comment, outside the license block."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Holds a source file split into its license block and the lines after it."""

    path: Path
    lines: list[str]
    body_start: int

    @property
    def is_python(self) -> bool:
        """Returns whether the file is Python."""
        return self.path.suffix in K_PYTHON_SUFFIXES


def _license_end_c(lines: list[str]) -> int:
    """Returns the index of the first line after a leading boxed license block."""
    index = 0
    while index < len(lines) and K_LICENSE_LINE.match(lines[index]):
        index += 1
    return index


def _license_end_python(lines: list[str]) -> int:
    """Returns the index of the first line after a leading run of `#` lines."""
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        index += 1
    return index


def _read_source(path: Path) -> SourceFile:
    """Returns a source file with its license block located."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = _license_end_python(lines) if path.suffix in K_PYTHON_SUFFIXES else _license_end_c(lines)
    return SourceFile(path, lines, end)


def _walk(paths: list[Path]) -> Iterator[Path]:
    """Yields every checked source file below the given paths."""
    suffixes = K_C_SUFFIXES | K_PYTHON_SUFFIXES
    for path in paths:
        if path.is_file():
            if path.suffix in suffixes:
                yield path
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix in suffixes:
                if not K_SKIPPED_FOLDERS.intersection(candidate.parts):
                    yield candidate


def collect(paths: list[Path]) -> Iterable[SourceFile]:
    """Returns every source file below the given paths."""
    return (_read_source(path) for path in _walk(paths))


def _comment_lines_c(source: SourceFile) -> Iterator[CommentLine]:
    """Yields the lines after the license that lie inside a C-style comment."""
    in_block = False
    for index in range(source.body_start, len(source.lines)):
        stripped = source.lines[index].strip()
        if in_block:
            yield CommentLine(index + 1, stripped)
            in_block = "*/" not in stripped
        elif stripped.startswith("/*"):
            yield CommentLine(index + 1, stripped)
            in_block = "*/" not in stripped
        elif "//" in stripped:
            yield CommentLine(index + 1, stripped[stripped.index("//"):])


def _comment_lines_python(source: SourceFile) -> Iterator[CommentLine]:
    """Yields the lines after the license that are `#` comments."""
    for index in range(source.body_start, len(source.lines)):
        stripped = source.lines[index].strip()
        if stripped.startswith("#"):
            yield CommentLine(index + 1, stripped)


def _comment_lines(source: SourceFile) -> Iterator[CommentLine]:
    """Yields the comment lines of a file in either language."""
    return _comment_lines_python(source) if source.is_python else _comment_lines_c(source)


def _doc_blocks_c(source: SourceFile) -> Iterator[tuple[int, int]]:
    """Yields the first and last line index of every `/**` block after the license."""
    index = source.body_start
    while index < len(source.lines):
        if source.lines[index].strip().startswith("/**"):
            first = index
            while index < len(source.lines) and "*/" not in source.lines[index]:
                index += 1
            yield first, index
        index += 1


def _slash_runs(source: SourceFile, marker: str) -> Iterator[tuple[int, int]]:
    """Yields the first line index and length of every run of lines starting with a marker."""
    index = source.body_start
    while index < len(source.lines):
        if source.lines[index].strip().startswith(marker):
            first = index
            while index < len(source.lines) and source.lines[index].strip().startswith(marker):
                index += 1
            yield first, index - first
        else:
            index += 1


def rule_file_header(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue when the file does not open with its `@file` block or module docstring."""
    if source.is_python:
        yield from _python_module_docstring(source)
        return

    index = source.body_start
    while index < len(source.lines) and not source.lines[index].strip():
        index += 1
    blocks = list(_doc_blocks_c(source))
    if not blocks or blocks[0][0] != index:
        yield Issue(source.path, index + 1, "file must open with a /** @file block after the license")
        return

    first, last = blocks[0]
    text = "\n".join(source.lines[first:last + 1])
    if f"@file {source.path.name}" not in text:
        yield Issue(source.path, first + 1, f"@file must name {source.path.name}")
    for tag in ("@brief", "@author"):
        if tag not in text:
            yield Issue(source.path, first + 1, f"file block lacks {tag}")
    if last - first + 1 > K_FILE_HEADER_CEILING:
        yield Issue(source.path, first + 1, f"file block exceeds {K_FILE_HEADER_CEILING} lines")


def _python_module_docstring(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue when a Python file lacks a short module docstring."""
    try:
        module = ast.parse("\n".join(source.lines))
    except SyntaxError as error:
        yield Issue(source.path, error.lineno or 1, "file does not parse")
        return

    docstring = ast.get_docstring(module, clean=False)
    if docstring is None:
        yield Issue(source.path, source.body_start + 1, "module must open with a docstring")
    elif len(docstring.strip().splitlines()) > K_FILE_HEADER_CEILING:
        yield Issue(source.path, source.body_start + 1, f"module docstring exceeds {K_FILE_HEADER_CEILING} lines")


def rule_block_ceiling(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue for each symbol comment block longer than the ceiling."""
    if source.is_python:
        return
    for position, (first, last) in enumerate(_doc_blocks_c(source)):
        if position > 0 and last - first + 1 > K_BLOCK_CEILING:
            yield Issue(source.path, first + 1, f"comment block exceeds {K_BLOCK_CEILING} lines")
    for first, length in _slash_runs(source, "///"):
        if length > K_BLOCK_CEILING:
            yield Issue(source.path, first + 1, f"/// run exceeds {K_BLOCK_CEILING} lines")


def rule_comment_runs(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue for each run of two or more line comments."""
    marker = "#" if source.is_python else "//"
    for first, length in _slash_runs(source, marker):
        text = source.lines[first].strip()
        if not source.is_python and text.startswith("///"):
            continue
        if length > 1:
            yield Issue(source.path, first + 1, f"{length} consecutive {marker} lines form a paragraph")


def rule_separators(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue for each separator line inside a comment outside the license."""
    for comment in _comment_lines(source):
        if K_SEPARATOR.search(comment.text):
            yield Issue(source.path, comment.number, "separator line outside the license block")


def rule_history(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue for each TODO-style tag or history word inside a comment."""
    for comment in _comment_lines(source):
        match = K_HISTORY.search(comment.text)
        if match:
            yield Issue(source.path, comment.number, f"'{match.group(0)}' belongs in the backlog or git")


def rule_no_date(source: SourceFile) -> Iterator[Issue]:
    """Yields an issue for each @date tag."""
    for comment in _comment_lines(source):
        if K_DATE_TAG.search(comment.text):
            yield Issue(source.path, comment.number, "@date is forbidden; git records dates")


K_CHECKER = Checker(
    description="Checks source comments against the source-comments skill.",
    collect=collect,
    rules={
        "rule_file_header": rule_file_header,
        "rule_block_ceiling": rule_block_ceiling,
        "rule_comment_runs": rule_comment_runs,
        "rule_separators": rule_separators,
        "rule_history": rule_history,
        "rule_no_date": rule_no_date,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

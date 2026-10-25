# check_device_sources.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that exactly the units which reach a kernel are listed in DEVICE_SOURCES.

An unlisted kernel unit compiles without -fsycl and fails at run time; a listed unit with no kernel
pays the device pass for nothing; a test or application source cannot be listed at all.

Usage: python tools/layering/check_device_sources.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import fnmatch
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from settings import K_MODULE_FUNCTION  # noqa: E402

# Include paths (fnmatch patterns) of the headers that launch a kernel; a unit reaching one needs -fsycl.
K_KERNEL_HEADERS = (
    "SushiBLAS/elementwise/dispatch.hpp",
    "SushiBLAS/functor_map/*",
    "SushiBLAS/launch/launch_1d.hpp",
    "SushiBLAS/reduction/*",
    "SushiBLAS/random/dispatch.hpp",
    "SushiBLAS/blas_seam_kernels/*",
    "SushiRuntime/api/detail/emit_*",
    "SushiRuntime/api/launch/nd_range.hpp",
)

K_MODULES = "modules"
K_OUTSIDE_FOLDERS = ("tests", "applications")
K_PUBLIC_FOLDER = "include"
K_SOURCE_FOLDER = "source"
K_PROJECT_PREFIX = "SushiAI/"
K_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")')
K_KERNEL_CALL = re.compile(r"\b(?:parallel_for|single_task)\(")
K_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
K_LINE_COMMENT = re.compile(r"//[^\n]*")
K_CMAKE_COMMENT = re.compile(r"#[^\n]*")
K_CMAKE_KEYWORD = re.compile(r"[A-Z][A-Z_]*")
K_CMAKE_TOKEN = re.compile(r"[^\s()]+")


@dataclass(frozen=True, slots=True)
class Reach:
    """Holds the first line of a unit that leads to a kernel and what it leads to."""

    line: int
    via: str


@dataclass(frozen=True, slots=True)
class Entry:
    """Holds one file named in a module's SOURCES or DEVICE_SOURCES, with its line in the build file."""

    path: Path
    build_file: Path
    line: int


@dataclass(frozen=True, slots=True)
class ModuleBuild:
    """Holds the SOURCES and DEVICE_SOURCES of one module's declaring call."""

    folder: Path
    sources: tuple[Entry, ...]
    device_sources: tuple[Entry, ...]


def _blank_comments(text: str, pattern: re.Pattern[str]) -> str:
    """Returns text with each match of a comment pattern replaced by blanks, keeping its newlines."""
    return pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)


def _parse_call(build_file: Path, function: str) -> ModuleBuild | None:
    """Returns the SOURCES and DEVICE_SOURCES of the module call in a build file, None without one."""
    text = _blank_comments(build_file.read_text(encoding="utf-8"), K_CMAKE_COMMENT)
    call = re.search(rf"\b{re.escape(function)}\s*\(([^)]*)\)", text)
    if call is None:
        return None
    lists: dict[str, list[Entry]] = {"SOURCES": [], "DEVICE_SOURCES": []}
    keyword = ""
    for token in K_CMAKE_TOKEN.finditer(call.group(1)):
        word = token.group(0)
        if K_CMAKE_KEYWORD.fullmatch(word):
            keyword = word
        elif keyword in lists and "$" not in word:
            line = text.count("\n", 0, call.start(1) + token.start()) + 1
            lists[keyword].append(Entry(build_file.parent / word, build_file, line))
    return ModuleBuild(build_file.parent, tuple(lists["SOURCES"]), tuple(lists["DEVICE_SOURCES"]))


class UnitTree:
    """Holds one repository's modules, include roots and the include graph of its C++ files."""

    def __init__(self, root: Path, function: str = K_MODULE_FUNCTION) -> None:
        """Remembers the root and the name of the module-declaring call."""
        self.root = root
        self.function = function
        self._lines: dict[Path, list[str]] = {}

    @cached_property
    def builds(self) -> list[ModuleBuild]:
        """Returns the declaring call of every modules/<tier>/<module>/CMakeLists.txt that has one."""
        found = (_parse_call(path, self.function)
                 for path in sorted((self.root / K_MODULES).glob("*/*/CMakeLists.txt")))
        return [build for build in found if build is not None]

    @cached_property
    def include_roots(self) -> list[Path]:
        """Returns every modules/<tier>/<module>/include folder."""
        return sorted(path for path in (self.root / K_MODULES).glob(f"*/*/{K_PUBLIC_FOLDER}") if path.is_dir())

    def lines_of(self, path: Path) -> list[str]:
        """Returns the lines of a C++ file with its comments blanked, empty when it is unreadable."""
        if path not in self._lines:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            text = _blank_comments(_blank_comments(text, K_BLOCK_COMMENT), K_LINE_COMMENT)
            self._lines[path] = text.splitlines()
        return self._lines[path]

    def _resolve(self, target: str, quoted: bool, includer: Path, source_folder: Path | None) -> Path | None:
        """Returns the repository file an include names, None when it names none."""
        candidates: list[Path] = []
        if quoted:
            candidates.append(includer.parent / target)
            if source_folder is not None:
                candidates.append(source_folder / target)
        if quoted or target.startswith(K_PROJECT_PREFIX):
            candidates += [root / target for root in self.include_roots]
        return next((path for path in candidates if path.is_file()), None)

    def reach(self, unit: Path, source_folder: Path | None) -> Reach | None:
        """Returns where a unit first leads to a kernel, directly or through SushiAI headers it includes."""
        return self._walk(unit, source_folder, {unit})

    def _walk(self, path: Path, source_folder: Path | None, seen: set[Path]) -> Reach | None:
        """Searches one file and the headers below it; `seen` keeps a cycle from looping."""
        for number, line in enumerate(self.lines_of(path), start=1):
            if K_KERNEL_CALL.search(line):
                return Reach(number, "a kernel launch")
            match = K_INCLUDE.match(line)
            if match is None:
                continue
            target = match.group(1) or match.group(2)
            if any(fnmatch.fnmatchcase(target, pattern) for pattern in K_KERNEL_HEADERS):
                return Reach(number, target)
            header = self._resolve(target, match.group(2) is not None, path, source_folder)
            if header is None or header in seen:
                continue
            seen.add(header)
            inner = self._walk(header, source_folder, seen)
            if inner is not None:
                return Reach(number, f"{target} -> {inner.via}")
        return None

    def outside_units(self) -> Iterator[Path]:
        """Yields every .cpp below tests/ and applications/."""
        for folder in K_OUTSIDE_FOLDERS:
            base = self.root / folder
            if base.is_dir():
                yield from sorted(path for path in base.rglob("*.cpp") if path.is_file())


def collect(paths: list[Path]) -> Iterable[UnitTree]:
    """Returns the unit tree of each given repository that has modules."""
    return [UnitTree(path) for path in paths if (path / K_MODULES).is_dir()]


def rule_device_source_missing(tree: UnitTree) -> Iterator[Issue]:
    """Yields an issue for each module source that reaches a kernel and is not in DEVICE_SOURCES."""
    for build in tree.builds:
        listed = {entry.path for entry in build.device_sources}
        for entry in build.sources:
            if entry.path in listed or not entry.path.is_file():
                continue
            reach = tree.reach(entry.path, build.folder / K_SOURCE_FOLDER)
            if reach is not None:
                yield Issue(entry.path, reach.line, f"{entry.path.name} reaches a kernel through {reach.via} "
                                                    f"and is not in DEVICE_SOURCES of {build.folder.name}")


def rule_device_source_unneeded(tree: UnitTree) -> Iterator[Issue]:
    """Yields an issue for each DEVICE_SOURCES entry that reaches no kernel."""
    for build in tree.builds:
        for entry in build.device_sources:
            if entry.path.is_file() and tree.reach(entry.path, build.folder / K_SOURCE_FOLDER) is None:
                yield Issue(entry.build_file, entry.line, f"{entry.path.name} is in DEVICE_SOURCES "
                                                          "and reaches no kernel; list it in SOURCES only")


def rule_device_source_outside_module(tree: UnitTree) -> Iterator[Issue]:
    """Yields an issue for each test or application source that reaches a kernel."""
    for unit in tree.outside_units():
        reach = tree.reach(unit, None)
        if reach is not None:
            yield Issue(unit, reach.line, f"{unit.name} reaches a kernel through {reach.via} "
                                          "and no module lists it in DEVICE_SOURCES")


K_CHECKER = Checker(
    description="Checks that the units reaching a kernel are exactly the DEVICE_SOURCES of their module.",
    collect=collect,
    rules={
        "rule_device_source_missing": rule_device_source_missing,
        "rule_device_source_unneeded": rule_device_source_unneeded,
        "rule_device_source_outside_module": rule_device_source_outside_module,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

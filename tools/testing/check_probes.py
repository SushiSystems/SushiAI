# check_probes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that probes and scenario benchmarks live in one place, decide nothing and are listed.

Usage: python tools/testing/check_probes.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402

K_MIRRORED = frozenset({"modules", "tests", "source", "include"})
K_SKIPPED = frozenset({".git", "third_party", "build", "docs", "node_modules", "__pycache__"})
K_TEST_FRAMEWORK = re.compile(r"#\s*include\s*[<\"](?:gtest|gmock|catch2|doctest)/|^\s*import pytest\b", re.MULTILINE)
K_TEST_RUNNER = re.compile(r"\b(?:add_test|gtest_discover_tests|gtest_add_tests|catch_discover_tests)\s*\(")


@dataclass(frozen=True, slots=True)
class ProgramKind:
    """Holds one kind of program that lives in one folder under applications/ and decides nothing."""

    label: str
    folder: Path
    folder_name: re.Pattern[str]


K_KINDS = (
    ProgramKind("probe", Path("applications") / "probes", re.compile(r"probes?|.+_probes?")),
    ProgramKind("benchmark", Path("applications") / "benchmarks", re.compile(r"benchmarks|.+_benchmarks?")),
)


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds a repository root and the files git tracks below it, outside the skipped folders."""

    root: Path
    files: tuple[Path, ...]

    def under(self, folder: Path) -> Iterator[Path]:
        """Yields the files below a folder of the repository."""
        absolute = self.root / folder
        return (path for path in self.files if absolute in path.parents)

    def programs(self, kind: ProgramKind) -> list[str]:
        """Returns the name of each program folder of a kind, sorted."""
        absolute = self.root / kind.folder
        return sorted({path.relative_to(absolute).parts[0] for path in self.under(kind.folder)
                       if len(path.relative_to(absolute).parts) > 1})


def discover(root: Path) -> Repository:
    """Returns a repository with its tracked files."""
    found = [path for path in sorted(root.rglob("*"))
             if path.is_file() and not K_SKIPPED.intersection(path.relative_to(root).parts)]
    return Repository(root, tuple(drop_ignored(root, found)))


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder as a repository."""
    return [discover(path) for path in paths if path.is_dir()]


def rule_program_outside_folder(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each folder named as probes or benchmarks outside its one place.

    The source and its test mirror are left out, since a unit there may carry such a name.
    """
    folders = {parent for path in repository.files for parent in path.relative_to(repository.root).parents}
    for folder in sorted(folders):
        mirrored = folder.parts[0] in K_MIRRORED if folder.parts else False
        for kind in K_KINDS:
            inside = folder == kind.folder or kind.folder in folder.parents
            if folder.name and kind.folder_name.fullmatch(folder.name) and not inside and not mirrored:
                yield Issue(repository.root / folder, 1, f"{kind.label}s live in {kind.folder.as_posix()}/<name>/")


def rule_program_test_framework(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe or benchmark file that uses a test framework, which makes it a test."""
    for kind in K_KINDS:
        for path in repository.under(kind.folder):
            text = path.read_text(encoding="utf-8", errors="ignore")
            match = K_TEST_FRAMEWORK.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                yield Issue(path, line, f"a {kind.label} uses a test framework; a program that decides pass or "
                                        "fail is a test")


def rule_program_in_test_runner(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each build file of a probe or benchmark that adds a test to the runner."""
    for kind in K_KINDS:
        for path in repository.under(kind.folder):
            if path.name != "CMakeLists.txt":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in K_TEST_RUNNER.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                yield Issue(path, line, f"a {kind.label} is never added to the test runner")


def rule_program_name_stem(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe or benchmark whose name opens with another of its kind's."""
    for kind in K_KINDS:
        names = repository.programs(kind)
        for name in names:
            for other in names:
                if other != name and name.startswith(other):
                    yield Issue(repository.root / kind.folder / name, 1,
                                f"{kind.label} {name} opens with {kind.label} {other}'s name; make it an option "
                                f"of {other} or give it a name of its own")


def rule_unlisted_program(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe or benchmark folder its kind's README.md does not name."""
    for kind in K_KINDS:
        readme = repository.root / kind.folder / "README.md"
        text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        for name in repository.programs(kind):
            if not re.search(rf"`{re.escape(name)}`|^#+\s+{re.escape(name)}\b", text, re.MULTILINE):
                yield Issue(readme, 1, f"{kind.label} {name} is not listed in {kind.folder.as_posix()}/README.md")


K_CHECKER = Checker(
    description="Checks that probes and scenario benchmarks live in one place, decide nothing and are listed.",
    collect=collect,
    rules={
        "rule_program_outside_folder": rule_program_outside_folder,
        "rule_program_test_framework": rule_program_test_framework,
        "rule_program_in_test_runner": rule_program_in_test_runner,
        "rule_program_name_stem": rule_program_name_stem,
        "rule_unlisted_program": rule_unlisted_program,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

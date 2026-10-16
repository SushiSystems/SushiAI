# check_probes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that probes live in one place, decide nothing and are listed, from `testing`.

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

K_PROBES = Path("applications") / "probes"
K_PROBE_FOLDER = re.compile(r"probes?|.+_probes?")
K_MIRRORED = frozenset({"modules", "tests", "source", "include"})
K_SKIPPED = frozenset({".git", "third_party", "build", "docs", "node_modules", "__pycache__"})
K_TEST_FRAMEWORK = re.compile(r"#\s*include\s*[<\"](?:gtest|gmock|catch2|doctest)/|^\s*import pytest\b", re.MULTILINE)
K_TEST_RUNNER = re.compile(r"\b(?:add_test|gtest_discover_tests|gtest_add_tests|catch_discover_tests)\s*\(")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds a repository root and the files git tracks below it, outside the skipped folders."""

    root: Path
    files: tuple[Path, ...]

    @property
    def probes(self) -> Path:
        """Returns the one folder probes live in."""
        return self.root / K_PROBES

    def under(self, folder: Path) -> Iterator[Path]:
        """Yields the files below a folder."""
        return (path for path in self.files if folder in path.parents)


def discover(root: Path) -> Repository:
    """Returns a repository with its tracked files."""
    found = [path for path in sorted(root.rglob("*"))
             if path.is_file() and not K_SKIPPED.intersection(path.relative_to(root).parts)]
    return Repository(root, tuple(drop_ignored(root, found)))


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder as a repository."""
    return [discover(path) for path in paths if path.is_dir()]


def rule_probe_outside_folder(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each folder named as probes outside applications/probes, the source and its test mirror."""
    folders = {parent for path in repository.files for parent in path.relative_to(repository.root).parents}
    for folder in sorted(folders):
        inside = folder == K_PROBES or K_PROBES in folder.parents
        mirrored = folder.parts[0] in K_MIRRORED if folder.parts else False
        if folder.name and K_PROBE_FOLDER.fullmatch(folder.name) and not inside and not mirrored:
            yield Issue(repository.root / folder, 1, f"probes live in {K_PROBES.as_posix()}/<name>/")


def rule_probe_test_framework(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe file that uses a test framework, which makes it a test."""
    for path in repository.under(repository.probes):
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = K_TEST_FRAMEWORK.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            yield Issue(path, line, "a probe uses a test framework; a program that decides pass or fail is a test")


def rule_probe_in_test_runner(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each build file under the probes that adds a test to the runner."""
    for path in repository.under(repository.probes):
        if path.name != "CMakeLists.txt":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_TEST_RUNNER.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield Issue(path, line, "a probe is never added to the test runner")


def _probe_names(repository: Repository) -> list[str]:
    """Returns the name of each probe folder, sorted."""
    folder = repository.probes
    return sorted({path.relative_to(folder).parts[0] for path in repository.under(folder)
                   if len(path.relative_to(folder).parts) > 1})


def rule_probe_name_stem(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe whose name opens with another probe's name."""
    names = _probe_names(repository)
    for name in names:
        for other in names:
            if other != name and name.startswith(other):
                yield Issue(repository.probes / name, 1,
                            f"probe {name} opens with probe {other}'s name; make it an argument of {other} "
                            "or give it a name of its own")


def rule_unlisted_probe(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each probe folder that applications/probes/README.md does not name."""
    folder = repository.probes
    if not folder.is_dir():
        return
    readme = folder / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    for name in _probe_names(repository):
        if not re.search(rf"`{re.escape(name)}`|^#+\s+{re.escape(name)}\b", text, re.MULTILINE):
            yield Issue(readme, 1, f"probe {name} is not listed in {K_PROBES.as_posix()}/README.md")


K_CHECKER = Checker(
    description="Checks that probes live in one place, decide nothing and are listed.",
    collect=collect,
    rules={
        "rule_probe_outside_folder": rule_probe_outside_folder,
        "rule_probe_test_framework": rule_probe_test_framework,
        "rule_probe_in_test_runner": rule_probe_in_test_runner,
        "rule_probe_name_stem": rule_probe_name_stem,
        "rule_unlisted_probe": rule_unlisted_probe,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# check_test_mirror.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that tests/ mirrors the source: one test file per header or source file, no other folder,
and no test that includes a private header from a source/ folder.

Usage: python tools/testing/check_test_mirror.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import ignored_folders  # noqa: E402

K_DATA_FOLDERS = frozenset({"fixtures", "goldens", "Fixtures", "Goldens"})
K_UNITY_MARKER = "ProjectSettings"
K_UNITY_TESTS = "Tests"
K_UNITY_TEST_SUFFIX = "Tests.cs"
K_UNITY_UNTESTED = frozenset({"Tests", "Legacy", "Content"})
K_UNITY_NOT_PRODUCTS = frozenset({
    "ThirdParty", "LargeFiles", "Resources", "Plugins", "StreamingAssets", "Gizmos", "Editor",
})
K_PACKAGE_FOLDER = "package"
K_APPLICATIONS_FOLDER = "applications"
K_PRIVATE_FOLDER = "source"
K_CPP_TEST_SUFFIXES = frozenset({".cpp", ".h", ".hpp"})
K_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')


@dataclass(frozen=True, slots=True)
class Language:
    """Says which files of one language a test mirrors and how its test file is named."""

    folder: str
    suffixes: frozenset[str]
    test_prefix: str
    test_suffix: str
    skipped_names: frozenset[str]
    has_project_folder: bool

    def test_name(self, unit: Path) -> str:
        """Returns the name of the test file of a unit."""
        return f"{self.test_prefix}{unit.stem}{self.test_suffix}"

    def is_test(self, name: str) -> bool:
        """Says whether a file name has this language's test file shape."""
        return name.startswith(self.test_prefix) and name.endswith(self.test_suffix)


K_CPP = Language("include", frozenset({".h", ".hpp"}), "test_", ".cpp", frozenset(), True)
K_LANGUAGES = (
    K_CPP,
    Language("source", frozenset({".py"}), "test_", ".py", frozenset({"__init__.py", "__main__.py"}), True),
    Language("source", frozenset({".ts", ".tsx"}), "", ".test.ts", frozenset({"index.ts"}), False),
)


@dataclass(frozen=True, slots=True)
class TestMirror:
    """Holds the units of one repository and the tests/ tree that must mirror them."""

    root: Path
    tests: Path
    expected: dict[Path, Path]
    owners: frozenset[Path]
    ignored: frozenset[Path]

    def allowed_folders(self) -> frozenset[Path]:
        """Returns every folder below tests/ that a unit, a module or an application accounts for."""
        allowed: set[Path] = set()
        unit_folders = {unit_folder(test) for test in self.expected}
        for folder in {*self.owners, *(test.parent for test in self.expected), *unit_folders}:
            while folder != self.tests and self.tests in folder.parents:
                allowed.add(folder)
                folder = folder.parent
        return frozenset(allowed)

    def walk(self) -> Iterator[Path]:
        """Yields every path below tests/ outside data folders, the package test and ignored folders."""
        if not self.tests.is_dir():
            return
        pending = [self.tests]
        while pending:
            for entry in sorted(pending.pop().iterdir()):
                if entry.resolve() in self.ignored:
                    continue
                if entry.is_dir() and entry.name in K_DATA_FOLDERS:
                    continue
                if entry.is_dir() and entry == self.tests / K_PACKAGE_FOLDER:
                    continue
                yield entry
                if entry.is_dir():
                    pending.append(entry)


def unit_folder(test: Path) -> Path:
    """Returns the folder that stands for a unit's test file when its tests need more than one file."""
    name = test.name
    for language in K_LANGUAGES:
        if language.is_test(name) and name.endswith(language.test_suffix):
            return test.parent / name[: len(name) - len(language.test_suffix)]
    return test.parent / test.stem


def _aspect_files(folder: Path) -> list[Path]:
    """Returns the test files in a unit's folder."""
    if not folder.is_dir():
        return []
    return [
        path for path in sorted(folder.iterdir())
        if path.is_file() and any(language.is_test(path.name) for language in K_LANGUAGES)
    ]


def _units(folder: Path, language: Language) -> Iterator[Path]:
    """Yields every file of a language below a folder that a test mirrors."""
    if folder.is_dir():
        yield from (
            path for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix in language.suffixes
            and path.name not in language.skipped_names and not path.name.startswith("_")
        )


def _strip(relative: Path, language: Language, owner: Path) -> Path:
    """Returns a unit's path below its project or package folder and its module folder.

    The module folder is `<module>` or `<tier>/<module>`; the longer one that the path opens with is
    stripped.
    """
    parts = relative.parts
    if language.has_project_folder and len(parts) > 1:
        parts = parts[1:]
    for prefix in (owner.parts, owner.parts[-1:]):
        if prefix and len(parts) > len(prefix) and parts[:len(prefix)] == prefix:
            return Path(*parts[len(prefix):])
    return Path(*parts)


def _add_owner(expected: dict[Path, Path], tests: Path, owner: Path, folder: Path) -> None:
    """Adds the expected test of every unit in a module, application or single-module root."""
    for language in K_LANGUAGES:
        base = folder / language.folder
        for unit in _units(base, language):
            relative = _strip(unit.relative_to(base), language, owner)
            test = tests / owner / relative.parent / language.test_name(unit)
            expected[test] = unit


def discover(root: Path) -> TestMirror:
    """Returns the mirror of a repository's modules or single root, and of its applications."""
    expected: dict[Path, Path] = {}
    owners: set[Path] = set()
    tests = root / "tests"
    modules = root / "modules"
    if modules.is_dir():
        for tier in sorted(entry for entry in modules.iterdir() if entry.is_dir()):
            for module in sorted(entry for entry in tier.iterdir() if entry.is_dir()):
                owners.add(tests / tier.name / module.name)
                _add_owner(expected, tests, Path(tier.name) / module.name, module)
    else:
        _add_owner(expected, tests, Path(), root)
    applications = root / K_APPLICATIONS_FOLDER
    if applications.is_dir():
        for application in sorted(entry for entry in applications.iterdir() if entry.is_dir()):
            owner = Path(K_APPLICATIONS_FOLDER) / application.name
            owners.add(tests / owner)
            for unit in _units(application, K_CPP):
                relative = unit.relative_to(application)
                expected[tests / owner / relative.parent / K_CPP.test_name(unit)] = unit
    return TestMirror(root, tests, expected, frozenset(owners), ignored_folders(root))


def discover_unity(root: Path) -> list[TestMirror]:
    """Returns one mirror per product under Assets/: <Product>/Tests/ mirrors its C# files."""
    mirrors: list[TestMirror] = []
    assets = root / "Assets"
    ignored = ignored_folders(root)
    for product in sorted(entry for entry in assets.iterdir() if entry.is_dir()) if assets.is_dir() else []:
        if product.name in K_UNITY_NOT_PRODUCTS:
            continue
        tests = product / K_UNITY_TESTS
        expected = {
            tests / unit.relative_to(product).parent / f"{unit.stem}{K_UNITY_TEST_SUFFIX}": unit
            for unit in sorted(product.rglob("*.cs"))
            if unit.relative_to(product).parts[0] not in K_UNITY_UNTESTED
        }
        mirrors.append(TestMirror(root, tests, expected, frozenset(), ignored))
    return mirrors


def collect(paths: list[Path]) -> Iterable[TestMirror]:
    """Returns the mirrors of each given repository root, Unity or not."""
    mirrors: list[TestMirror] = []
    for path in paths:
        if path.is_dir():
            unity = (path / K_UNITY_MARKER).is_dir()
            mirrors.extend(discover_unity(path) if unity else [discover(path)])
    return mirrors


def rule_untested_unit(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each header or source file whose test file does not exist."""
    for test, header in sorted(mirror.expected.items(), key=lambda item: str(item[1])):
        if not test.is_file() and not _aspect_files(unit_folder(test)):
            shown = test.relative_to(mirror.root).as_posix()
            yield Issue(header, 1, f"no test file; expected {shown}")


def rule_orphan_test(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each test file that mirrors no header or source file."""
    for path in mirror.walk():
        is_test = path.name.endswith(K_UNITY_TEST_SUFFIX) or any(
            language.is_test(path.name) for language in K_LANGUAGES
        )
        in_unit_folder = any(unit_folder(test) == path.parent for test in mirror.expected)
        if path.is_file() and is_test and path not in mirror.expected and not in_unit_folder:
            yield Issue(path, 1, "mirrors no header or source file; rename it or move it to its unit's mirror")


def rule_mirror_folders(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each folder below tests/ that no tier, module, application or unit accounts for."""
    allowed = mirror.allowed_folders()
    for path in mirror.walk():
        if path.is_dir() and path not in allowed:
            yield Issue(path, 1, "folder has no counterpart in the source tree")


def rule_unit_file_and_folder(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each unit that has both a test file and a folder of aspect files."""
    for test in sorted(mirror.expected):
        if test.is_file() and _aspect_files(unit_folder(test)):
            yield Issue(test, 1, "the unit has a test file and a test folder; keep one")


def _private_headers(root: Path) -> frozenset[str]:
    """Returns the path below its source/ folder of every header private to a module or the root."""
    folders = [root / K_PRIVATE_FOLDER]
    modules = root / "modules"
    if modules.is_dir():
        folders += [module / K_PRIVATE_FOLDER for tier in modules.iterdir() if tier.is_dir()
                    for module in tier.iterdir() if module.is_dir()]
    return frozenset(
        header.relative_to(folder).as_posix()
        for folder in folders if folder.is_dir()
        for header in folder.rglob("*") if header.is_file() and header.suffix in K_CPP.suffixes
    )


def rule_private_include(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each include in a test that names a header under a source/ folder."""
    private = _private_headers(mirror.root)
    for path in mirror.walk():
        if not path.is_file() or path.suffix not in K_CPP_TEST_SUFFIXES:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            match = K_INCLUDE.match(line)
            if not match:
                continue
            quote, target = match.groups()
            through_source = f"/{K_PRIVATE_FOLDER}/" in f"/{target}"
            if not through_source and quote == '"' and (path.parent / target).is_file():
                continue
            if through_source or target in private:
                yield Issue(path, number, f"includes the private header {target}; test it through the public surface")


K_CHECKER = Checker(
    description="Checks that tests/ mirrors the source of the repository.",
    collect=collect,
    rules={
        "rule_untested_unit": rule_untested_unit,
        "rule_orphan_test": rule_orphan_test,
        "rule_mirror_folders": rule_mirror_folders,
        "rule_unit_file_and_folder": rule_unit_file_and_folder,
        "rule_private_include": rule_private_include,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# check_test_mirror.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks that tests/ mirrors the source: one test file per header or source file, no other folder,
and no test that includes a private header from a source/ folder.

The rules read values only; collect() reads the disk into Entries.

Usage: python tools/testing/check_test_mirror.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import ignored_folders  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402

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
K_TREE_FOLDERS = ("tests", "modules", K_APPLICATIONS_FOLDER, "include", K_PRIVATE_FOLDER, "Assets",
                  K_UNITY_MARKER)
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
    entry_names: frozenset[str] = frozenset()

    def test_name(self, unit: Path) -> str:
        """Returns the name of the test file of a unit; a package entry such as `__main__` drops its underscores."""
        return f"{self.test_prefix}{unit.stem.strip('_')}{self.test_suffix}"

    def is_test(self, name: str) -> bool:
        """Says whether a file name has this language's test file shape."""
        return name.startswith(self.test_prefix) and name.endswith(self.test_suffix)


K_CPP = Language("include", frozenset({".h", ".hpp"}), "test_", ".cpp", frozenset(), True)
K_LANGUAGES = (
    K_CPP,
    Language("source", frozenset({".py"}), "test_", ".py", frozenset({"__init__.py"}), True,
             frozenset({"__main__.py"})),
    Language("source", frozenset({".ts", ".tsx"}), "", ".test.ts", frozenset({"index.ts"}), False),
)


@dataclass(frozen=True, slots=True)
class Entries:
    """Holds what a repository holds below its test, source and product folders, as values."""

    root: Path
    files: frozenset[Path]
    folders: frozenset[Path]
    ignored: frozenset[Path]
    sources: tuple[SourceFile, ...]


@dataclass(frozen=True, slots=True)
class TestMirror:
    """Holds the units of one repository and the tests/ tree that must mirror them, as values."""

    root: Path
    tests: Path
    expected: dict[Path, Path]
    owners: frozenset[Path]
    files: frozenset[Path]
    folders: frozenset[Path]
    walked: tuple[Path, ...]
    private: frozenset[str]
    sources: tuple[SourceFile, ...]

    def allowed_folders(self) -> frozenset[Path]:
        """Returns every folder below tests/ that a unit, a module or an application accounts for."""
        allowed: set[Path] = set()
        unit_folders = {unit_folder(test) for test in self.expected}
        for folder in {*self.owners, *(test.parent for test in self.expected), *unit_folders}:
            while folder != self.tests and self.tests in folder.parents:
                allowed.add(folder)
                folder = folder.parent
        return frozenset(allowed)


def unit_folder(test: Path) -> Path:
    """Returns the folder that stands for a unit's test file when its tests need more than one file."""
    name = test.name
    for language in K_LANGUAGES:
        if language.is_test(name) and name.endswith(language.test_suffix):
            return test.parent / name[: len(name) - len(language.test_suffix)]
    return test.parent / test.stem


def _subfolders(folders: frozenset[Path], parent: Path) -> list[Path]:
    """Returns the folders directly below a parent, sorted."""
    return sorted(folder for folder in folders if folder.parent == parent)


def _aspect_files(files: frozenset[Path], folder: Path) -> list[Path]:
    """Returns the test files that sit directly in a unit's folder."""
    return [path for path in files if path.parent == folder and any(language.is_test(path.name)
                                                                   for language in K_LANGUAGES)]


def _units(files: frozenset[Path], base: Path, language: Language) -> list[Path]:
    """Returns, sorted, every file of a language below a folder that a test mirrors."""
    return sorted(
        path for path in files
        if path.is_relative_to(base) and path.suffix in language.suffixes
        and path.name not in language.skipped_names
        and (not path.name.startswith("_") or path.name in language.entry_names)
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


def _add_owner(expected: dict[Path, Path], tests: Path, owner: Path, folder: Path,
               files: frozenset[Path]) -> None:
    """Adds the expected test of every unit in a module, application or single-module root."""
    for language in K_LANGUAGES:
        base = folder / language.folder
        for unit in _units(files, base, language):
            relative = _strip(unit.relative_to(base), language, owner)
            test = tests / owner / relative.parent / language.test_name(unit)
            expected[test] = unit


def _private_headers(entries: Entries) -> frozenset[str]:
    """Returns the path below its source/ folder of every header private to a module or the root."""
    modules = entries.root / "modules"
    candidates = [entries.root / K_PRIVATE_FOLDER] + [
        module / K_PRIVATE_FOLDER
        for tier in _subfolders(entries.folders, modules)
        for module in _subfolders(entries.folders, tier)
    ]
    sources = [folder for folder in candidates if folder in entries.folders]
    return frozenset(
        path.relative_to(folder).as_posix()
        for folder in sources for path in entries.files
        if path.is_relative_to(folder) and path.suffix in K_CPP.suffixes
    )


def _walked(tests: Path, entries: Entries) -> tuple[Path, ...]:
    """Returns the entries below tests/ that the rules judge, in the order a walk visits them.

    A data folder, the package test and an entry git ignores are left out, with what sits below them.
    """
    if tests not in entries.folders:
        return ()
    children: dict[Path, list[Path]] = defaultdict(list)
    for path in entries.files | entries.folders:
        children[path.parent].append(path)
    walked: list[Path] = []
    pending = [tests]
    while pending:
        for entry in sorted(children.get(pending.pop(), [])):
            if entry in entries.ignored:
                continue
            is_folder = entry in entries.folders
            if is_folder and (entry.name in K_DATA_FOLDERS or entry == tests / K_PACKAGE_FOLDER):
                continue
            walked.append(entry)
            if is_folder:
                pending.append(entry)
    return tuple(walked)


def _mirror(entries: Entries, tests: Path, expected: dict[Path, Path], owners: frozenset[Path]) -> TestMirror:
    """Returns the mirror of one tests/ tree over the entries of its repository."""
    return TestMirror(
        root=entries.root, tests=tests, expected=expected, owners=owners,
        files=entries.files, folders=entries.folders, walked=_walked(tests, entries),
        private=_private_headers(entries), sources=entries.sources,
    )


def _layered(entries: Entries) -> TestMirror:
    """Returns the mirror of a repository laid out as modules, applications or one root."""
    root = entries.root
    tests = root / "tests"
    expected: dict[Path, Path] = {}
    owners: set[Path] = set()
    modules = root / "modules"
    if modules in entries.folders:
        for tier in _subfolders(entries.folders, modules):
            for module in _subfolders(entries.folders, tier):
                owners.add(tests / tier.name / module.name)
                _add_owner(expected, tests, Path(tier.name) / module.name, module, entries.files)
    else:
        _add_owner(expected, tests, Path(), root, entries.files)
    for application in _subfolders(entries.folders, root / K_APPLICATIONS_FOLDER):
        owner = Path(K_APPLICATIONS_FOLDER) / application.name
        owners.add(tests / owner)
        for unit in _units(entries.files, application, K_CPP):
            relative = unit.relative_to(application)
            expected[tests / owner / relative.parent / K_CPP.test_name(unit)] = unit
    return _mirror(entries, tests, expected, frozenset(owners))


def _unity(entries: Entries) -> list[TestMirror]:
    """Returns one mirror per product under Assets/: <Product>/Tests/ mirrors its C# files."""
    mirrors: list[TestMirror] = []
    for product in _subfolders(entries.folders, entries.root / "Assets"):
        if product.name in K_UNITY_NOT_PRODUCTS:
            continue
        tests = product / K_UNITY_TESTS
        expected = {
            tests / unit.relative_to(product).parent / f"{unit.stem}{K_UNITY_TEST_SUFFIX}": unit
            for unit in sorted(entries.files)
            if unit.is_relative_to(product) and unit.suffix == ".cs"
            and unit.relative_to(product).parts[0] not in K_UNITY_UNTESTED
        }
        mirrors.append(_mirror(entries, tests, expected, frozenset()))
    return mirrors


def mirrors_of(entries: Entries) -> list[TestMirror]:
    """Returns one mirror per product of a Unity project, or the one mirror of any other repository."""
    if entries.root / K_UNITY_MARKER in entries.folders:
        return _unity(entries)
    return [_layered(entries)]


def _every(path: Path) -> bool:
    """Returns True for every path; the text of each C++ test file is read."""
    return True


def _entries_of(root: Path) -> Entries:
    """Reads the files and folders a mirror is built from, and the text of each C++ file below tests/."""
    found: set[Path] = set()
    for name in K_TREE_FOLDERS:
        top = root / name
        if top.is_dir():
            found.add(top)
            found.update(top.rglob("*"))
    ignored_under = ignored_folders(root)
    files = frozenset(path for path in found if path.is_file())
    folders = frozenset(path for path in found if path.is_dir())
    ignored = frozenset(path for path in found if path.resolve() in ignored_under) if ignored_under else frozenset()
    tests = root / "tests"
    headers = sorted(path for path in files if path.is_relative_to(tests) and path.suffix in K_CPP_TEST_SUFFIXES)
    return Entries(root, files, folders, ignored, read_files(headers, _every))


def collect(paths: list[Path]) -> Iterable[TestMirror]:
    """Returns the mirrors of each given repository root, Unity or not."""
    mirrors: list[TestMirror] = []
    for path in paths:
        if path.is_dir():
            mirrors.extend(mirrors_of(_entries_of(path)))
    return mirrors


def rule_untested_unit(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each header or source file whose test file does not exist."""
    for test, header in sorted(mirror.expected.items(), key=lambda item: str(item[1])):
        if test not in mirror.files and not _aspect_files(mirror.files, unit_folder(test)):
            shown = test.relative_to(mirror.root).as_posix()
            yield Issue(header, 1, f"no test file; expected {shown}")


def rule_orphan_test(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each test file that mirrors no header or source file."""
    unit_folders = {unit_folder(test) for test in mirror.expected}
    for path in mirror.walked:
        is_test = path.name.endswith(K_UNITY_TEST_SUFFIX) or any(
            language.is_test(path.name) for language in K_LANGUAGES
        )
        if path in mirror.files and is_test and path not in mirror.expected and path.parent not in unit_folders:
            yield Issue(path, 1, "mirrors no header or source file; rename it or move it to its unit's mirror")


def rule_mirror_folders(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each folder below tests/ that no tier, module, application or unit accounts for."""
    allowed = mirror.allowed_folders()
    for path in mirror.walked:
        if path in mirror.folders and path not in allowed:
            yield Issue(path, 1, "folder has no counterpart in the source tree")


def rule_unit_file_and_folder(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each unit that has both a test file and a folder of aspect files."""
    for test in sorted(mirror.expected):
        if test in mirror.files and _aspect_files(mirror.files, unit_folder(test)):
            yield Issue(test, 1, "the unit has a test file and a test folder; keep one")


def rule_private_include(mirror: TestMirror) -> Iterator[Issue]:
    """Yields an issue for each include in a test that names a header under a source/ folder."""
    walked = frozenset(mirror.walked)
    for source in mirror.sources:
        if source.path not in walked:
            continue
        for number, line in enumerate(source.text.splitlines(), start=1):
            match = K_INCLUDE.match(line)
            if not match:
                continue
            quote, target = match.groups()
            through_source = f"/{K_PRIVATE_FOLDER}/" in f"/{target}"
            beside = Path(os.path.normpath(source.path.parent / target)) in mirror.files
            if not through_source and quote == '"' and beside:
                continue
            if through_source or target in mirror.private:
                yield Issue(source.path, number,
                            f"includes the private header {target}; test it through the public surface")


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

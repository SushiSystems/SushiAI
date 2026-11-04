# check_agnostic.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that the doctrine names no product, company tool, identity or machine path.

The folders it reads are K_AGNOSTIC_PATHS in tools/settings.py, below the repository root; a
repository that sets none is never checked. The terms come from the file K_AGNOSTIC_TERMS names.

Usage: python tools/layering/check_agnostic.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402
import settings  # noqa: E402

K_AGNOSTIC_PATHS: tuple[str, ...] = getattr(settings, "K_AGNOSTIC_PATHS", ())
K_AGNOSTIC_TERMS: str = getattr(settings, "K_AGNOSTIC_TERMS", "")
K_TECHNOLOGY_EXEMPT: tuple[str, ...] = getattr(settings, "K_TECHNOLOGY_EXEMPT", ())
K_TEXT_SUFFIXES = frozenset({".md", ".py", ".toml", ".txt", ".json", ".yml", ".yaml", ".sh"})
K_HASH_SUFFIXES = frozenset({".py", ".sh"})
K_CONFIG_FILES = frozenset({"tools/settings.py"})
K_CATEGORIES = ("product", "tool", "identity", "technology")
K_ALLOW = "allow"
K_TERM_LINE = re.compile(r"^(\w+):\s*(.+?)\s*$")
K_MACHINE_PATH = re.compile(r"(?<![\w/])[A-Za-z]:[\\/]\w+[\\/]|(?<!\w)/(?:Users|home)/\w")
K_ALLOW_MARK = "agnostic: allow"


@dataclass(frozen=True, slots=True)
class Terms:
    """Holds the patterns of one terms file, by category, and the matches it allows."""

    path: Path
    patterns: dict[str, tuple[re.Pattern[str], ...]]
    allowed: tuple[re.Pattern[str], ...]
    error: str


@dataclass(frozen=True, slots=True)
class Scope:
    """Holds the text files below the checked folders of one repository, its terms and its exempt folders."""

    root: Path
    files: tuple[SourceFile, ...]
    terms: Terms | None
    technology_exempt: tuple[str, ...] = ()


def parse_terms(path: Path, text: str | None) -> Terms:
    """Returns the terms a file's text holds; None stands for a missing file, and the error names the first bad line."""
    if text is None:
        return Terms(path, {}, (), "no such terms file")
    patterns: dict[str, list[re.Pattern[str]]] = {name: [] for name in K_CATEGORIES}
    allowed: list[re.Pattern[str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parsed = K_TERM_LINE.match(line)
        if parsed is None or parsed.group(1) not in (*K_CATEGORIES, K_ALLOW):
            return Terms(path, {}, (), f"{path.name}:{number}: not '<category>: <pattern>'")
        try:
            pattern = re.compile(parsed.group(2))
        except re.error as error:
            return Terms(path, {}, (), f"{path.name}:{number}: {error}")
        (allowed if parsed.group(1) == K_ALLOW else patterns[parsed.group(1)]).append(pattern)
    return Terms(path, {name: tuple(found) for name, found in patterns.items()}, tuple(allowed), "")


def is_checked_file(relative: str, folders: tuple[str, ...]) -> bool:
    """Returns whether a root-relative path is a text file below one of the folders and not a settings file."""
    path = PurePosixPath(relative)
    if relative in K_CONFIG_FILES or path.suffix not in K_TEXT_SUFFIXES:
        return False
    return any(
        path.parts[: len(parts)] == parts and len(path.parts) > len(parts)
        for parts in (PurePosixPath(folder).parts for folder in folders)
    )


def _is_exempt(scope: Scope, path: Path, category: str) -> bool:
    """Returns whether a file sits below a folder that settings exempt from a technology check."""
    if category != "technology":
        return False
    relative = path.relative_to(scope.root).as_posix()
    return any(relative.startswith(entry.rstrip("/") + "/") for entry in scope.technology_exempt)


def _lines(source: SourceFile) -> Iterator[tuple[int, str]]:
    """Yields the number and text of each line of a file, leaving out a leading license header of `#`
    lines and each line that carries the mark `agnostic: allow`."""
    in_header = source.path.suffix in K_HASH_SUFFIXES
    for number, line in enumerate(source.text.splitlines(), 1):
        in_header = in_header and line.startswith("#")
        if not in_header and K_ALLOW_MARK not in line:
            yield number, line


def _term_issues(scope: Scope, category: str) -> Iterator[Issue]:
    """Yields an issue for each match of a category that the allowed patterns do not cover."""
    if scope.terms is None or scope.terms.error:
        return
    for source in scope.files:
        if _is_exempt(scope, source.path, category):
            continue
        for number, line in _lines(source):
            for pattern in scope.terms.patterns[category]:
                for match in pattern.finditer(line):
                    if not any(allow.fullmatch(match.group()) for allow in scope.terms.allowed):
                        yield Issue(source.path, number, f"names {match.group()!r}")


def rule_product_name(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each product or library name the terms list."""
    return _term_issues(scope, "product")


def rule_company_tool(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each company tool, command or file the terms list."""
    return _term_issues(scope, "tool")


def rule_technology_name(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each engine, vendor or device name in a skill that is not its own."""
    return _term_issues(scope, "technology")


def rule_identity(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each person, company, address or host the terms list."""
    return _term_issues(scope, "identity")


def rule_machine_path(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each drive path or user folder."""
    for source in scope.files:
        for number, line in _lines(source):
            for match in K_MACHINE_PATH.finditer(line):
                yield Issue(source.path, number, f"names the machine path {match.group()!r}")


def rule_terms_file(scope: Scope) -> Iterator[Issue]:
    """Yields an issue when the settings name a terms file that is missing or unreadable."""
    if scope.terms is not None and scope.terms.error:
        yield Issue(scope.terms.path, 1, scope.terms.error)


K_RULES = {
    "rule_product_name": rule_product_name,
    "rule_company_tool": rule_company_tool,
    "rule_identity": rule_identity,
    "rule_technology_name": rule_technology_name,
    "rule_machine_path": rule_machine_path,
    "rule_terms_file": rule_terms_file,
}


def _read_files_below(root: Path) -> tuple[SourceFile, ...]:
    """Reads the text files below each checked folder that git does not ignore and that are not settings."""
    candidates = [
        path
        for name in K_AGNOSTIC_PATHS if (root / name).is_dir()
        for path in sorted((root / name).rglob("*"))
        if path.is_file() and is_checked_file(path.relative_to(root).as_posix(), (name,))
    ]
    return read_files(drop_ignored(root, candidates), lambda _: True)


def _read_terms(root: Path) -> Terms | None:
    """Reads the terms file the settings name, or returns None when they name none."""
    if not K_AGNOSTIC_TERMS:
        return None
    path = root / K_AGNOSTIC_TERMS
    found = read_files([path], Path.is_file)
    return parse_terms(path, found[0].text if found else None)


def _read_scope(root: Path) -> Scope:
    """Reads one repository's checked files and terms into a scope; the only reader of the disk here."""
    return Scope(root, _read_files_below(root), _read_terms(root), K_TECHNOLOGY_EXEMPT)


def collect(paths: list[Path]) -> Iterable[Scope]:
    """Returns the scope of each given repository root."""
    return [_read_scope(path) for path in paths if path.is_dir()]


K_CHECKER = Checker(
    description="Check that the doctrine names no product, company tool, identity or machine path.",
    collect=collect,
    rules=K_RULES,
)


if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

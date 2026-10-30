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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
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
    """Holds the text files below the checked folders of one repository, and its terms."""

    root: Path
    files: tuple[Path, ...]
    terms: Terms | None


def _parse_terms(path: Path) -> Terms:
    """Returns the terms a file holds; the error names the first line that cannot be read."""
    patterns: dict[str, list[re.Pattern[str]]] = {name: [] for name in K_CATEGORIES}
    allowed: list[re.Pattern[str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def _load_terms(root: Path) -> Terms | None:
    """Returns the terms file the settings name, or None when they name none."""
    if not K_AGNOSTIC_TERMS:
        return None
    path = root / K_AGNOSTIC_TERMS
    if not path.is_file():
        return Terms(path, {}, (), "no such terms file")
    return _parse_terms(path)


def _files_below(root: Path) -> tuple[Path, ...]:
    """Returns the text files git does not ignore below each checked folder."""
    found: list[Path] = []
    for name in K_AGNOSTIC_PATHS:
        folder = root / name
        if folder.is_dir():
            found.extend(path for path in sorted(folder.rglob("*"))
                         if path.is_file() and path.suffix in K_TEXT_SUFFIXES
                         and path.relative_to(root).as_posix() not in K_CONFIG_FILES)
    return tuple(drop_ignored(root, found))


def collect(paths: list[Path]) -> Iterable[Scope]:
    """Returns the scope of each given repository root."""
    return [Scope(path, _files_below(path), _load_terms(path))
            for path in paths if path.is_dir()]


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yields the number and text of each line of a file, leaving out a leading license header of `#`
    lines and each line that carries the mark `agnostic: allow`."""
    in_header = path.suffix in K_HASH_SUFFIXES
    text = path.read_text(encoding="utf-8", errors="ignore")
    for number, line in enumerate(text.splitlines(), 1):
        in_header = in_header and line.startswith("#")
        if not in_header and K_ALLOW_MARK not in line:
            yield number, line


def _is_exempt(scope: Scope, path: Path, category: str) -> bool:
    """Returns whether a file sits below a folder that settings exempt from a technology check."""
    if category != "technology":
        return False
    relative = path.relative_to(scope.root).as_posix()
    return any(relative.startswith(entry.rstrip("/") + "/") for entry in K_TECHNOLOGY_EXEMPT)


def _term_issues(scope: Scope, category: str) -> Iterator[Issue]:
    """Yields an issue for each match of a category that the allowed patterns do not cover."""
    if scope.terms is None or scope.terms.error:
        return
    for path in scope.files:
        if _is_exempt(scope, path, category):
            continue
        for number, line in _lines(path):
            for pattern in scope.terms.patterns[category]:
                for match in pattern.finditer(line):
                    if not any(allow.fullmatch(match.group()) for allow in scope.terms.allowed):
                        yield Issue(path, number, f"names {match.group()!r}")


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
    for path in scope.files:
        for number, line in _lines(path):
            for match in K_MACHINE_PATH.finditer(line):
                yield Issue(path, number, f"names the machine path {match.group()!r}")


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

K_CHECKER = Checker(
    description="Check that the doctrine names no product, company tool, identity or machine path.",
    collect=collect,
    rules=K_RULES,
)


if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

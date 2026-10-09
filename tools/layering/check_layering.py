# check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every include points down the tier order and never across two domains.

The tiers are those of docs/design/DOMAIN_MODULES.md.
Usage: python tools/layering/check_layering.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import posixpath
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_TIER_ORDER: tuple[str, ...] = ("core", "layers", "domains", "programs")
K_DOMAIN_TIER = "domains"
K_MODULE_TIERS: dict[str, str] = {
    "core": "core",
    "graph": "core",
    "autograd": "core",
    "optim": "core",
    "config": "core",
    "eager": "core",
    "data": "core",
    "nn": "layers",
    "train": "layers",
    "io": "layers",
    "vision": "domains",
}
K_LIBRARY_ROOTS: tuple[str, ...] = ("include/SushiAI", "src")
K_SCOPED_ROOTS: dict[str, str] = {
    "tests/common": "layers",
    "tests/functional": "layers",
    "tests/vision": "domains",
    "apps": "programs",
    "examples": "programs",
}
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".inl"})
K_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")')
K_MODULE_HEADER = re.compile(r"SushiAI/([^/]+)/.+")


@dataclass(frozen=True, slots=True)
class Layout:
    """Holds the declared tiers of one repository and the folders they cover."""

    tier_order: tuple[str, ...]
    domain_tier: str
    module_tiers: Mapping[str, str]
    library_roots: tuple[str, ...]
    scoped_roots: Mapping[str, str]

    def rank(self, tier: str) -> int | None:
        """Returns a tier's position in the order, lowest first, or None when undeclared."""
        return self.tier_order.index(tier) if tier in self.tier_order else None


K_LAYOUT = Layout(K_TIER_ORDER, K_DOMAIN_TIER, K_MODULE_TIERS, K_LIBRARY_ROOTS, K_SCOPED_ROOTS)


@dataclass(frozen=True, slots=True)
class Scope:
    """Holds one module or one scoped root: its name, its tier and the folders it owns.

    A scope of the domain tier is named after its domain.
    """

    label: str
    name: str
    tier: str
    folders: tuple[Path, ...]
    is_module: bool


@dataclass(frozen=True, slots=True)
class Include:
    """Holds one include directive of a source file."""

    path: Path
    line: int
    target: str
    is_quoted: bool

    @property
    def module(self) -> str | None:
        """Returns the module an angle include names, or None for any other include."""
        match = None if self.is_quoted else K_MODULE_HEADER.fullmatch(self.target)
        return match.group(1) if match else None


def _read_includes(folder: Path) -> Iterator[Include]:
    """Yields every include directive of every source file below a folder."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix not in K_SOURCE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if match := K_INCLUDE.match(line):
                yield Include(path, number, match.group(1) or match.group(2), not match.group(1))


class SourceTree:
    """Holds one repository's module folders, its scoped roots and their includes."""

    def __init__(self, root: Path, layout: Layout) -> None:
        """Remembers the root and the layout; the tree is read when a rule first asks."""
        self.root = root
        self.layout = layout

    @cached_property
    def module_folders(self) -> dict[str, tuple[Path, ...]]:
        """Returns the folders directly under the library roots, by module name."""
        found: dict[str, list[Path]] = {}
        for library_root in self.layout.library_roots:
            base = self.root / library_root
            if base.is_dir():
                for folder in sorted(entry for entry in base.iterdir() if entry.is_dir()):
                    found.setdefault(folder.name, []).append(folder)
        return {name: tuple(folders) for name, folders in found.items()}

    @cached_property
    def modules(self) -> dict[str, Scope]:
        """Returns the scope of every declared module that has a folder, by module name."""
        return {
            name: Scope(name, name, self.layout.module_tiers[name], folders, True)
            for name, folders in self.module_folders.items()
            if name in self.layout.module_tiers
        }

    @cached_property
    def scopes(self) -> tuple[Scope, ...]:
        """Returns every module scope, then every scoped root that exists."""
        scoped = [
            Scope(label, posixpath.basename(label), tier, (self.root / label,), False)
            for label, tier in self.layout.scoped_roots.items()
            if (self.root / label).is_dir()
        ]
        return (*self.modules.values(), *scoped)

    @cached_property
    def includes(self) -> dict[str, tuple[Include, ...]]:
        """Returns the include directives of each scope, by scope label."""
        return {
            scope.label: tuple(
                include for folder in scope.folders for include in _read_includes(folder)
            )
            for scope in self.scopes
        }

    def module_includes(self, scope: Scope) -> Iterator[tuple[Include, Scope]]:
        """Yields each include of a scope that names a declared module, with that module."""
        for include in self.includes[scope.label]:
            target = self.modules.get(include.module or "")
            if target is not None:
                yield include, target


def collect(paths: list[Path]) -> Iterable[SourceTree]:
    """Returns one source tree per path given, read against the declared layout."""
    return [SourceTree(path if path.is_dir() else path.parent, K_LAYOUT) for path in paths]


def rule_declared_tiers(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue where the declared tables and the folders of the tree disagree."""
    layout = tree.layout
    library = tree.root / layout.library_roots[0]
    for name, folders in sorted(tree.module_folders.items()):
        if name not in layout.module_tiers:
            yield Issue(folders[0], 1, f"module folder '{name}' is not in K_MODULE_TIERS")
    for name, tier in layout.module_tiers.items():
        if name not in tree.module_folders:
            yield Issue(library / name, 1, f"module '{name}' is declared and has no folder")
        if layout.rank(tier) is None:
            yield Issue(library / name, 1, f"tier '{tier}' of '{name}' is not in K_TIER_ORDER")
    domains = {name for name, tier in layout.module_tiers.items() if tier == layout.domain_tier}
    for label, tier in layout.scoped_roots.items():
        if not (tree.root / label).is_dir():
            yield Issue(tree.root / label, 1, f"scoped root '{label}' is declared and missing")
        if layout.rank(tier) is None:
            yield Issue(tree.root / label, 1, f"tier '{tier}' of '{label}' is not in K_TIER_ORDER")
        if tier == layout.domain_tier and posixpath.basename(label) not in domains:
            yield Issue(tree.root / label, 1, f"scoped root '{label}' is named after no domain")


def rule_upward_includes(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that ranks above the includer's tier."""
    for scope in tree.scopes:
        own = tree.layout.rank(scope.tier)
        for include, target in tree.module_includes(scope):
            other = tree.layout.rank(target.tier)
            if own is not None and other is not None and other > own:
                reached = f"{target.name} ({target.tier})"
                message = f"{scope.label} ({scope.tier}) includes up into {reached}"
                yield Issue(include.path, include.line, message)


def rule_cross_domain_includes(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each include by which a file of one domain names another domain."""
    domain_tier = tree.layout.domain_tier
    for scope in tree.scopes:
        if scope.tier != domain_tier:
            continue
        for include, target in tree.module_includes(scope):
            if target.tier == domain_tier and target.name != scope.name:
                message = f"{scope.label} of domain {scope.name} includes domain {target.name}"
                yield Issue(include.path, include.line, message)


def rule_private_reach(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each quoted include of a module file that leaves its module's folder."""
    for scope in tree.modules.values():
        for include in tree.includes[scope.label]:
            if not include.is_quoted:
                continue
            home = next(folder for folder in scope.folders if folder in include.path.parents)
            reached = Path(posixpath.normpath((include.path.parent / include.target).as_posix()))
            if home != reached and home not in reached.parents:
                message = f'"{include.target}" leaves the folder of module {scope.name}'
                yield Issue(include.path, include.line, message)


K_CHECKER = Checker(
    description="Checks includes against the declared tiers and the domain boundaries.",
    collect=collect,
    rules={
        "rule_declared_tiers": rule_declared_tiers,
        "rule_upward_includes": rule_upward_includes,
        "rule_cross_domain_includes": rule_cross_domain_includes,
        "rule_private_reach": rule_private_reach,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that modules under modules/<tier>/<module>/ include only down the tier order.

The tier table is the one cmake/SushiAILayers.cmake holds for the build.
Usage: python tools/layering/check_layering.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import posixpath
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_TIER_ORDER: tuple[str, ...] = ("core", "layers", "vision")
K_DOMAIN_TIERS: tuple[str, ...] = ("vision",)
K_MODULE_TIERS: dict[str, str] = {
    "core": "core",
    "version": "core",
    "graph": "core",
    "config": "core",
    "autograd": "core",
    "optim": "core",
    "data": "core",
    "eager": "core",
    "nn": "layers",
    "io": "layers",
    "train": "layers",
    "umbrella": "layers",
    "vision/imaging": "vision",
    "vision/detect": "vision",
    "vision/blocks": "vision",
    "vision/datasets": "vision",
    "vision/registration": "vision",
    "vision/umbrella": "vision",
}
K_SCOPED_ROOTS: dict[str, str | None] = {
    "tests/common": "layers",
    "tests": None,
    "applications": None,
}
K_MODULES_ROOT = "modules"
K_PUBLIC_FOLDER = "include"
K_PRIVATE_FOLDERS: tuple[str, ...] = ("source", "tests")
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".inl"})
K_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")')


@dataclass(frozen=True, slots=True)
class Layout:
    """Holds the declared tiers of one repository and the roots outside the modules.

    A scoped root maps to the highest tier its files may include, or to None for any tier.
    """

    tier_order: tuple[str, ...]
    domain_tiers: tuple[str, ...]
    module_tiers: Mapping[str, str]
    scoped_roots: Mapping[str, str | None]

    def rank(self, tier: str) -> int | None:
        """Returns a tier's rank, lowest first and equal for every domain, or None."""
        if tier not in self.tier_order:
            return None
        if tier in self.domain_tiers:
            return min(self.tier_order.index(domain) for domain in self.domain_tiers)
        return self.tier_order.index(tier)

    def key(self, tier: str, name: str) -> str:
        """Returns a module's key: `<tier>/<name>` in a domain tier, the name elsewhere."""
        return f"{tier}/{name}" if tier in self.domain_tiers else name


K_LAYOUT = Layout(K_TIER_ORDER, K_DOMAIN_TIERS, K_MODULE_TIERS, K_SCOPED_ROOTS)


@dataclass(frozen=True, slots=True)
class Module:
    """Holds one module folder, the tier folder it sits under and its key."""

    key: str
    tier: str
    name: str
    folder: Path


@dataclass(frozen=True, slots=True)
class Scope:
    """Holds one module or one scoped root: its folder and the tier that bounds its includes.

    The nested folders belong to another scope and are not read as part of this one.
    """

    label: str
    tier: str | None
    folder: Path
    nested: tuple[Path, ...]
    module: Module | None


@dataclass(frozen=True, slots=True)
class Include:
    """Holds one include directive of a source file."""

    path: Path
    line: int
    target: str
    is_quoted: bool


def _read_includes(scope: Scope) -> Iterator[Include]:
    """Yields every include directive of every source file a scope owns."""
    for path in sorted(scope.folder.rglob("*")):
        if not path.is_file() or path.suffix not in K_SOURCE_SUFFIXES:
            continue
        if any(folder in path.parents for folder in scope.nested):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if match := K_INCLUDE.match(line):
                yield Include(path, number, match.group(1) or match.group(2), not match.group(1))


class SourceTree:
    """Holds one repository's modules, its scoped roots and their includes."""

    def __init__(self, root: Path, layout: Layout) -> None:
        """Remembers the root and the layout; the tree is read when a rule first asks."""
        self.root = root
        self.layout = layout

    @cached_property
    def discovered(self) -> tuple[Module, ...]:
        """Returns every folder two levels below the modules root as a module."""
        base = self.root / K_MODULES_ROOT
        if not base.is_dir():
            return ()
        return tuple(
            Module(self.layout.key(tier.name, folder.name), tier.name, folder.name, folder)
            for tier in sorted(entry for entry in base.iterdir() if entry.is_dir())
            for folder in sorted(entry for entry in tier.iterdir() if entry.is_dir())
        )

    @cached_property
    def modules(self) -> dict[str, Module]:
        """Returns the discovered modules by key; the first folder holds a repeated key."""
        found: dict[str, Module] = {}
        for module in self.discovered:
            found.setdefault(module.key, module)
        return found

    @cached_property
    def scopes(self) -> tuple[Scope, ...]:
        """Returns every module scope, then every scoped root that exists."""
        roots = {
            label: self.root / label
            for label in self.layout.scoped_roots
            if (self.root / label).is_dir()
        }
        scoped = [
            Scope(
                label,
                self.layout.scoped_roots[label],
                folder,
                tuple(other for other in roots.values() if folder in other.parents),
                None,
            )
            for label, folder in roots.items()
        ]
        modules = [
            Scope(module.key, module.tier, module.folder, (), module)
            for module in self.modules.values()
        ]
        return (*modules, *scoped)

    @cached_property
    def includes(self) -> dict[str, tuple[Include, ...]]:
        """Returns the include directives of each scope, by scope label."""
        return {scope.label: tuple(_read_includes(scope)) for scope in self.scopes}

    @cached_property
    def owners(self) -> dict[str, Module | None]:
        """Returns the module of each public header and header folder, None when shared.

        The paths are those an include names, relative to a module's include/ folder.
        """
        found: dict[str, Module | None] = {}
        for module in self.modules.values():
            public = module.folder / K_PUBLIC_FOLDER
            for path in public.rglob("*") if public.is_dir() else ():
                name = path.relative_to(public).as_posix()
                found[name] = module if found.get(name, module) is module else None
        return found

    def resolve(self, target: str) -> Module | None:
        """Returns the module whose public headers an include target names, or None."""
        path = PurePosixPath(posixpath.normpath(target))
        for candidate in (path, *path.parents):
            if owner := self.owners.get(candidate.as_posix()):
                return owner
        return None

    def module_includes(self, scope: Scope) -> Iterator[tuple[Include, Module]]:
        """Yields each include of a scope that names another module, with that module."""
        for include in self.includes[scope.label]:
            target = self.resolve(include.target)
            if target is not None and target is not scope.module:
                yield include, target

    def private_owners(self, target: str) -> Iterator[tuple[Module, str]]:
        """Yields each module whose source/ or tests/ a path names, with that folder's name."""
        parts = PurePosixPath(posixpath.normpath(target)).parts
        tiers = {module.tier for module in self.discovered}
        for index, part in enumerate(parts):
            if index == 0 or part not in K_PRIVATE_FOLDERS:
                continue
            above = parts[index - 2] if index >= 2 else None
            for module in self.modules.values():
                if module.name != parts[index - 1]:
                    continue
                if above not in tiers or above == module.tier:
                    yield module, part


def collect(paths: list[Path]) -> Iterable[SourceTree]:
    """Returns one source tree per path given, read against the declared layout."""
    return [SourceTree(path if path.is_dir() else path.parent, K_LAYOUT) for path in paths]


def rule_declared_tiers(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue where the declared tables and the folders of the tree disagree."""
    layout = tree.layout
    base = tree.root / K_MODULES_ROOT
    for domain in layout.domain_tiers:
        if domain not in layout.tier_order:
            yield Issue(base / domain, 1, f"domain tier '{domain}' is not in K_TIER_ORDER")
    for module in tree.discovered:
        declared = layout.module_tiers.get(module.key)
        if tree.modules[module.key] is not module:
            yield Issue(module.folder, 1, f"module key '{module.key}' names a second folder")
        elif declared is None:
            yield Issue(module.folder, 1, f"module folder '{module.key}' is not in K_MODULE_TIERS")
        elif declared != module.tier:
            message = f"module '{module.key}' is declared {declared} and sits under {module.tier}"
            yield Issue(module.folder, 1, message)
    for key, tier in layout.module_tiers.items():
        if key not in tree.modules:
            yield Issue(base / tier / key.split("/")[-1], 1, f"module '{key}' has no folder")
        if layout.rank(tier) is None:
            yield Issue(base / tier, 1, f"tier '{tier}' of '{key}' is not in K_TIER_ORDER")
    for label, tier in layout.scoped_roots.items():
        if not (tree.root / label).is_dir():
            yield Issue(tree.root / label, 1, f"scoped root '{label}' is declared and missing")
        if tier is None:
            continue
        if layout.rank(tier) is None:
            yield Issue(tree.root / label, 1, f"tier '{tier}' of '{label}' is not in K_TIER_ORDER")
        if tier in layout.domain_tiers:
            yield Issue(tree.root / label, 1, f"scoped root '{label}' is bound to domain '{tier}'")


def rule_upward_includes(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that ranks above the includer's tier."""
    for scope in tree.scopes:
        own = None if scope.tier is None else tree.layout.rank(scope.tier)
        for include, target in tree.module_includes(scope):
            other = tree.layout.rank(target.tier)
            if own is not None and other is not None and other > own:
                reached = f"{target.key} ({target.tier})"
                message = f"{scope.label} ({scope.tier}) includes up into {reached}"
                yield Issue(include.path, include.line, message)


def rule_cross_domain_includes(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each include by which a file of one domain names another domain."""
    domains = tree.layout.domain_tiers
    for scope in tree.scopes:
        if scope.tier not in domains:
            continue
        for include, target in tree.module_includes(scope):
            if target.tier in domains and target.tier != scope.tier:
                message = f"{scope.label} of domain {scope.tier} includes {target.key}"
                yield Issue(include.path, include.line, message)


def _private_reach(tree: SourceTree, scope: Scope, include: Include) -> str | None:
    """Returns what is wrong with one include's path, or None when it stays where it may."""
    owners = list(tree.private_owners(include.target))
    if owners and all(module is not scope.module for module, _ in owners):
        module, folder = owners[0]
        return f'"{include.target}" reaches {folder}/ of module {module.key}'
    if scope.module is None or not include.is_quoted:
        return None
    home = PurePosixPath(posixpath.normpath(scope.folder.as_posix()))
    reached = PurePosixPath(
        posixpath.normpath((include.path.parent / include.target).as_posix())
    )
    if home != reached and home not in reached.parents:
        return f'"{include.target}" leaves the folder of module {scope.module.key}'
    return None


def rule_private_reach(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each include that names another module's source/ or tests/.

    A quoted include of a module file that leaves the module's folder is reported too.
    """
    for scope in tree.scopes:
        for include in tree.includes[scope.label]:
            if message := _private_reach(tree, scope, include):
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

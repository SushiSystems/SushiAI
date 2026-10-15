# check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that modules under modules/<tier>/<module>/ depend only down the tier order.

Usage: python tools/layering/check_layering.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import posixpath
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from settings import (  # noqa: E402
    K_DOMAIN_TIERS,
    K_INTERNAL_MARK,
    K_MODULE_FUNCTION,
    K_PRIVATE_DEPENDS,
    K_PUBLIC_DEPENDS,
    K_SAME_TIER_EDGES,
    K_TIER_ORDER,
)

K_SETTINGS = Path(__file__).resolve().parents[1] / "settings.py"
K_MODULES = "modules"
K_TESTS = "tests"
K_PUBLIC_FOLDER = "include"
K_UMBRELLA = "umbrella"
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".inl", ".cpp", ".glsl", ".comp", ".vert", ".frag"})
K_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")')
K_CMAKE_SOURCE_REACH = re.compile(r"(?:^|[/\s\"'])([A-Za-z0-9_]+)/source\b")
K_CMAKE_COMMENT = re.compile(r"#.*")
K_CMAKE_KEYWORD = re.compile(r"[A-Z][A-Z_]*")


@dataclass(frozen=True, slots=True)
class Layout:
    """Holds a repository's declared tiers, its domains and its build declaration names.

    Domain tiers share one rank and never include each other.
    """

    tier_order: tuple[str, ...]
    domain_tiers: tuple[str, ...] = ()
    same_tier_edges: frozenset[tuple[str, str]] = frozenset()
    module_function: str = ""
    public_depends: str = "DEPENDS"
    private_depends: str = "PRIVATE_DEPENDS"
    internal_mark: str = "INTERNAL"

    def rank(self, tier: str) -> int | None:
        """Returns a tier's rank, lowest first and equal for every domain, or None when undeclared."""
        if tier not in self.tier_order:
            return None
        if tier in self.domain_tiers:
            return min(self.tier_order.index(domain) for domain in self.domain_tiers)
        return self.tier_order.index(tier)


K_LAYOUT = Layout(
    K_TIER_ORDER, K_DOMAIN_TIERS, K_SAME_TIER_EDGES, K_MODULE_FUNCTION,
    K_PUBLIC_DEPENDS, K_PRIVATE_DEPENDS, K_INTERNAL_MARK,
)


@dataclass(frozen=True, slots=True)
class Module:
    """Holds one module folder and its tier; `<tier>/<name>` is its key."""

    tier: str
    name: str
    folder: Path

    @property
    def key(self) -> str:
        """Returns the module's key, unique in the repository."""
        return f"{self.tier}/{self.name}"


@dataclass(frozen=True, slots=True)
class Include:
    """Holds one include directive of a source file."""

    path: Path
    line: int
    target: str
    is_quoted: bool


def _includes(folder: Path) -> Iterator[Include]:
    """Yields every include directive of every C-family source file below a folder."""
    for path in sorted(folder.rglob("*")) if folder.is_dir() else ():
        if not path.is_file() or path.suffix not in K_SOURCE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if match := K_INCLUDE.match(line):
                yield Include(path, number, match.group(1) or match.group(2), not match.group(1))


class ModuleTree:
    """Holds the modules of one repository, their includes and their build declarations."""

    def __init__(self, root: Path, layout: Layout) -> None:
        """Remembers the root and the layout; the tree is read when a rule first asks."""
        self.root = root
        self.layout = layout

    @cached_property
    def modules(self) -> dict[str, Module]:
        """Returns every folder two levels below modules/, by key."""
        base = self.root / K_MODULES
        return {
            module.key: module
            for tier in sorted(entry for entry in base.iterdir() if entry.is_dir())
            for module in (Module(tier.name, folder.name, folder)
                           for folder in sorted(entry for entry in tier.iterdir() if entry.is_dir()))
        }

    @cached_property
    def owners(self) -> dict[str, Module | None]:
        """Returns the module of each path below an include/ folder, None when several share it."""
        found: dict[str, Module | None] = {}
        for module in self.modules.values():
            public = module.folder / K_PUBLIC_FOLDER
            for path in public.rglob("*") if public.is_dir() else ():
                name = path.relative_to(public).as_posix()
                found[name] = module if found.get(name, module) is module else None
        return found

    def resolve(self, target: str) -> Module | None:
        """Returns the module whose public tree holds an include target, or None."""
        path = PurePosixPath(posixpath.normpath(target))
        for candidate in (path, *path.parents):
            if owner := self.owners.get(candidate.as_posix()):
                return owner
        return None

    def module_includes(self, module: Module, folder: Path) -> Iterator[tuple[Include, Module]]:
        """Yields each include below a folder that names another module, with that module."""
        for include in _includes(folder):
            target = self.resolve(include.target)
            if target is not None and target is not module:
                yield include, target

    def scopes(self) -> Iterator[tuple[Module, Path, bool]]:
        """Yields each module with its own folder, then with its test folder; True marks the tests."""
        for module in self.modules.values():
            yield module, module.folder, False
        for module in self.modules.values():
            tests = self.root / K_TESTS / module.tier / module.name
            if tests.is_dir():
                yield module, tests, True

    def call_tokens(self, module: Module) -> list[str]:
        """Returns the arguments of the module's declaring call in its build file."""
        build_file = module.folder / "CMakeLists.txt"
        if not self.layout.module_function or not build_file.is_file():
            return []
        text = K_CMAKE_COMMENT.sub("", build_file.read_text(encoding="utf-8"))
        call = re.search(rf"\b{re.escape(self.layout.module_function)}\s*\(([^)]*)\)", text)
        return call.group(1).split() if call else []

    def key_of(self, module: Module, token: str) -> str:
        """Returns the module key a build file token names.

        `<tier>/<name>` is taken as written. A bare name is the one module of that name outside the
        domain tiers, since a domain module that shares its name is written with its tier; failing
        that, the module of that name in the module's own tier, then the one module of that name.
        A token that names no module stays as written.
        """
        if "/" in token:
            return token
        named = [other for other in self.modules.values() if other.name == token]
        shared = [other for other in named if other.tier not in self.layout.domain_tiers]
        if len(shared) == 1:
            return shared[0].key
        if f"{module.tier}/{token}" in self.modules:
            return f"{module.tier}/{token}"
        return named[0].key if len(named) == 1 else token

    def declared(self, module: Module) -> dict[str, frozenset[str]]:
        """Returns the module keys a build file lists, by public and private keyword."""
        listed: dict[str, set[str]] = {self.layout.public_depends: set(), self.layout.private_depends: set()}
        keyword = ""
        for token in self.call_tokens(module):
            if K_CMAKE_KEYWORD.fullmatch(token):
                keyword = token
            elif keyword in listed:
                listed[keyword].add(self.key_of(module, token))
        return {keyword: frozenset(names) for keyword, names in listed.items()}

    def listed_edges(self, module: Module) -> frozenset[str]:
        """Returns the modules a module may include from its own tier.

        With a declaring function the build file is the one list; without one K_SAME_TIER_EDGES is.
        """
        if self.layout.module_function:
            return frozenset().union(*self.declared(module).values())
        return frozenset(f"{module.tier}/{included}" for includer, included in self.layout.same_tier_edges
                         if includer == module.name)

    def is_internal(self, module: Module) -> bool:
        """Returns whether a module's build file marks it internal, so its headers are not installed."""
        return self.layout.internal_mark in self.call_tokens(module)

    def is_public(self, module: Module, include: Include) -> bool:
        """Returns whether an include sits in a header the module installs."""
        return (module.folder / K_PUBLIC_FOLDER) in include.path.parents


def collect(paths: list[Path]) -> Iterable[ModuleTree]:
    """Returns the module tree of each given repository that has one."""
    return [ModuleTree(path, K_LAYOUT) for path in paths if (path / K_MODULES).is_dir()]


def rule_declared_tiers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each tier folder or domain missing from the declared order."""
    if not tree.layout.tier_order:
        yield Issue(K_SETTINGS, 1, "K_TIER_ORDER is empty; declare the repository's tiers")
        return
    for domain in tree.layout.domain_tiers:
        if domain not in tree.layout.tier_order:
            yield Issue(K_SETTINGS, 1, f"domain tier '{domain}' is not in K_TIER_ORDER")
    for tier in sorted({module.tier for module in tree.modules.values()}):
        if tree.layout.rank(tier) is None:
            yield Issue(tree.root / K_MODULES / tier, 1, f"tier '{tier}' is not in K_TIER_ORDER")


def rule_upward_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include, in a module or its tests, of a module in a higher tier.

    A test may include an umbrella header: it is what a consumer includes.
    """
    for module, folder, is_test in tree.scopes():
        own = tree.layout.rank(module.tier)
        for include, target in tree.module_includes(module, folder):
            if is_test and target.name == K_UMBRELLA:
                continue
            other = tree.layout.rank(target.tier)
            if own is not None and other is not None and other > own:
                yield Issue(include.path, include.line, f"{module.key} includes up into {target.key}")


def rule_cross_domain_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include, in a module or its tests, of a module in another domain."""
    domains = tree.layout.domain_tiers
    for module, folder, _is_test in tree.scopes():
        if module.tier not in domains:
            continue
        for include, target in tree.module_includes(module, folder):
            if target.tier in domains and target.tier != module.tier:
                yield Issue(include.path, include.line, f"{module.key} of one domain includes {target.key}")


def rule_same_tier_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module of the same tier whose edge is not listed.

    The edge is listed in the module's build file when a declaring function is named, else in K_SAME_TIER_EDGES.

    An umbrella includes every module of its scope, so its includes need no listed edge.
    """
    for module in tree.modules.values():
        if module.name == K_UMBRELLA:
            continue
        for include, target in tree.module_includes(module, module.folder):
            if target.tier == module.tier and target.key not in tree.listed_edges(module):
                source = "its CMakeLists.txt" if tree.layout.module_function else "K_SAME_TIER_EDGES"
                yield Issue(include.path, include.line, f"{module.key} includes {target.name}, an edge {source} does not list")


def rule_private_reach(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include or build line that reaches another module's source/."""
    names = {module.name for module in tree.modules.values()}
    for module in tree.modules.values():
        for include in _includes(module.folder):
            if include.is_quoted and ("../" in include.target or "source/" in include.target):
                yield Issue(include.path, include.line, "quoted include leaves the module or names a source/ folder")
        build_file = module.folder / "CMakeLists.txt"
        if not build_file.is_file():
            continue
        for number, line in enumerate(build_file.read_text(encoding="utf-8").splitlines(), start=1):
            for name in K_CMAKE_SOURCE_REACH.findall(line):
                if name != module.name and name in names:
                    yield Issue(build_file, number, f"reaches into {name}/source")


def rule_declared_dependencies(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that the includer's build file does not list."""
    layout = tree.layout
    for module in tree.modules.values():
        if not tree.call_tokens(module):
            continue
        declared = tree.declared(module)
        for include, target in tree.module_includes(module, module.folder):
            keywords = ((layout.public_depends,) if tree.is_public(module, include)
                        else (layout.public_depends, layout.private_depends))
            if not any(target.key in declared[keyword] for keyword in keywords):
                yield Issue(include.path, include.line, f"{module.name} includes {target.name}, which its "
                                                        f"CMakeLists.txt does not list in {' or '.join(keywords)}")


def rule_unused_dependencies(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each module a build file lists beyond what the module's files include."""
    layout = tree.layout
    for module in tree.modules.values():
        if not tree.call_tokens(module):
            continue
        declared = tree.declared(module)
        included: dict[bool, set[str]] = {True: set(), False: set()}
        for include, target in tree.module_includes(module, module.folder):
            included[tree.is_public(module, include)].add(target.key)
        build_file = module.folder / "CMakeLists.txt"
        for name in sorted(declared[layout.public_depends] - included[True]):
            reason = (f"only source/ includes it; list it in {layout.private_depends}"
                      if name in included[False] else "no file includes it")
            yield Issue(build_file, 1, f"{module.name} lists {name} in {layout.public_depends} and {reason}")
        for name in sorted(declared[layout.private_depends] - included[True] - included[False]):
            yield Issue(build_file, 1, f"{module.name} lists {name} in {layout.private_depends} "
                                       "and no file includes it")


def rule_public_headers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of an internal module in a header a public module installs."""
    internal = {module.key for module in tree.modules.values() if tree.is_internal(module)}
    for module in tree.modules.values():
        if module.key in internal:
            continue
        for include, target in tree.module_includes(module, module.folder):
            if tree.is_public(module, include) and target.key in internal:
                yield Issue(include.path, include.line, f"{module.name} is installed and includes "
                                                        f"{target.name}, an internal module whose headers are not")


K_CHECKER = Checker(
    description="Checks module dependencies against the declared tier order.",
    collect=collect,
    rules={
        "rule_declared_tiers": rule_declared_tiers,
        "rule_upward_includes": rule_upward_includes,
        "rule_cross_domain_includes": rule_cross_domain_includes,
        "rule_same_tier_includes": rule_same_tier_includes,
        "rule_private_reach": rule_private_reach,
        "rule_declared_dependencies": rule_declared_dependencies,
        "rule_unused_dependencies": rule_unused_dependencies,
        "rule_public_headers": rule_public_headers,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

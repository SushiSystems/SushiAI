# check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
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

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402
from settings import (  # noqa: E402
    K_DOMAIN_TIERS,
    K_INTERNAL_MARK,
    K_MODULE_FUNCTION,
    K_PRIVATE_DEPENDS,
    K_PRIVATE_DEPENDS_ALSO,
    K_PUBLIC_DEPENDS,
    K_SAME_TIER_EDGES,
    K_TIER_ORDER,
)

K_SETTINGS = Path(__file__).resolve().parents[1] / "settings.py"
K_MODULES = "modules"
K_TESTS = "tests"
# The folders below the root whose files the restricted-include rule reads.
K_INCLUDE_ROOTS = (K_MODULES, "applications", K_TESTS)
K_PUBLIC_FOLDER = "include"
K_BUILD_FILE = "CMakeLists.txt"
K_UMBRELLA = "umbrella"
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".inl", ".cpp", ".glsl", ".comp", ".vert", ".frag"})
K_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")')
K_CMAKE_SOURCE_REACH = re.compile(r"(?:^|[/\s\"'])([A-Za-z0-9_]+)/source\b")
K_CMAKE_COMMENT = re.compile(r"#.*")
K_CMAKE_KEYWORD = re.compile(r"[A-Z][A-Z_]*")
# A plain add_library that compiles sources; an IMPORTED or INTERFACE target compiles nothing.
K_PLAIN_ADD_LIBRARY = re.compile(r"(?<![\w])add_library\s*\(\s*[\w${}]+(?![^)]*\b(?:IMPORTED|INTERFACE)\b)",
                                 re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Layout:
    """Holds a repository's declared tiers, its domains, its build declaration names and its settings file.

    Domain tiers share one rank and never include each other.
    """

    tier_order: tuple[str, ...]
    domain_tiers: tuple[str, ...] = ()
    same_tier_edges: frozenset[tuple[str, str]] = frozenset()
    module_function: str = ""
    public_depends: str = "DEPENDS"
    private_depends: str = "PRIVATE_DEPENDS"
    internal_mark: str = "INTERNAL"
    private_depends_also: tuple[str, ...] = ()
    include_doors: tuple[tuple[str, tuple[str, ...]], ...] = ()
    foreign_roots: tuple[tuple[str, str], ...] = ()
    foreign_providers: tuple[tuple[str, str], ...] = ()
    settings_path: Path = Path("settings.py")

    def rank(self, tier: str) -> int | None:
        """Returns a tier's rank, lowest first and equal for every domain, or None when undeclared."""
        if tier not in self.tier_order:
            return None
        if tier in self.domain_tiers:
            return min(self.tier_order.index(domain) for domain in self.domain_tiers if domain in self.tier_order)
        return self.tier_order.index(tier)


K_LAYOUT = Layout(
    K_TIER_ORDER, K_DOMAIN_TIERS, K_SAME_TIER_EDGES, K_MODULE_FUNCTION,
    K_PUBLIC_DEPENDS, K_PRIVATE_DEPENDS, K_INTERNAL_MARK, K_PRIVATE_DEPENDS_ALSO,
    getattr(settings, "K_INCLUDE_DOORS", ()),
    tuple(getattr(settings, "K_FOREIGN_INCLUDE_ROOTS", {}).items()),
    tuple(getattr(settings, "K_FOREIGN_INCLUDE_PROVIDERS", {}).items()), K_SETTINGS,
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

    @property
    def build_file(self) -> Path:
        """Returns the path of the module's build file, which may not exist."""
        return self.folder / K_BUILD_FILE

    @property
    def public_folder(self) -> Path:
        """Returns the path of the folder whose headers the module installs, which may not exist."""
        return self.folder / K_PUBLIC_FOLDER


@dataclass(frozen=True, slots=True)
class ModuleFiles:
    """Holds what the shell read of one module: its include/ names, its sources, its tests and its build file."""

    module: Module
    has_public_folder: bool
    public_names: tuple[str, ...]
    sources: tuple[SourceFile, ...]
    tests: tuple[SourceFile, ...]
    build_text: str | None
    build_files: tuple[SourceFile, ...] = ()


@dataclass(frozen=True, slots=True)
class Include:
    """Holds one include directive of a source file."""

    path: Path
    line: int
    target: str
    is_quoted: bool


def _includes(sources: Iterable[SourceFile]) -> Iterator[Include]:
    """Yields every include directive of the given source files, in the order given."""
    for source in sources:
        for number, line in enumerate(source.text.splitlines(), start=1):
            if match := K_INCLUDE.match(line):
                yield Include(source.path, number, match.group(1) or match.group(2), not match.group(1))


class ModuleTree:
    """Holds the modules of one repository, the files the shell read for each, and its layout."""

    def __init__(
        self, root: Path, layout: Layout, entries: Iterable[ModuleFiles], restricted_sources: Iterable[SourceFile] = ()
    ) -> None:
        """Remembers the root, the layout, each module's files by the module's key, and the sources the restricted-include rule reads."""
        self.root = root
        self.layout = layout
        self.restricted_sources = tuple(restricted_sources)
        self._entries = {entry.module.key: entry for entry in entries}

    @cached_property
    def modules(self) -> dict[str, Module]:
        """Returns every module by key, in the order the shell read them."""
        return {key: entry.module for key, entry in self._entries.items()}

    def files_of(self, module: Module) -> ModuleFiles:
        """Returns what the shell read of a module."""
        return self._entries[module.key]

    @cached_property
    def owners(self) -> dict[str, Module | None]:
        """Returns the module of each name below an include/ folder, None when several share it."""
        found: dict[str, Module | None] = {}
        for module in self.modules.values():
            for name in self.files_of(module).public_names:
                found[name] = module if found.get(name, module) is module else None
        return found

    def resolve(self, target: str) -> Module | None:
        """Returns the module whose public tree holds an include target, or None.

        An include under a foreign root, such as Runtime/, names the module that links it.
        """
        path = PurePosixPath(posixpath.normpath(target))
        for root, door in self.layout.foreign_roots:
            if len(path.parts) > 1 and path.parts[0] == root:
                return next((module for module in self.modules.values() if module.name == door), None)
        for candidate in (path, *path.parents):
            if owner := self.owners.get(candidate.as_posix()):
                return owner
        return self.generated_owners.get(path.parts[1]) if len(path.parts) > 2 else None

    @cached_property
    def generated_owners(self) -> dict[str, Module]:
        """Returns each module with no include/ folder by name, when no other module shares the name.

        Such a module ships only headers its build generates, so an include of
        <Project/module/...> names it although no file below modules/ matches.
        """
        counts: dict[str, int] = {}
        for module in self.modules.values():
            counts[module.name] = counts.get(module.name, 0) + 1
        return {module.name: module for module in self.modules.values()
                if counts[module.name] == 1 and not self.files_of(module).has_public_folder}

    def module_includes(
        self, module: Module, sources: Iterable[SourceFile]
    ) -> Iterator[tuple[Include, Module]]:
        """Yields each include of the given sources that names another module, with that module."""
        for include in _includes(sources):
            target = self.resolve(include.target)
            if target is not None and target is not module and not self.is_provided(module, include):
                yield include, target

    def is_provided(self, module: Module, include: Include) -> bool:
        """Returns whether a foreign-root include reaches the module through an external target it links.

        Such a module names the root's provider in its build call, so it needs no edge to the door.
        """
        root = include.target.split("/", 1)[0]
        provided = {provider for provider, owned in self.layout.foreign_providers if owned == root}
        return any(token.lower() in provided for token in self.call_tokens(module))

    def own_includes(self, module: Module) -> Iterator[tuple[Include, Module]]:
        """Yields each include of a module's own sources that names another module, with that module."""
        return self.module_includes(module, self.files_of(module).sources)

    def scopes(self) -> Iterator[tuple[Module, tuple[SourceFile, ...], bool]]:
        """Yields each module with its sources, then with its tests; True marks the tests."""
        for module in self.modules.values():
            yield module, self.files_of(module).sources, False
        for module in self.modules.values():
            yield module, self.files_of(module).tests, True

    def call_tokens(self, module: Module) -> list[str]:
        """Returns the arguments of the module's declaring call in its build file."""
        text = self.files_of(module).build_text
        if not self.layout.module_function or text is None:
            return []
        text = K_CMAKE_COMMENT.sub("", text)
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
        """Returns the module keys a build file lists, by public and private keyword.

        A keyword in `private_depends_also` lists into the private keyword's set.
        """
        listed: dict[str, set[str]] = {self.layout.public_depends: set(), self.layout.private_depends: set()}
        keyword = ""
        for token in self.call_tokens(module):
            if K_CMAKE_KEYWORD.fullmatch(token):
                keyword = self.layout.private_depends if token in self.layout.private_depends_also else token
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
        return module.public_folder in include.path.parents


def rule_declared_tiers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each tier folder or domain missing from the declared order."""
    if not tree.layout.tier_order:
        yield Issue(tree.layout.settings_path, 1, "K_TIER_ORDER is empty; declare the repository's tiers")
        return
    for domain in tree.layout.domain_tiers:
        if domain not in tree.layout.tier_order:
            yield Issue(tree.layout.settings_path, 1, f"domain tier '{domain}' is not in K_TIER_ORDER")
    for tier in sorted({module.tier for module in tree.modules.values()}):
        if tree.layout.rank(tier) is None:
            yield Issue(tree.root / K_MODULES / tier, 1, f"tier '{tier}' is not in K_TIER_ORDER")


def rule_upward_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include, in a module or its tests, of a module in a higher tier.

    A test may include an umbrella header: it is what a consumer includes.
    """
    for module, sources, is_test in tree.scopes():
        own = tree.layout.rank(module.tier)
        for include, target in tree.module_includes(module, sources):
            if is_test and target.name == K_UMBRELLA:
                continue
            other = tree.layout.rank(target.tier)
            if own is not None and other is not None and other > own:
                yield Issue(include.path, include.line, f"{module.key} includes up into {target.key}")


def rule_cross_domain_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include, in a module or its tests, of a module in another domain."""
    domains = tree.layout.domain_tiers
    for module, sources, _is_test in tree.scopes():
        if module.tier not in domains:
            continue
        for include, target in tree.module_includes(module, sources):
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
        for include, target in tree.own_includes(module):
            if target.tier == module.tier and target.key not in tree.listed_edges(module):
                source = "its CMakeLists.txt" if tree.layout.module_function else "K_SAME_TIER_EDGES"
                yield Issue(include.path, include.line, f"{module.key} includes {target.name}, an edge {source} does not list")


def rule_private_reach(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include or build line that reaches another module's source/."""
    names = {module.name for module in tree.modules.values()}
    for module in tree.modules.values():
        for include in _includes(tree.files_of(module).sources):
            if include.is_quoted and ("../" in include.target or "source/" in include.target):
                yield Issue(include.path, include.line, "quoted include leaves the module or names a source/ folder")
        build_text = tree.files_of(module).build_text
        if build_text is None:
            continue
        for number, line in enumerate(build_text.splitlines(), start=1):
            for name in K_CMAKE_SOURCE_REACH.findall(line):
                if name != module.name and name in names:
                    yield Issue(module.build_file, number, f"reaches into {name}/source")


def rule_declared_dependencies(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that the includer's build file does not list."""
    layout = tree.layout
    for module in tree.modules.values():
        if not tree.call_tokens(module):
            continue
        declared = tree.declared(module)
        for include, target in tree.own_includes(module):
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
        for include, target in tree.own_includes(module):
            included[tree.is_public(module, include)].add(target.key)
        build_file = module.build_file
        for name in sorted(declared[layout.public_depends] - included[True]):
            reason = (f"only source/ includes it; list it in {layout.private_depends}"
                      if name in included[False] else "no file includes it")
            yield Issue(build_file, 1, f"{module.name} lists {name} in {layout.public_depends} and {reason}")
        for name in sorted(declared[layout.private_depends] - included[True] - included[False]):
            yield Issue(build_file, 1, f"{module.name} lists {name} in {layout.private_depends} "
                                       "and no file includes it")


def rule_declared_libraries(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each plain add_library call in a build file below modules/.

    Every library under modules/ is declared through the module function, so the tier it belongs to is read
    from one declaration. Skipped when the layout names no module function.
    """
    if not tree.layout.module_function:
        return
    for module in tree.modules.values():
        for build_file in tree.files_of(module).build_files:
            for number, line in enumerate(build_file.text.splitlines(), start=1):
                if K_PLAIN_ADD_LIBRARY.search(K_CMAKE_COMMENT.sub("", line)):
                    yield Issue(build_file.path, number, f"{module.key} declares a library with add_library; "
                                                         f"declare it through {tree.layout.module_function}")


def rule_public_headers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of an internal module in a header a public module installs."""
    internal = {module.key for module in tree.modules.values() if tree.is_internal(module)}
    for module in tree.modules.values():
        if module.key in internal:
            continue
        for include, target in tree.own_includes(module):
            if tree.is_public(module, include) and target.key in internal:
                yield Issue(include.path, include.line, f"{module.name} is installed and includes "
                                                        f"{target.name}, an internal module whose headers are not")


def rule_restricted_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a restricted prefix outside the paths that may hold it.

    `include_doors` pairs an include prefix with the path prefixes, below the root, whose files may
    include it. Source, applications and tests are all read. Skipped when the layout names no door.
    """
    for prefix, doors in tree.layout.include_doors:
        for include in _includes(tree.restricted_sources):
            relative = include.path.relative_to(tree.root).as_posix()
            if include.target.startswith(prefix) and not relative.startswith(doors):
                yield Issue(include.path, include.line,
                            f"includes {include.target}; only the listed paths may include {prefix}, "
                            "so go through the front module")


def _read_module_folders(root: Path) -> list[Module]:
    """Returns every folder two levels below modules/, in tier and name order."""
    base = root / K_MODULES
    return [
        Module(tier.name, folder.name, folder)
        for tier in sorted(entry for entry in base.iterdir() if entry.is_dir())
        for folder in sorted(entry for entry in tier.iterdir() if entry.is_dir())
    ]


def _read_sources(folder: Path) -> tuple[SourceFile, ...]:
    """Reads every C-family source below a folder, in path order; none when the folder is absent."""
    if not folder.is_dir():
        return ()
    paths = [
        path for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in K_SOURCE_SUFFIXES
    ]
    return read_files(paths, lambda _: True)


def _read_public_names(module: Module) -> tuple[bool, tuple[str, ...]]:
    """Returns whether a module has an include/ folder, and the name of each entry below it."""
    public = module.public_folder
    if not public.is_dir():
        return False, ()
    return True, tuple(path.relative_to(public).as_posix() for path in public.rglob("*"))


def _read_build_text(module: Module) -> str | None:
    """Returns the text of a module's build file, or None when it has none."""
    found = read_files([module.build_file], Path.is_file)
    return found[0].text if found else None


def _read_module(root: Path, module: Module) -> ModuleFiles:
    """Reads the files of one module into values."""
    has_public, names = _read_public_names(module)
    return ModuleFiles(
        module=module,
        has_public_folder=has_public,
        public_names=names,
        sources=_read_sources(module.folder),
        tests=_read_sources(root / K_TESTS / module.tier / module.name),
        build_text=_read_build_text(module),
        build_files=read_files(sorted(module.folder.rglob(K_BUILD_FILE)), Path.is_file),
    )


def _read_repository(root: Path, layout: Layout) -> ModuleTree:
    """Reads one repository's modules into a tree; the only reader of the disk in this checker."""
    restricted = tuple(source for folder in K_INCLUDE_ROOTS for source in _read_sources(root / folder))
    return ModuleTree(root, layout, [_read_module(root, module) for module in _read_module_folders(root)], restricted)


def collect(paths: list[Path]) -> Iterable[ModuleTree]:
    """Returns the module tree of each given repository that has one."""
    return [_read_repository(path, K_LAYOUT) for path in paths if (path / K_MODULES).is_dir()]


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
        "rule_declared_libraries": rule_declared_libraries,
        "rule_public_headers": rule_public_headers,
        "rule_restricted_includes": rule_restricted_includes,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Checks that modules under modules/<tier>/<module>/ depend only down the tier order.

Usage: python tools/layering/check_layering.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_TIER_ORDER: tuple[str, ...] = ()
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".glsl", ".comp", ".vert", ".frag"})
K_ANGLE_INCLUDE = re.compile(r'^\s*#\s*include\s*<[^/>]+/([^/>]+)/[^>]+>')
K_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
K_CMAKE_SOURCE_REACH = re.compile(r"(?:^|[/\s\"'])([A-Za-z0-9_]+)/source\b")


@dataclass(frozen=True, slots=True)
class Module:
    """Holds one module folder and its tier."""

    tier: str
    name: str
    folder: Path


@dataclass(frozen=True, slots=True)
class ModuleTree:
    """Holds every module of one repository and the declared tier order."""

    root: Path
    modules: dict[str, Module]
    tier_order: tuple[str, ...]

    def rank(self, tier: str) -> int | None:
        """Returns a tier's position in the order, lowest first, or None when undeclared."""
        return self.tier_order.index(tier) if tier in self.tier_order else None


def _discover(root: Path) -> dict[str, Module]:
    """Returns every module below root/modules, keyed by module name."""
    modules: dict[str, Module] = {}
    base = root / "modules"
    for tier in sorted(entry for entry in base.iterdir() if entry.is_dir()):
        for folder in sorted(entry for entry in tier.iterdir() if entry.is_dir()):
            modules[folder.name] = Module(tier.name, folder.name, folder)
    return modules


def collect(paths: list[Path]) -> Iterable[ModuleTree]:
    """Returns the module tree of each given repository that has one."""
    return [
        ModuleTree(path, _discover(path), K_TIER_ORDER)
        for path in paths if (path / "modules").is_dir()
    ]


def _source_lines(module: Module) -> Iterator[tuple[Path, int, str]]:
    """Yields every line of every C-family source file in a module."""
    for path in sorted(module.folder.rglob("*")):
        if path.is_file() and path.suffix in K_SOURCE_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                yield path, number, line


def rule_declared_tiers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each tier folder missing from the declared order."""
    if not tree.tier_order:
        yield Issue(Path(__file__), 1, "K_TIER_ORDER is empty; declare the repository's tiers")
        return
    for tier in sorted({module.tier for module in tree.modules.values()}):
        if tree.rank(tier) is None:
            yield Issue(tree.root / "modules" / tier, 1, f"tier '{tier}' is not in K_TIER_ORDER")


def rule_upward_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that sits in a higher tier."""
    for module in tree.modules.values():
        own = tree.rank(module.tier)
        for path, number, line in _source_lines(module):
            match = K_ANGLE_INCLUDE.match(line)
            target = tree.modules.get(match.group(1)) if match else None
            if own is None or target is None:
                continue
            other = tree.rank(target.tier)
            if other is not None and other > own:
                yield Issue(path, number, f"{module.tier}/{module.name} includes up into {target.tier}/{target.name}")


def rule_private_reach(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include or build line that reaches another module's source/."""
    for module in tree.modules.values():
        for path, number, line in _source_lines(module):
            match = K_QUOTED_INCLUDE.match(line)
            if match and ("../" in match.group(1) or "source/" in match.group(1)):
                yield Issue(path, number, "quoted include leaves the module or names a source/ folder")
        build_file = module.folder / "CMakeLists.txt"
        if not build_file.is_file():
            continue
        for number, line in enumerate(build_file.read_text(encoding="utf-8").splitlines(), start=1):
            for name in K_CMAKE_SOURCE_REACH.findall(line):
                if name != module.name and name in tree.modules:
                    yield Issue(build_file, number, f"reaches into {name}/source")


K_CHECKER = Checker(
    description="Checks module dependencies against the declared tier order.",
    collect=collect,
    rules={
        "rule_declared_tiers": rule_declared_tiers,
        "rule_upward_includes": rule_upward_includes,
        "rule_private_reach": rule_private_reach,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

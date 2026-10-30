# tooling_bridge.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the tooling packages that own the backlog and the design rules, for the checkers that call them.

The backlog and the design topics are written by the `backlog` and `design` modules of one tooling
package, and their rules live there once. K_TOOLING_PACKAGE in tools/settings.py names the package.
A checker here runs those rules; it finds the package installed, or a checkout of it beside the
repository, and otherwise reports that the rules could not run.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Issue  # noqa: E402

K_PACKAGE: str = getattr(settings, "K_TOOLING_PACKAGE", "")
K_MINIMUM_VERSION: str = getattr(settings, "K_TOOLING_MINIMUM_VERSION", "")
K_SIBLING_SOURCE: str = getattr(settings, "K_TOOLING_SIBLING_SOURCE", "")


def load(name: str, root: Path) -> ModuleType | None:
    """Returns the module *name* of the tooling package, or None when no install or sibling checkout has it."""
    if not K_PACKAGE:
        return None
    module = _import(f"{K_PACKAGE}.{name}")
    if module is not None:
        return module
    sibling = root.resolve().parent / K_SIBLING_SOURCE if K_SIBLING_SOURCE else None
    if sibling is not None and sibling.is_dir() and str(sibling) not in sys.path:
        sys.path.append(str(sibling))
        return _import(f"{K_PACKAGE}.{name}")
    return None


def missing(path: Path, name: str) -> Issue:
    """Returns the issue that says the rules of the module *name* could not run."""
    if not K_PACKAGE:
        return Issue(path, 1, f"tools/settings.py sets no K_TOOLING_PACKAGE, so the {name} rules cannot run")
    version = f" {K_MINIMUM_VERSION} or later" if K_MINIMUM_VERSION else ""
    return Issue(path, 1, f"{K_PACKAGE}.{name} is not importable; install {K_PACKAGE}{version} to run these rules")


class Rooted(Protocol):
    """A checker subject that knows its repository root."""

    root: Path


def design_rule(name: str, reports_missing: bool) -> Callable[[Rooted], Iterator[Issue]]:
    """Returns a checker rule that runs the design rule *name* of the tooling package.

    The rule with *reports_missing* says once that the package could not be loaded; the others say nothing.
    """

    def rule(repository: Rooted) -> Iterator[Issue]:
        """Yields each finding of one design rule."""
        design = load("design", repository.root)
        if design is None:
            if reports_missing:
                yield missing(repository.root / "docs" / "design", "design")
            return
        for finding in design.check_design(design.DesignStore(repository.root), [name]):
            yield Issue(finding.path, finding.line, finding.message)

    rule.__doc__ = f"Yields each finding of the design rule {name}."
    return rule


def _import(name: str) -> ModuleType | None:
    """Returns a module, or None when it or what it needs is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None

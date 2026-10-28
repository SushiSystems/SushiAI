# sushicore_bridge.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the sushicore packages that own the backlog and the design rules, for the checkers that call them.

The backlog and the design topics are written by `sushicore.backlog` and `sushicore.design`, and
their rules live there once. A checker here runs those rules; it finds sushicore installed, or a
checkout of it beside the repository, and otherwise reports that the rules could not run.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Protocol

from checking.checker import Issue

K_MINIMUM_VERSION = "0.10.0"
K_SIBLING_SOURCE = Path("sushicore") / "source"


def load(name: str, root: Path) -> ModuleType | None:
    """Returns the sushicore package *name*, or None when neither an install nor a sibling checkout has it."""
    module = _import(name)
    if module is not None:
        return module
    sibling = root.resolve().parent / K_SIBLING_SOURCE
    if sibling.is_dir() and str(sibling) not in sys.path:
        sys.path.append(str(sibling))
        return _import(name)
    return None


def missing(path: Path, name: str) -> Issue:
    """Returns the issue that says the rules of *name* could not run."""
    return Issue(path, 1, f"{name} is not importable; install sushicore {K_MINIMUM_VERSION} or later to run these rules")


class Rooted(Protocol):
    """A checker subject that knows its repository root."""

    root: Path


def design_rule(name: str, reports_missing: bool) -> Callable[[Rooted], Iterator[Issue]]:
    """Returns a checker rule that runs the sushicore.design rule *name*.

    The rule with *reports_missing* says once that sushicore could not be loaded; the others say nothing.
    """

    def rule(repository: Rooted) -> Iterator[Issue]:
        """Yields each finding of one design rule."""
        design = load("sushicore.design", repository.root)
        if design is None:
            if reports_missing:
                yield missing(repository.root / "docs" / "design", "sushicore.design")
            return
        for finding in design.check_design(design.DesignStore(repository.root), [name]):
            yield Issue(finding.path, finding.line, finding.message)

    rule.__doc__ = f"Yields each finding of the sushicore.design rule {name}."
    return rule


def _import(name: str) -> ModuleType | None:
    """Returns a module, or None when it or what it needs is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None

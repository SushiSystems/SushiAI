# nunit_results.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Reads the times of the NUnit 3 results file that Unity writes for an EditMode or a PlayMode run."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator

from testing.timing import Timing

K_ROOT_TAG = "test-run"
K_SIZES = ("small", "medium", "large")
K_BENCHMARK = "benchmark"
K_SKIPPED = "Skipped"


def _categories(element: ElementTree.Element) -> set[str]:
    """Returns the values of the `Category` properties an element holds directly."""
    return {prop.get("value", "") for prop in element.findall("properties/property") if prop.get("name") == "Category"}


def _size_of(element: ElementTree.Element, inherited: str | None) -> str | None:
    """Returns the size an element's categories give, `None` for a benchmark, else the size it inherits."""
    names = _categories(element)
    if K_BENCHMARK in names:
        return None
    return next((size for size in K_SIZES if size in names), inherited)


def _name_of(case: ElementTree.Element) -> str:
    """Returns the name a test case is judged and exempted by: its fullname, with its method put in when a set name hid it."""
    fullname = case.get("fullname", case.get("name", ""))
    method = case.get("methodname", "")
    if not method or f".{method}" in fullname:
        return fullname
    return f"{case.get('classname', '')}.{method}.{case.get('name', '')}"


def _cases(element: ElementTree.Element, inherited: str | None) -> Iterator[Timing]:
    """Yields the timing of each test case below an element, sized by the nearest category above it."""
    for child in element:
        if child.tag == "test-suite":
            yield from _cases(child, _size_of(child, inherited))
        elif child.tag == "test-case" and child.get("result") != K_SKIPPED:
            yield Timing(_name_of(child), float(child.get("duration", "0") or 0),
                         _size_of(child, inherited), child.get("classname", ""))


def parse_nunit(root: ElementTree.Element) -> list[Timing]:
    """Returns the time of each test that ran in an NUnit 3 `test-run`, with the size its category gives.

    A fixture's `Category` property sizes its tests, a test's own property beats it, and a benchmark has no
    size. A skipped test is left out.
    """
    return list(_cases(root, None))

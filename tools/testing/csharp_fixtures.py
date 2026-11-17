# csharp_fixtures.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Finds the NUnit fixtures of a C# source: the top-level classes that declare tests, and their test methods."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from testing.source_blanking import blank, matching_brace

K_CLASS = re.compile(
    r"^[ \t]*((?:\[[^\]\n]*\][ \t]*\n?[ \t]*)*)"
    r"((?:(?:public|internal|private|protected|sealed|static|abstract|partial)\s+)*)"
    r"class\s+(\w+)",
    re.MULTILINE,
)
K_TEST_ATTRIBUTE = re.compile(r"\[\s*(?:Test|UnityTest|TestCase|TestCaseSource|Theory)\b")
K_CATEGORY = re.compile(r"Category\s*\(\s*\"(\w+)\"\s*\)")
K_METHOD = re.compile(
    r"((?:\[[^\]]*\]\s*)+)(?:(?:public|internal|private|protected|static|async|override|virtual)\s+)*"
    r"[\w<>\[\],.?]+\s+(\w+)\s*\([^)]*\)\s*(?:where[^{;]*)?\{"
)


@dataclass(frozen=True, slots=True)
class TestMethod:
    """Holds one test method of a fixture: its name, its categories and the offsets of its body."""

    name: str
    categories: frozenset[str]
    start: int
    opening: int
    closing: int


@dataclass(frozen=True, slots=True)
class Fixture:
    """Holds one NUnit fixture: its name, its categories, where its attributes open and its test methods."""

    name: str
    categories: frozenset[str]
    start: int
    methods: tuple[TestMethod, ...]


def _top_level(code: str, start: int = 0, end: int | None = None) -> Iterator[tuple[re.Match[str], int, int]]:
    """Yields each class declaration between two offsets that lies in no other class, with the offsets of its braces."""
    taken_until = -1
    for match in K_CLASS.finditer(code, start, len(code) if end is None else end):
        if match.start() < taken_until:
            continue
        opening = code.find("{", match.end())
        closing = matching_brace(code, opening) if opening != -1 else None
        if closing is None:
            continue
        taken_until = closing
        yield match, opening, closing


def _methods(text: str, code: str, opening: int, closing: int) -> tuple[TestMethod, ...]:
    """Returns the test methods whose declaration lies between two offsets of a class body."""
    found = []
    nested = [(inner.start(), end) for inner, _, end in _top_level(code, opening + 1, closing)]
    for match in K_METHOD.finditer(code, opening + 1, closing):
        if not K_TEST_ATTRIBUTE.search(code, match.start(1), match.end(1)):
            continue
        if any(start < match.start(1) < end for start, end in nested):
            continue
        body = match.end() - 1
        end = matching_brace(code, body)
        if end is not None:
            found.append(TestMethod(match.group(2), frozenset(K_CATEGORY.findall(text[match.start(1):match.end(1)])),
                                    match.start(1), body, end))
    return tuple(found)


def fixtures_of(text: str) -> list[Fixture]:
    """Returns the fixtures of a C# source: top-level, non-abstract classes that carry `[TestFixture]` or a test.

    A class nested in another is a helper or a stub and is never a fixture.
    """
    code = blank(text, backtick=False, raw=False)
    fixtures = []
    for match, opening, closing in _top_level(code):
        methods = _methods(text, code, opening, closing)
        is_fixture = "TestFixture" in match.group(1) or K_TEST_ATTRIBUTE.search(code, opening, closing)
        if not is_fixture or "abstract" in match.group(2).split():
            continue
        start = match.start(1) if match.group(1) else match.start(3)
        fixtures.append(Fixture(match.group(3), frozenset(K_CATEGORY.findall(text[match.start(1):match.end(1)])),
                                start, methods))
    return fixtures

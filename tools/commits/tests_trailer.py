# tests_trailer.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Decides whether a fix or feature commit carries a test, or a `Tests:` line that says why not.

The commit-msg hook calls tests_problems() when K_TESTS_LINE is set in tools/settings.py.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402

#: Whether a fix or feature commit must stage a test or carry a `Tests:` line.
K_TESTS_LINE: bool = getattr(settings, "K_TESTS_LINE", False)
#: The commit types that change behaviour and so need a test.
K_TESTED_TYPES = ("fix", "feat")
K_SUBJECT = re.compile(r"^(?P<type>[a-z]+)(\([^)]*\))?!?:")
K_NONE_LINE = re.compile(r"^Tests: none \(.+\)\s*$", re.MULTILINE)
K_TEST_FOLDERS = ("tests", "test", "Tests")
K_TEST_NAME = re.compile(r"(^test_.*|.*_test\.[^.]+|.*Tests?\.[^.]+)$")


def is_test_path(path: str) -> bool:
    """Returns whether a repository path is a test file or lies in a test folder."""
    pure = PurePosixPath(path)
    return any(part in K_TEST_FOLDERS for part in pure.parts[:-1]) or bool(K_TEST_NAME.match(pure.name))


def tests_problems(message: str, staged: Iterable[str]) -> list[str]:
    """Returns the problem with a fix or feature commit that stages no test and says no reason; else empty."""
    subject = message.lstrip().splitlines()[0] if message.strip() else ""
    match = K_SUBJECT.match(subject)
    if match is None or match.group("type") not in K_TESTED_TYPES:
        return []
    if any(is_test_path(path) for path in staged) or K_NONE_LINE.search(message):
        return []
    return [
        f"a '{match.group('type')}' commit stages no test; add one, or write 'Tests: none (<reason>)'"
    ]

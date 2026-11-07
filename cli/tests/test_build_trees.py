# test_build_trees.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that each build type and sanitizer set has a tree of its own."""

from __future__ import annotations

from pathlib import Path

import pytest

from sushiai.services import build_trees
from sushiai.services.sanitizer import Sanitizer


@pytest.mark.small
@pytest.mark.parametrize(("build_type", "sanitizer", "tree"), [
    ("release", None, "default-release"),
    ("debug", None, "default-debug"),
    ("debug", Sanitizer.address_undefined, "asan-debug"),
    ("release", Sanitizer.thread, "tsan-release"),
    ("debug", Sanitizer.none, "default-debug"),
])
def test_each_build_type_and_sanitizer_has_a_tree_of_its_own(build_type, sanitizer, tree):
    """A build type and a sanitizer set name one folder under build/."""
    root = Path("checkout")
    arguments = () if sanitizer is None else (sanitizer,)
    assert build_trees.build_dir(root, build_type, *arguments) == root / "build" / tree

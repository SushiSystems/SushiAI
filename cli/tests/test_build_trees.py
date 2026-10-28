# test_build_trees.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that each build type and sanitizer set has a tree of its own."""

from __future__ import annotations

import pytest

from sushiai.services import build_trees
from sushiai.services.sanitizer import Sanitizer

pytestmark = pytest.mark.small


def test_each_build_type_and_sanitizer_has_a_tree_of_its_own(tmp_path):
    assert build_trees.build_dir(tmp_path, "release") == tmp_path / "build" / "default-release"
    assert build_trees.build_dir(tmp_path, "debug") == tmp_path / "build" / "default-debug"
    assert build_trees.build_dir(tmp_path, "debug", Sanitizer.address_undefined) == (
        tmp_path / "build" / "asan-debug")
    assert build_trees.build_dir(tmp_path, "release", Sanitizer.thread) == (
        tmp_path / "build" / "tsan-release")
    assert build_trees.build_dir(tmp_path, "debug", Sanitizer.none) == (
        tmp_path / "build" / "default-debug")



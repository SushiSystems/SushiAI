# test_run.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the choice `sa run --pick` offers: the registered examples, or every program."""

from __future__ import annotations

from pathlib import Path

import pytest

from sushiai.services import build_trees, project

from recorded_tools import Recorded, recorded_tools, write_executable


@pytest.fixture
def examples_tree(project_root: Path) -> Path:
    """Returns a checkout whose one registered example is built beside a second program."""
    examples = project_root / "applications" / "examples"
    examples.mkdir(parents=True)
    (examples / "CMakeLists.txt").write_text("add_executable(demo demo.cpp)\n", encoding="utf-8")
    tree = build_trees.build_dir(project_root, project.K_DEFAULT_TYPE.value)
    write_executable(tree / "demo")
    write_executable(tree / "zeta_tests")
    return project_root


def _run_sorted(recorded: Recorded, *, list_all: bool) -> int:
    """Runs `sa run --pick` with no memory ceiling, and returns its exit code."""
    return project.run(recorded.tools, sort=True, memory_limit_gb=0.0, list_all=list_all)


@pytest.mark.medium
def test_run_sort_offers_only_the_registered_examples(examples_tree):
    """`sa run --pick` numbers the example alone, so a second program cannot be picked."""
    recorded = recorded_tools(answer="2")

    assert _run_sorted(recorded, list_all=False) == 1
    assert recorded.runner.commands == []
    assert recorded.errors() == ["'2' is not a number from 1 to 1."]


@pytest.mark.medium
def test_run_sort_all_runs_the_program_picked_from_every_program(examples_tree):
    """`sa run --pick --all` numbers the second program too, and runs the one picked."""
    recorded = recorded_tools(answer="2")

    assert _run_sorted(recorded, list_all=True) == 0
    assert Path(recorded.runner.commands[0].argv[0]).stem == "zeta_tests"

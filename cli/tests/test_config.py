# test_config.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for project-root resolution and config-dir selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from sushiai import config


@pytest.mark.small
def test_find_project_root_raises_outside_checkout():
    """No ancestor is a root -> a clear SystemExit, not a crash."""
    with pytest.raises(SystemExit):
        config.find_project_root(Path("nowhere") / "below", is_root=lambda folder: False)


@pytest.mark.medium
def test_find_project_root_walks_up_to_marker(tmp_path):
    """The root is the nearest ancestor holding CMakeLists.txt."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert config.find_project_root(nested) == tmp_path


@pytest.mark.small
@pytest.mark.parametrize("root", [Path("checkout"), Path("workspace") / "sushiai"])
def test_config_dir_is_the_cli_folder_of_the_root(root):
    """config_dir() returns <root>/cli, whether or not the folder or its config.toml exist."""
    assert config.config_dir(root) == root / "cli"

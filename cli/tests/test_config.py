# test_config.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for project-root resolution and config-dir selection."""

from __future__ import annotations

import pytest

from sushiai import config


def test_find_project_root_raises_outside_checkout(tmp_path):
    """No CMakeLists.txt anywhere above -> a clear SystemExit, not a crash."""
    with pytest.raises(SystemExit):
        config.find_project_root(tmp_path)


def test_find_project_root_walks_up_to_marker(tmp_path):
    """The root is the nearest ancestor holding CMakeLists.txt."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert config.find_project_root(nested) == tmp_path


def test_config_dir_prefers_cli(tmp_path):
    """config_dir() returns <root>/cli when it holds config.toml."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    (cli_dir / "config.toml").write_text("")
    assert config.config_dir(tmp_path) == cli_dir


def test_config_dir_defaults_to_cli_when_absent(tmp_path):
    """With no config.toml present, config_dir() still defaults to <root>/cli."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    assert config.config_dir(tmp_path) == tmp_path / "cli"


def test_runtime_dir_defaults_to_sibling(tmp_path):
    """With no override, runtime_dir() resolves to <root>/../sushiruntime."""
    root = tmp_path / "sushiai"
    root.mkdir()
    cfg = config.Config()
    assert cfg.runtime_dir(root) == (root / ".." / "sushiruntime").resolve()


def test_blas_dir_defaults_to_sibling(tmp_path):
    """With no override, blas_dir() resolves to <root>/../sushiblas."""
    root = tmp_path / "sushiai"
    root.mkdir()
    cfg = config.Config()
    assert cfg.blas_dir(root) == (root / ".." / "sushiblas").resolve()

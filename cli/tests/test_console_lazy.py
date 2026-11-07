# test_console_lazy.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that the console is built lazily, through sushicore's LazyConsole.

The reasoning is in cli/tests/README.md under "test_console_lazy.py".
"""

from __future__ import annotations

import importlib

import pytest


def _fresh_console_module():
    """Import (or re-import) sushiai.console with a clean module state."""
    import sushiai.console as console_mod
    importlib.reload(console_mod)
    return console_mod


@pytest.mark.medium
def test_import_does_not_build_console():
    """Importing the module must not materialise the console."""
    console_mod = _fresh_console_module()
    assert console_mod._lazy.built is False


@pytest.mark.medium
def test_output_helper_falls_back_outside_project(monkeypatch, tmp_path):
    """console.info(...) outside a project builds a config-less console, no crash."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()

    # No project means no config files to layer -- and, crucially, no crash.
    assert list(console_mod._lazy.sources()) == []

    console_mod.info("hello from outside a checkout")
    assert console_mod._lazy.built is True


@pytest.mark.medium
def test_sources_inside_project_are_this_repos_config(project_root):
    """Inside a checkout the console layers this CLI's own two config files."""
    console_mod = _fresh_console_module()
    names = [p.name for p in console_mod._lazy.sources()]
    assert names == ["config.toml", "config.local.toml"]
    assert {p.parent for p in console_mod._lazy.sources()} == {project_root / "cli"}


@pytest.mark.medium
def test_console_attribute_is_the_console_the_lazy_holder_built(monkeypatch, tmp_path):
    """The raw Rich `console` attribute is the one inside the console current() returns."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    assert console_mod.console is console_mod.current().console


@pytest.mark.medium
def test_unknown_attribute_raises_attribute_error():
    """A name outside sushicore's list raises AttributeError."""
    console_mod = _fresh_console_module()
    with pytest.raises(AttributeError):
        _ = console_mod.does_not_exist


@pytest.mark.medium
def test_the_output_helpers_print_through_the_console_that_current_returns(monkeypatch, tmp_path):
    """The help group and the output helpers share one console."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    assert console_mod.info == console_mod.current().info
    assert console_mod.error == console_mod.current().error

# test_console_lazy.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
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


def test_import_does_not_build_console():
    """Importing the module must not materialise the console."""
    console_mod = _fresh_console_module()
    assert console_mod._lazy.built is False


def test_import_outside_project_does_not_raise(monkeypatch, tmp_path):
    """Simulate running outside a checkout: import + reload must not SystemExit."""
    monkeypatch.chdir(tmp_path)
    _fresh_console_module()


def test_output_helper_falls_back_outside_project(monkeypatch, tmp_path):
    """console.info(...) outside a project builds a config-less console, no crash."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()

    # No project means no config files to layer -- and, crucially, no crash.
    assert list(console_mod._lazy.sources()) == []

    console_mod.info("hello from outside a checkout")
    assert console_mod._lazy.built is True


def test_sources_inside_project_are_this_repos_config(monkeypatch):
    """Inside a checkout the console layers this CLI's own two config files."""
    console_mod = _fresh_console_module()
    names = [p.name for p in console_mod._lazy.sources()]
    assert names == ["config.toml", "config.local.toml"]


def test_console_attribute_resolves(monkeypatch, tmp_path):
    """The raw Rich `console` attribute resolves lazily too."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    assert console_mod.console is not None


def test_unknown_attribute_raises_attribute_error():
    console_mod = _fresh_console_module()
    with pytest.raises(AttributeError):
        _ = console_mod.does_not_exist


def test_current_returns_the_console_the_lazy_holder_builds(monkeypatch, tmp_path):
    """current() hands the help group the same sushicore Console the output helpers use."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    assert console_mod.current() is console_mod._lazy.get()
    assert console_mod._lazy.built is True

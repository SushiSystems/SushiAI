"""Regression tests for the lazy console initialisation.

The bug: console.py used to build the console at import time, which called
config_dir() -> find_project_root() and raised SystemExit outside a checkout.
Because cli.py imports console transitively, *every* `sa` invocation —
including `sa --help`, which Typer can answer with no project — died with
"Not inside a SushiAI project". These tests pin the fix: importing the
module does no filesystem work, and producing output outside a project
degrades to a default console instead of raising.
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
    # The lazy singleton stays unset until an attribute is actually accessed.
    assert console_mod._console is None


def test_import_outside_project_does_not_raise(monkeypatch, tmp_path):
    """Simulate running outside a checkout: import + reload must not SystemExit."""
    monkeypatch.chdir(tmp_path)  # tmp_path has no CMakeLists.txt ancestor
    # Reloading re-runs module top-level code; the old bug raised here.
    _fresh_console_module()


def test_output_helper_falls_back_outside_project(monkeypatch, tmp_path):
    """console.info(...) outside a project builds a config-less console, no crash."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()

    # First real use triggers construction; must not raise SystemExit.
    console_mod.info("hello from outside a checkout")

    assert console_mod._console is not None
    # Fell back to the config-less source list.
    assert console_mod._console.sources == []


def test_console_attribute_resolves(monkeypatch, tmp_path):
    """The raw Rich `console` attribute resolves lazily too."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    raw = console_mod.console
    assert raw is not None


def test_unknown_attribute_raises_attribute_error():
    console_mod = _fresh_console_module()
    with pytest.raises(AttributeError):
        _ = console_mod.does_not_exist

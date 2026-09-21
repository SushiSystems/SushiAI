"""Regression tests for the lazy console initialisation.

The bug: console.py used to build the console at import time, which called
config_dir() -> find_project_root() and raised SystemExit outside a checkout.
Because cli.py imports console transitively, *every* `sa` invocation —
including `sa --help`, which Typer can answer with no project — died with
"Not inside a SushiAI project".

Four of the five CLIs fixed this independently and sushiengine's copy never
did, so `se --help` outside a checkout failed for as long as the copies
existed. The lifecycle now lives in sushicore.cli_console.LazyConsole; these
tests pin that *this* CLI is wired to it, which is the part a shared
implementation cannot check on its own.
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

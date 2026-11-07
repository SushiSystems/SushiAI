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

from sushiai.console import make_console_holder


def _fresh_console_module():
    """Import (or re-import) sushiai.console with a clean module state."""
    import sushiai.console as console_mod
    importlib.reload(console_mod)
    return console_mod


@pytest.mark.small
def test_import_does_not_build_console(tmp_path):
    """A holder made over a config directory builds nothing until it is used."""
    holder = make_console_holder(lambda: tmp_path / "cli")
    assert holder.built is False


@pytest.mark.small
def test_output_helper_falls_back_outside_project(capsys):
    """A holder whose config_dir reports no checkout layers no files and still prints."""
    def outside_a_checkout():
        raise SystemExit(1)

    holder = make_console_holder(outside_a_checkout)

    assert list(holder.sources()) == []
    holder.attribute("info")("hello from outside a checkout")
    assert holder.built is True
    assert "hello from outside a checkout" in capsys.readouterr().out


@pytest.mark.small
def test_sources_inside_project_are_this_repos_config(tmp_path):
    """Inside a checkout the console layers this CLI's own two config files."""
    holder = make_console_holder(lambda: tmp_path / "cli")
    names = [p.name for p in holder.sources()]
    assert names == ["config.toml", "config.local.toml"]
    assert {p.parent for p in holder.sources()} == {tmp_path / "cli"}


@pytest.mark.small
def test_console_attribute_is_the_console_the_lazy_holder_built(tmp_path):
    """The raw Rich `console` attribute is the one inside the console the holder builds."""
    holder = make_console_holder(lambda: tmp_path / "cli")
    assert holder.attribute("console") is holder.get().console


@pytest.mark.small
def test_unknown_attribute_raises_attribute_error(tmp_path):
    """A name outside sushicore's list raises AttributeError."""
    holder = make_console_holder(lambda: tmp_path / "cli")
    with pytest.raises(AttributeError):
        holder.attribute("does_not_exist")


@pytest.mark.medium
def test_the_output_helpers_print_through_the_console_that_current_returns(monkeypatch, tmp_path):
    """The help group and the output helpers share one console."""
    monkeypatch.chdir(tmp_path)
    console_mod = _fresh_console_module()
    assert console_mod.info == console_mod.current().info
    assert console_mod.error == console_mod.current().error

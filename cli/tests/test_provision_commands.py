# test_provision_commands.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that sa carries the shared provision commands."""

from __future__ import annotations

from typer.testing import CliRunner

from sushicore.provision.config import ProvisionSettings
from sushiai.cli import app
from sushiai.config import Config


def test_the_four_provision_commands_are_registered():
    """Assert setup, doctor, link and unlink appear in the help."""
    out = CliRunner().invoke(app, ["--help"]).output
    for name in ("setup", "doctor", "link", "unlink"):
        assert name in out


def test_setup_takes_the_shared_options():
    """Assert setup offers --dry-run, --toolchain and --no-gpu."""
    out = CliRunner().invoke(app, ["setup", "--help"]).output
    for option in ("--dry-run", "--toolchain", "--no-gpu"):
        assert option in out


def test_config_is_a_provision_config():
    """Assert the config carries every field provision reads."""
    assert issubclass(Config, ProvisionSettings)


def test_setup_stops_when_a_sibling_checkout_is_missing(tmp_path, monkeypatch):
    """Assert setup exits 2 and names sushiruntime and sushiblas when neither has a checkout."""
    root = tmp_path / "sushiai"
    (root / "cli").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("")
    (root / "cli" / "config.toml").write_text("")
    (root / "cli" / "sushistack.deps.toml").write_text(
        '[module]\ndepends_on = ["sushiruntime", "sushiblas"]\n')
    monkeypatch.chdir(root)
    monkeypatch.setenv("SUSHISYSTEMS_HOME", str(tmp_path / "home"))
    for name in ("SUSHISTACK_HOME", "SUSHIRUNTIME_DIR", "SUSHIBLAS_DIR", "SUSHISTACK_DEPS_DIR"):
        monkeypatch.delenv(name, raising=False)
    result = CliRunner().invoke(app, ["setup", "--dry-run"])
    assert result.exit_code == 2
    assert "sushiruntime" in result.output
    assert "sushiblas" in result.output

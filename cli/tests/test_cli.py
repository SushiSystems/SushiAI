# test_cli.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for sa's entry point and the surfaces it takes from sushicore."""

from __future__ import annotations

import importlib
import json
import sys

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from sushicore.root_options import version_line
from sushiai import cli
from sushiai import console as console_module
from sushiai.services import project

K_VISIBLE_COMMANDS = [
    "build", "clean", "config", "demo mlp", "docs", "docs bundle", "doctor", "env", "inference",
    "link", "run", "setup", "test", "train", "unlink",
]
K_BROKEN_TOML = "tool = [\n"


def test_docs_runs_the_api_build_and_bundle_is_its_subcommand(monkeypatch, tmp_path):
    """`sa docs` still builds the reference, and `sa docs bundle` asks for a release."""
    monkeypatch.setattr(project, "doxygen", lambda: 6)
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["docs"]).exit_code == 6
    assert runner.invoke(cli.app, ["doxygen"]).exit_code == 6
    missing = runner.invoke(cli.app, ["docs", "bundle"])
    assert missing.exit_code == 2
    assert "--release" in missing.output


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """Yields an empty SushiAI checkout as the working directory, with an unbuilt console."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "400")
    importlib.reload(console_module)
    yield tmp_path
    importlib.reload(console_module)


def _described() -> dict:
    """Returns the catalogue `sa --describe` prints."""
    result = CliRunner().invoke(cli.app, ["--describe"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.parametrize("name", ["config.toml", "config.local.toml"])
def test_main_reports_a_malformed_config_as_one_line(name, checkout, monkeypatch, capsys):
    """A config file that is not TOML ends in one line naming it and exit code 1."""
    (checkout / "cli" / name).write_text(K_BROKEN_TOML, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["sa", "config"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    captured = capsys.readouterr()
    lines = [line for line in (captured.out + captured.err).splitlines() if line.strip()]
    assert raised.value.code == 1
    assert len(lines) == 1, lines
    assert name in lines[0] and "Traceback" not in lines[0]


def test_main_exits_zero_when_the_command_succeeds(monkeypatch, capsys):
    """A command that succeeds leaves exit code 0."""
    monkeypatch.setattr(sys, "argv", ["sa", "--version"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == version_line("sushiai-cli")


def test_version_prints_the_distribution_and_its_version():
    """--version prints one line naming sushiai-cli and runs no command."""
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == version_line("sushiai-cli")


def test_describe_lists_every_visible_command():
    """--describe names the program and each command a user can run, sorted."""
    document = _described()
    assert document["program"] == "sa"
    assert [command["name"] for command in document["commands"]] == K_VISIBLE_COMMANDS


def test_describe_leaves_hidden_commands_out():
    """--describe carries no command that the Typer application marks hidden."""
    hidden = [name for name, command in get_command(cli.app).commands.items() if command.hidden]
    described = {command["name"] for command in _described()["commands"]}
    assert hidden and described.isdisjoint(hidden)


@pytest.mark.parametrize("name", ["config", "env"])
def test_the_diagnostic_commands_are_the_ones_sushicore_declares(name):
    """config and env are registered by sushicore, not declared in this package."""
    callback = get_command(cli.app).commands[name].callback
    assert callback.__module__ == "sushicore.diag_commands"


def test_env_takes_the_short_form_of_all():
    """env answers -a as well as --all."""
    env = get_command(cli.app).commands["env"]
    assert {"--all", "-a"} <= {flag for param in env.params for flag in param.opts}

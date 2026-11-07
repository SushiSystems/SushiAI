# test_cli.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for sa's entry point and the surfaces it takes from sushicore."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from recorded_tools import doxygen_ready, recorded_tools, write_executable
from sushiai import cli
from sushiai import console as console_module
from sushiai.services import build_trees
from sushiai.services.profiler import run_profiler
from sushiai.services.project import K_DEFAULT_TYPE

#: The groups sushicore adds whole; their commands are sushicore's to list, so only the group is checked.
K_SUSHICORE_GROUPS = ("backlog", "design")
K_VISIBLE_COMMANDS = [
    "build", "clean", "config", "demo mlp", "docs", "docs bundle", "doctor", "env", "import",
    "inference", "link", "profile", "run", "setup", "test", "tokenize", "train", "unlink",
]
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
K_BROKEN_TOML = "tool = [\n"
K_GIT = (
    "git", "-c", "user.name=tests", "-c", "user.email=tests@example.invalid",
    "-c", "commit.gpgsign=false",
)
K_PUBLISH_LIST = (
    'name = "sample"\n'
    'title = "Sample"\n'
    'summary = "A sample repository."\n'
    'sections = ["guides"]\n'
    'api = false\n'
)
K_ARCHIVE = "docs-bundle-1.2.3.tar.gz"
K_DOXYFILE = Path(__file__).resolve().parents[2] / ".config" / "doxygen" / "Doxyfile"


@pytest.mark.medium
@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_docs_and_its_old_word_return_the_exit_code_of_the_api_build(word, project_root):
    """`sa docs` and the old `sa doxygen` both return the exit code of the reference build."""
    recorded = doxygen_ready(project_root, 6)
    result = CliRunner().invoke(cli.app, [word], obj=recorded.tools)
    assert result.exit_code == 6


@pytest.mark.medium
def test_docs_bundle_asks_for_a_release(project_root):
    """`sa docs bundle` without --release exits 2 and names the option."""
    recorded = doxygen_ready(project_root, 6)
    missing = CliRunner().invoke(cli.app, ["docs", "bundle"], obj=recorded.tools)
    assert missing.exit_code == 2
    assert "--release" in K_ANSI_ESCAPE.sub("", missing.output)


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


@pytest.mark.medium
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


@pytest.mark.medium
def test_main_exits_zero_when_the_command_succeeds(monkeypatch):
    """A command that succeeds leaves exit code 0."""
    monkeypatch.setattr(sys, "argv", ["sa", "--version"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 0


@pytest.mark.medium
def test_version_prints_the_distribution_and_its_version():
    """--version prints one line naming sushiai-cli and runs no command."""
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"sushiai-cli {importlib.metadata.version('sushiai-cli')}"


@pytest.mark.medium
def test_describe_lists_every_visible_command():
    """--describe names the program and each command a user can run."""
    document = _described()
    assert document["program"] == "sa"
    names = [command["name"] for command in document["commands"]]
    assert [name for name in names if name.split()[0] not in K_SUSHICORE_GROUPS] == K_VISIBLE_COMMANDS
    assert {name.split()[0] for name in names} >= set(K_SUSHICORE_GROUPS)


@pytest.mark.medium
def test_env_takes_the_short_form_of_all():
    """env answers -a as well as --all."""
    env = get_command(cli.app).commands["env"]
    assert {"--all", "-a"} <= {flag for param in env.params for flag in param.opts}


def _published_checkout(root: Path) -> None:
    """Writes the CLI's root marker, a one-page docs/ tree and a git commit that holds them."""
    guides = root / "docs" / "guides"
    guides.mkdir(parents=True)
    (root / cli.PROFILE.root_marker).write_text("", encoding="utf-8")
    (root / "docs" / "publish.toml").write_text(K_PUBLISH_LIST, encoding="utf-8")
    (root / "docs" / "README.md").write_text(
        "# Sample\n\n- [Using](guides/USING.md)\n", encoding="utf-8")
    (guides / "USING.md").write_text("# Using\n\nRun it.\n", encoding="utf-8")
    for arguments in (("init", "--quiet"), ("add", "--all"),
                      ("commit", "--quiet", "--message", "docs")):
        subprocess.run([*K_GIT, *arguments], cwd=root, check=True, capture_output=True)


def _doxyfile_value(key: str, default: str) -> str:
    """Returns the value the repository's Doxyfile gives *key*, or *default* when it gives none."""
    for line in K_DOXYFILE.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip().strip('"')
    return default


@pytest.mark.medium
def test_docs_bundle_writes_the_archive_of_the_checkout_it_runs_in(tmp_path, monkeypatch):
    """Writes the archive and its digest file at the checkout's root and reports both lines."""
    _published_checkout(tmp_path)
    monkeypatch.chdir(tmp_path / "docs" / "guides")
    recorded = recorded_tools()
    result = CliRunner().invoke(
        cli.app, ["docs", "bundle", "--release", "1.2.3"], obj=recorded.tools)
    said = recorded.console.texts("success")
    assert result.exit_code == 0, result.output
    bundle = tmp_path / "build" / "docs" / "bundle"
    assert (bundle / K_ARCHIVE).is_file()
    assert (bundle / (K_ARCHIVE + ".sha256")).is_file()
    assert len(said) == 2, said
    assert said[0].endswith(f"{K_ARCHIVE} (1 pages)")
    assert said[1].startswith("sha256 ")


@pytest.mark.medium
def test_the_api_source_reads_the_xml_doxygen_writes_under_the_site_folder(project_root):
    """Names `<OUTPUT_DIRECTORY>/<XML_OUTPUT>` of the Doxyfile, below the repository root."""
    written = Path(_doxyfile_value("OUTPUT_DIRECTORY", "")) / _doxyfile_value("XML_OUTPUT", "xml")
    assert cli._api_source().xml_dir == project_root / written


@pytest.mark.medium
@pytest.mark.parametrize("command, target, positional", [
    ("train", "sushiai_train", ["train.json"]),
    ("inference", "sushiai_inference", ["model.json", "model.sushi", "eval.json"]),
])
def test_train_forwards_profile(project_root, command, target, positional):
    """`--profile <config>` reaches the program whole, wherever it stands on the line."""
    recorded = recorded_tools()
    write_executable(build_trees.build_dir(project_root, K_DEFAULT_TYPE.value) / target)
    runner = CliRunner()

    after = runner.invoke(
        cli.app, [command, *positional, "--profile", "profile.json"], obj=recorded.tools)
    before = runner.invoke(
        cli.app, [command, "--profile", "profile.json", *positional], obj=recorded.tools)

    assert after.exit_code == 0 and before.exit_code == 0, after.output + before.output
    assert [list(ran.argv[1:]) for ran in recorded.runner.commands] == [
        [*positional, "--profile", "profile.json"],
        ["--profile", "profile.json", *positional],
    ]
    assert {Path(ran.argv[0]).stem for ran in recorded.runner.commands} == {target}


@pytest.fixture
def sp_on_path(tmp_path, monkeypatch):
    """Puts a program named `sp` on the path and returns where `which` finds it."""
    folder = tmp_path / "tools"
    write_executable(folder / "sp")
    monkeypatch.setenv("PATH", str(folder))
    return shutil.which("sp")


@pytest.mark.medium
def test_profile_runs_sp(project_root, sp_on_path):
    """`sa profile <args>` runs `sp <args>` from the project root and returns its exit code."""
    recorded = recorded_tools()
    recorded.runner.script(3)

    result = CliRunner().invoke(
        cli.app, ["profile", "record", "show", "run.sprec.json"], obj=recorded.tools)

    assert result.exit_code == 3, result.output
    assert [(list(ran.argv), ran.cwd) for ran in recorded.runner.commands] == [
        ([sp_on_path, "record", "show", "run.sprec.json"], project_root)]


@pytest.mark.small
def test_profile_without_sp_says_to_install_it():
    """A machine with no `sp` on the path gets one error line that names SushiProfiler."""
    recorded = recorded_tools()

    code = run_profiler(recorded.tools, ["record", "show", "run.sprec.json"],
                        which=lambda _program: None)

    assert code == 1
    assert recorded.runner.commands == []
    assert "SushiProfiler" in K_ANSI_ESCAPE.sub("", " ".join(recorded.errors()))

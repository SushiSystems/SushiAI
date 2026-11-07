# test_help_screen.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""sa's help screen groups its commands under three headings and carries examples."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recorded_tools import recorded_tools, write_executable
from sushiai.cli import app
from sushiai.services import build_trees
from sushiai.services.project import K_DEFAULT_TYPE

pytestmark = pytest.mark.medium

K_GROUPS = {
    "Project": ["build", "test", "run", "clean", "docs", "backlog", "design"],
    "Models": ["train", "inference", "import", "tokenize", "profile", "demo"],
    "Diagnostics": ["config", "env", "setup", "doctor", "link", "unlink"],
}

# Commands whose --help is forwarded to the program they run once that program is built.
K_FORWARDED_PROGRAMS = {
    "train": "sushiai_train",
    "inference": "sushiai_inference",
    "import": "sushiai_import",
    "tokenize": "sushiai_tokenize",
}
K_FORWARDING_LINES = [("train",), ("inference",), ("import",), ("tokenize",), ("demo", "mlp")]

K_EXAMPLES = {
    "build": "sa build --type release",
    "test": "sa test --label small",
    "run": "sa run --sort",
    "clean": "sa clean",
    "docs": "sa docs",
    "config": "sa config",
    "env": "sa env --all",
}

K_RAW_COLOUR = re.compile(r"\[/?(green|red|yellow|cyan|blue|dim)[\] ]")
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
K_PACKAGE = Path(__file__).resolve().parents[1] / "sushiai"


def _run(*args: str) -> str:
    """Return the output of one sa invocation, asserting it exited zero."""
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return K_ANSI_ESCAPE.sub("", result.output)


def _commands_by_heading(page: str) -> dict[str, list[str]]:
    """Return the first word of each row, keyed by the heading above it.

    A row starts two columns in; a wrapped description sits deeper and is skipped.
    """
    grouped: dict[str, list[str]] = {}
    heading = ""
    for line in page.splitlines():
        if line and not line.startswith(" "):
            heading = line.rstrip()
            grouped[heading] = []
        elif line.startswith("  ") and not line.startswith("   "):
            grouped[heading].append(line.split()[0])
    return grouped


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Return every string literal in ``path`` with its line number, f-string parts included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_root_help_lists_the_three_groups_in_order():
    lines = [line.rstrip() for line in _run("--help").splitlines()]
    positions = [lines.index(heading) for heading in K_GROUPS]
    assert positions == sorted(positions)


def test_no_command_is_listed_under_a_group_it_does_not_belong_to():
    grouped = _commands_by_heading(_run("--help"))
    for heading, names in K_GROUPS.items():
        assert set(grouped[heading]) == set(names)


def test_a_bare_invocation_prints_the_same_page():
    result = CliRunner().invoke(app, [])
    assert K_ANSI_ESCAPE.sub("", result.output).strip() == _run("--help").strip()


@pytest.mark.parametrize("name", list(K_EXAMPLES))
def test_a_command_help_carries_examples(name):
    out = _run(name, "--help")
    assert "Examples" in out and K_EXAMPLES[name] in out


def test_a_sub_app_draws_its_own_page_with_examples():
    out = _run("demo", "--help")
    assert "Usage: sa demo" in out and "mlp" in out
    assert "Examples" in out and "sa demo mlp --epochs 4 --optimizer sgd" in out


@pytest.mark.parametrize(("name", "target"), sorted(K_FORWARDED_PROGRAMS.items()))
def test_a_forwarding_command_hands_help_to_the_program_it_runs(name, target, project_root):
    """With the program built, `--help` reaches it as its own argument."""
    recorded = recorded_tools()
    write_executable(build_trees.build_dir(project_root, K_DEFAULT_TYPE.value) / target)

    result = CliRunner().invoke(app, [name, "--help"], obj=recorded.tools)

    assert result.exit_code == 0, result.output
    assert [(Path(ran.argv[0]).stem, list(ran.argv[1:])) for ran in recorded.runner.commands] == [
        (target, ["--help"])]


def test_profile_hands_help_to_sp(project_root, tmp_path, monkeypatch):
    """With `sp` on the path, `sa profile --help` reaches it as its own argument."""
    folder = tmp_path / "tools"
    program = write_executable(folder / "sp")
    monkeypatch.setenv("PATH", str(folder))
    recorded = recorded_tools()

    result = CliRunner().invoke(app, ["profile", "--help"], obj=recorded.tools)

    assert result.exit_code == 0, result.output
    assert [list(ran.argv[1:]) for ran in recorded.runner.commands] == [["--help"]]
    assert Path(recorded.runner.commands[0].argv[0]).stem == program.stem


@pytest.mark.parametrize("words", K_FORWARDING_LINES, ids=" ".join)
def test_a_forwarding_command_draws_its_own_page_while_its_program_is_not_built(
        words, tmp_path, monkeypatch):
    """Without a built program to forward to, --help prints the command's page with examples."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [*words, "--help"])
    line = "sa " + " ".join(words)
    page = K_ANSI_ESCAPE.sub("", result.output)
    assert f"Usage: {line}" in page
    assert "Examples" in page and f"{line} " in page.split("Examples")[1]
    assert result.exit_code == 1


@pytest.mark.parametrize("path", sorted(K_PACKAGE.rglob("*.py")), ids=lambda p: p.name)
def test_no_raw_rich_colour_name_is_left_in_the_source(path):
    left = [(line, text) for line, text in _string_literals(path) if K_RAW_COLOUR.search(text)]
    assert left == []

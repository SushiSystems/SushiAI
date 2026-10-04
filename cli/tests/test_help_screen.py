"""sa's help screen groups its commands under three headings and carries examples."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from sushiai.cli import app

K_GROUPS = {
    "Project": ["build", "test", "run", "clean", "doxygen"],
    "Models": ["train", "inference", "demo"],
    "Diagnostics": ["config", "env", "setup", "doctor", "link", "unlink"],
}

# Commands whose --help is forwarded to the program they run, so no page is drawn for them.
K_FORWARDING = {"train", "inference"}

K_EXAMPLES = {
    "build": "sa build --type debug",
    "test": "sa test --suite unit",
    "run": "sa run --sort",
    "clean": "sa clean",
    "doxygen": "sa doxygen",
    "config": "sa config",
    "env": "sa env --all",
}

K_RAW_COLOUR = re.compile(r"\[/?(green|red|yellow|cyan|blue|dim)[\] ]")
K_PACKAGE = Path(__file__).resolve().parents[1] / "sushiai"


def _run(*args: str) -> str:
    """Return the output of one sa invocation, asserting it exited zero."""
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


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


@pytest.mark.parametrize("heading", list(K_GROUPS))
def test_every_command_sits_under_its_own_group(heading):
    listed = _commands_by_heading(_run("--help"))[heading]
    assert [name for name in listed if name in K_GROUPS[heading]] == K_GROUPS[heading]


def test_no_command_is_listed_under_a_group_it_does_not_belong_to():
    grouped = _commands_by_heading(_run("--help"))
    for heading, names in K_GROUPS.items():
        assert set(grouped[heading]) == set(names)


def test_a_bare_invocation_prints_the_same_page():
    result = CliRunner().invoke(app, [])
    assert result.output.strip() == _run("--help").strip()


@pytest.mark.parametrize("name", list(K_EXAMPLES))
def test_a_command_help_carries_examples(name):
    out = _run(name, "--help")
    assert "Examples" in out and K_EXAMPLES[name] in out


def test_a_sub_app_draws_its_own_page_with_examples():
    out = _run("demo", "--help")
    assert "Usage: sa demo" in out and "mlp" in out
    assert "Examples" in out and "sa demo mlp --epochs 4 --optimizer sgd" in out


@pytest.mark.parametrize("name", sorted(K_FORWARDING))
def test_a_forwarding_command_keeps_its_help_for_the_program_it_runs(name):
    command = get_command(app).commands[name]
    assert command.get_help_option_names(command.make_context(name, [])) == []


@pytest.mark.parametrize("path", sorted(K_PACKAGE.rglob("*.py")), ids=lambda p: p.name)
def test_no_raw_rich_colour_name_is_left_in_the_source(path):
    left = [(line, text) for line, text in _string_literals(path) if K_RAW_COLOUR.search(text)]
    assert left == []

# test_aliases.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the old spellings sa still answers to."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sushiai.aliases import K_ALIASES, K_REMOVED_AFTER
from recorded_tools import doxygen_ready
from sushiai.cli import app

pytestmark = pytest.mark.small

K_OLD_SPELLINGS = ["sa doxygen"]
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
K_DOCS_ROW = re.compile(r"^\s+docs\b", re.MULTILINE)
K_DOXYGEN_ROW = re.compile(r"^\s+doxygen\b", re.MULTILINE)


@pytest.fixture
def doxygen_runs(project_root):
    """Returns the recorded tools of a checkout where `doxygen` is found and exits with 7."""
    return doxygen_ready(project_root, 7)


@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_the_old_word_runs_the_new_command(word, doxygen_runs):
    """`sa doxygen` runs what `sa docs` runs and returns its exit code."""
    result = CliRunner().invoke(app, [word], obj=doxygen_runs.tools)
    assert [Path(command.argv[1]).name for command in doxygen_runs.runner.commands] == ["Doxyfile"]
    assert result.exit_code == 7


def test_the_old_word_is_absent_from_the_help_page():
    """The root help lists docs and no doxygen row."""
    page = K_ANSI_ESCAPE.sub("", CliRunner().invoke(app, ["--help"]).output)
    assert K_DOCS_ROW.search(page)
    assert not K_DOXYGEN_ROW.search(page)


def test_the_old_word_is_absent_from_the_catalogue():
    """--describe names docs and not doxygen."""
    document = json.loads(CliRunner().invoke(app, ["--describe"]).output)
    names = [command["name"] for command in document["commands"]]
    assert "docs" in names and "doxygen" not in names


@pytest.mark.parametrize("old", K_OLD_SPELLINGS)
def test_every_row_leaves_on_the_shared_day(old):
    """Each old spelling carries 2027-03-25, the day every Sushi CLI drops its aliases."""
    assert K_REMOVED_AFTER == date(2027, 3, 25)
    assert K_ALIASES.alias_for(old).removed_after == date(2027, 3, 25)


def test_no_row_has_expired_today():
    """No old spelling is past its last day."""
    assert K_ALIASES.expired(date.today()) == []


def test_the_notice_names_the_new_spelling():
    """The notice tells the user to type `sa docs`."""
    assert "`sa docs`" in K_ALIASES.notice(K_ALIASES.alias_for("sa doxygen"))

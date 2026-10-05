# test_aliases.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the old spellings sa still answers to."""

from __future__ import annotations

import json
import re
from datetime import date

import pytest
from typer.testing import CliRunner

from sushiai.aliases import K_ALIASES, K_REMOVED_AFTER
from sushiai.cli import app
from sushiai.services import project

K_OLD_SPELLINGS = ["sa doxygen"]
K_DOCS_ROW = re.compile(r"^\s+docs\b", re.MULTILINE)
K_DOXYGEN_ROW = re.compile(r"^\s+doxygen\b", re.MULTILINE)


@pytest.fixture
def doxygen_runs(monkeypatch):
    """Replaces the service behind `sa docs` and returns the list its calls are recorded in."""
    calls: list[str] = []

    def record() -> int:
        """Records one run and reports the exit code 7."""
        calls.append("doxygen")
        return 7

    monkeypatch.setattr(project, "doxygen", record)
    return calls


@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_the_old_word_runs_the_new_command(word, doxygen_runs):
    """`sa doxygen` runs what `sa docs` runs and returns its exit code."""
    result = CliRunner().invoke(app, [word])
    assert doxygen_runs == ["doxygen"]
    assert result.exit_code == 7


def test_the_old_word_is_absent_from_the_help_page():
    """The root help lists docs and no doxygen row."""
    page = CliRunner().invoke(app, ["--help"]).output
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

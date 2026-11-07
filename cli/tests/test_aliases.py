# test_aliases.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the old spellings sa still answers to."""

from __future__ import annotations

from datetime import date

import pytest

from sushiai.aliases import K_ALIASES

K_OLD_SPELLINGS = ["sa doxygen"]
K_LAST_DAY = date(2027, 3, 25)


@pytest.mark.small
@pytest.mark.parametrize("old", K_OLD_SPELLINGS)
def test_every_row_leaves_on_the_shared_day(old):
    """Each old spelling carries 2027-03-25, the day every Sushi CLI drops its aliases."""
    assert K_ALIASES.alias_for(old).removed_after == date(2027, 3, 25)


@pytest.mark.small
@pytest.mark.parametrize(("today", "expired"), [
    (date(2027, 3, 24), []),
    (K_LAST_DAY, []),
    (date(2027, 3, 26), ["sa doxygen"]),
])
def test_a_row_is_expired_only_after_its_last_day(today, expired):
    """The old spelling works through 2027-03-25 and is expired from 2027-03-26."""
    assert [alias.old for alias in K_ALIASES.expired(today)] == expired


@pytest.mark.small
def test_the_notice_names_the_new_spelling():
    """The notice tells the user to type `sa docs`."""
    assert "`sa docs`" in K_ALIASES.notice(K_ALIASES.alias_for("sa doxygen"))

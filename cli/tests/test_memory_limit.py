# test_memory_limit.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests how the memory ceiling of `sa test` and `sa run` is resolved.

The reasoning is in cli/tests/README.md under "test_memory_limit.py".
"""

from __future__ import annotations

import pytest

from sushiai import memory_limit

pytestmark = pytest.mark.small

K_GB = memory_limit.K_BYTES_PER_GB


def test_default_is_half_of_the_physical_memory() -> None:
    """Checks that no option and no variable give half of the machine's memory."""
    assert memory_limit.resolve_limit_bytes(None, {}, 16 * K_GB) == 8 * K_GB


def test_the_variable_replaces_the_default() -> None:
    """Checks that SA_MEMORY_LIMIT_GB is read when the flag is absent."""
    environment = {memory_limit.K_ENVIRONMENT_KEY: "3"}
    assert memory_limit.resolve_limit_bytes(None, environment, 16 * K_GB) == 3 * K_GB


def test_the_option_wins_over_the_variable() -> None:
    """Checks that --memory-limit is taken before the variable."""
    environment = {memory_limit.K_ENVIRONMENT_KEY: "3"}
    assert memory_limit.resolve_limit_bytes(1.5, environment, 16 * K_GB) == int(1.5 * K_GB)


def test_zero_applies_no_limit() -> None:
    """Checks that a ceiling of 0 resolves to none and applies none."""
    assert memory_limit.resolve_limit_bytes(0.0, {}, 16 * K_GB) == 0
    assert memory_limit.apply_limit(0) is False


def test_an_unknown_machine_size_gives_no_default() -> None:
    """Checks that a machine whose memory cannot be read gets no default ceiling."""
    assert memory_limit.resolve_limit_bytes(None, {}, 0) == 0


def test_a_variable_that_is_no_number_is_refused_with_its_text() -> None:
    """Checks that a malformed variable is refused and named."""
    environment = {memory_limit.K_ENVIRONMENT_KEY: "lots"}
    with pytest.raises(memory_limit.LimitError, match="'lots'"):
        memory_limit.resolve_limit_bytes(None, environment, 16 * K_GB)


def test_a_negative_limit_is_refused() -> None:
    """Checks that a ceiling below zero is refused."""
    with pytest.raises(memory_limit.LimitError, match="at least 0"):
        memory_limit.resolve_limit_bytes(None, {memory_limit.K_ENVIRONMENT_KEY: "-1"}, K_GB)


def test_the_machine_reports_its_memory() -> None:
    """Checks that the physical memory of this machine is read as a positive size."""
    assert memory_limit.physical_memory_bytes() > 0

# test_distributed.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests `sa build --distributed` and the SA_ENABLE_DISTRIBUTED entry it passes to CMake."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from typer.testing import CliRunner

from sushiai import cli
from sushiai.services import distributed, project

pytestmark = pytest.mark.small


def test_distributed_define_switches_the_distributed_layer() -> None:
    """distributed_define gives the SA_ENABLE_DISTRIBUTED entry, or nothing for None."""
    assert distributed.distributed_define(True) == ["SA_ENABLE_DISTRIBUTED=ON"]
    assert distributed.distributed_define(False) == ["SA_ENABLE_DISTRIBUTED=OFF"]
    assert distributed.distributed_define(None) == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--distributed"], True),
        (["build", "--no-distributed"], False),
        (["build"], None),
    ],
)
def test_build_command_forwards_the_distributed_option(
    monkeypatch: pytest.MonkeyPatch, args: list[str], expected: bool | None
) -> None:
    """The build command hands the tri-state --distributed option to the service unchanged."""
    received: list[bool | None] = []

    def build(
        build_type: project.BuildType,
        clean: bool,
        define: Sequence[str] | None,
        sanitize: project.Sanitizer | None,
        distributed_option: bool | None,
    ) -> int:
        """Records the option and reports success."""
        received.append(distributed_option)
        return 0

    monkeypatch.setattr(project, "build", build)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]

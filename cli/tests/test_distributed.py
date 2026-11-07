# test_distributed.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests `sa build --distributed` and the SA_ENABLE_DISTRIBUTED entry it passes to CMake."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from recorded_tools import offline_config, recorded_tools
from sushiai import cli
from sushiai.services import distributed


@pytest.mark.small
def test_distributed_define_switches_the_distributed_layer() -> None:
    """distributed_define gives the SA_ENABLE_DISTRIBUTED entry, or nothing for None."""
    assert distributed.distributed_define(True) == ["SA_ENABLE_DISTRIBUTED=ON"]
    assert distributed.distributed_define(False) == ["SA_ENABLE_DISTRIBUTED=OFF"]
    assert distributed.distributed_define(None) == []


@pytest.mark.medium
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--distributed"], "-DSA_ENABLE_DISTRIBUTED=ON"),
        (["build", "--no-distributed"], "-DSA_ENABLE_DISTRIBUTED=OFF"),
        (["build"], None),
    ],
)
def test_build_command_forwards_the_distributed_option(
    project_root: Path, args: list[str], expected: str | None
) -> None:
    """The build command hands the tri-state --distributed option to the configure unchanged."""
    recorded = recorded_tools(offline_config())
    result = CliRunner().invoke(cli.app, args, obj=recorded.tools)
    assert result.exit_code == 0, result.output
    configure = [command for command in recorded.runner.commands if "-S" in command.argv]
    assert len(configure) == 1
    entries = [word for word in configure[0].argv if word.startswith("-DSA_ENABLE_DISTRIBUTED")]
    assert entries == ([] if expected is None else [expected])

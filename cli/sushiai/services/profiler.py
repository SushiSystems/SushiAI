# profiler.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Runs SushiProfiler's `sp` for `sa profile`, which reads the records `--profile` writes."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

from ..config import find_project_root
from .toolbox import Toolbox

#: The program `sa profile` forwards to; SushiProfiler's own command line.
K_PROFILER_PROGRAM = "sp"

#: Where SushiProfiler's command line comes from.
K_PROFILER_HOME = "https://github.com/SushiSystems/SushiProfiler"


def is_installed() -> bool:
    """Returns whether `sp` is on the path."""
    return shutil.which(K_PROFILER_PROGRAM) is not None


def run_profiler(tools: Toolbox, args: Sequence[str]) -> int:
    """Runs `sp` with *args* from the project root.

    Args:
        tools: The console and the runner the command works through.
        args: The arguments after `sa profile`, forwarded verbatim.

    Returns:
        1 when `sp` is not on the path, otherwise its own exit code.
    """
    console = tools.console
    console.header("SushiProfiler")
    program = shutil.which(K_PROFILER_PROGRAM)
    if program is None:
        console.error(
            f"`{K_PROFILER_PROGRAM}` is not on the path. Install SushiProfiler's command line "
            f"({K_PROFILER_HOME}) and open a new shell.")
        return 1

    console.info(f"Executing: {K_PROFILER_PROGRAM}")
    return tools.runner.run([program, *args], find_project_root(), dict(os.environ))

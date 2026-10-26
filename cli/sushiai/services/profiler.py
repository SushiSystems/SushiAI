# profiler.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs SushiProfiler's `sp` for `sa profile`, which reads the records `--profile` writes."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

from sushicore.proc import Runner

from .. import console
from ..config import PROFILE, find_project_root

#: The program `sa profile` forwards to; SushiProfiler's own command line.
K_PROFILER_PROGRAM = "sp"

#: Where SushiProfiler's command line comes from.
K_PROFILER_HOME = "https://github.com/SushiSystems/SushiProfiler"

_RUNNER = Runner(console, PROFILE.program, catch_interrupt=True)


def is_installed() -> bool:
    """Returns whether `sp` is on the path."""
    return shutil.which(K_PROFILER_PROGRAM) is not None


def run_profiler(args: Sequence[str]) -> int:
    """Runs `sp` with *args* from the project root.

    Args:
        args: The arguments after `sa profile`, forwarded verbatim.

    Returns:
        1 when `sp` is not on the path, otherwise its own exit code.
    """
    console.header("SushiProfiler")
    program = shutil.which(K_PROFILER_PROGRAM)
    if program is None:
        console.error(
            f"`{K_PROFILER_PROGRAM}` is not on the path. Install SushiProfiler's command line "
            f"({K_PROFILER_HOME}) and open a new shell.")
        return 1

    console.info(f"Executing: {K_PROFILER_PROGRAM}")
    return _RUNNER.run([program, *args], find_project_root(), dict(os.environ))

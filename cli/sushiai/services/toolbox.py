# toolbox.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Holds the console, the process runner, the CMake driver and the config sources a service uses.

The reasoning is in cli/README.md under "services/toolbox.py".
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sushicore.cmake_driver import CMakeDriver
from sushicore.proc import CommandRunner, Runner

from ..config import PROFILE, Config
from .service_console import ServiceConsole


@dataclass(frozen=True, slots=True)
class Toolbox:
    """What a service reaches the outside through, so that a test hands it recorders.

    Attributes:
        console: Receives every line the service prints.
        runner: Runs every external command.
        driver: Builds the cmake and ctest command lines and runs them through *runner*.
        load_config: Returns the resolved configuration of this machine.
        load_build_env: Returns the environment a build of a tree runs under.
    """

    console: ServiceConsole
    runner: CommandRunner
    driver: CMakeDriver
    load_config: Callable[[], Config]
    load_build_env: Callable[[Config, Path], dict[str, str]]


@functools.cache
def default_toolbox() -> Toolbox:
    """Returns the toolbox of a real run: the lazy console, a process runner and the loaded config."""
    from .. import console
    from ..config import load_config
    from ..env import load_build_env

    runner = Runner(console, PROFILE.program, catch_interrupt=True)
    return Toolbox(
        console=console,
        runner=runner,
        driver=CMakeDriver(console, runner),
        load_config=load_config,
        load_build_env=load_build_env,
    )


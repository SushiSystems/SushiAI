# env.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the environment that build, test and run subprocesses inherit.

The reasoning is in cli/README.md under "env.py".
"""

from __future__ import annotations

from pathlib import Path

from sushicore.build_env import StackBuildEnv

from . import console
from .config import Config, PROFILE, find_project_root

_ENV = StackBuildEnv(profile=PROFILE, console=console, find_root=find_project_root)


def load_build_env(cfg: Config, build_dir: Path) -> dict[str, str]:
    """Return the environment for build/test/run subprocesses."""
    return _ENV.load(cfg, build_dir)

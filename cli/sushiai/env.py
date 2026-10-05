# env.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Build-environment snapshotting for the SushiAI CLI.

A parent process cannot ``call vcvars64.bat`` and inherit the result, so the
shell is run as a child and its environment dumped -- see
:mod:`sushicore.build_env` for the mechanism and the cache.

SushiAI selects no SYCL toolchain of its own: it builds with the intel/llvm
clang++ in the shared SushiStack dependency tree. That makes its environment
work exactly :class:`sushicore.build_env.StackBuildEnv` -- load MSVC on Windows,
put the shared toolchain's bin/lib on PATH -- with the sibling checkouts in the
cache key coming from PROFILE rather than being listed here again.
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

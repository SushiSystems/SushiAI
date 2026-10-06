# diag.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Wires SushiAI's profile, config module and build environment into the shared diagnostics.

The reasoning is in cli/README.md under "services/diag.py".
"""

from __future__ import annotations

from pathlib import Path

from sushicore.diag import Diagnostics

from .. import config as config_module
from .. import console
from ..config import PROFILE


def _build_env(cfg, build_dir: Path):
    """The environment a build runs under.

    Imported here rather than at module scope: ``env`` imports ``config``, which
    would close a cycle at module load.
    """
    from ..env import load_build_env

    return load_build_env(cfg, build_dir)


DIAGNOSTICS = Diagnostics(
    profile=PROFILE,
    console=console,
    config=config_module,
    build_env=_build_env,
)

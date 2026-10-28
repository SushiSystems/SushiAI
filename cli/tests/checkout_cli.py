# checkout_cli.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs this checkout's CLI in a child process, never a `sa` or `sushiai` found on PATH."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sushiai.services import build_trees
from sushiai.services.project import K_DEFAULT_TYPE

# cli/tests/checkout_cli.py -> cli/tests -> cli
K_CLI_ROOT = Path(__file__).resolve().parents[1]

#: The tree `sa build` makes with no option, which the commands that run a program read.
K_DEFAULT_TREE = build_trees.build_dir(K_CLI_ROOT.parent, K_DEFAULT_TYPE.value)


def run_cli(*args: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    """Runs `python -m sushiai <args>` from *cwd* with this checkout's `cli/` first on the path.

    The output is decoded as UTF-8, replacing what does not decode.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(K_CLI_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-m", "sushiai", *args], cwd=cwd, env=env, capture_output=True,
        encoding="utf-8", errors="replace", timeout=timeout)

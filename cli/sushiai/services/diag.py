"""Diagnostic commands: inspect the resolved config and the build environment.

Read-only troubleshooting. When a build picks the "wrong" compiler or a path
looks off, `sa config` answers *what* the CLI resolved and *where each value
came from*, and `sa env` answers *which environment* cmake/ctest run under.

Both are :class:`sushicore.diag.Diagnostics`; what this file supplies is the
three things that are actually SushiAI's -- its profile, its config module, and
how it finds its build directory.
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


_DIAG = Diagnostics(
    profile=PROFILE,
    console=console,
    config=config_module,
    build_env=_build_env,
)

config_show = _DIAG.config_show
env_dump = _DIAG.env_dump

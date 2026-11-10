# config.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Loads the layered configuration of the SushiAI CLI.

Precedence, low to high: built-in defaults, config.toml, config.local.toml, SA_* variables.
The reasoning is in cli/README.md under "config.py".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sushicore.module_config import ModuleConfig
from sushicore.profile import ModuleProfile
from sushicore.stack_config import StackConfig
from sushicore.workspace import walk_up

#: Describes SushiAI to the shared machinery; every consumer reads this one profile.
PROFILE = ModuleProfile(
    name="SushiAI",
    program="sa",
    env_prefix="SA",
    root_marker="CMakeLists.txt",
    siblings=("sushiruntime", "sushiblas"),
    default_target="sushiai_example",
    # These are unprefixed because every CLI that builds the siblings reads the same ones.
    extra_env_overrides={
        "sushiruntime_dir": "SUSHIRUNTIME_DIR",
        "sushiblas_dir": "SUSHIBLAS_DIR",
    },
    # SYCL, MSVC's INCLUDE/LIB, and every SUSHI* checkout override.
    env_of_interest=("SYCL", "INCLUDE", "LIB", "SUSHI"),
)

_MODULE = ModuleConfig(PROFILE)

config_dir = _MODULE.config_dir


def find_project_root(start: Path | None = None,
                      is_root: Callable[[Path], bool] | None = None) -> Path:
    """Returns the nearest ancestor of *start* (the working directory by default) that is a root.

    @param start Where the walk begins.
    @param is_root Tells whether a folder is the project root. The default is the check of
        sushicore, which looks for the root marker on disk; a test passes its own to walk a
        tree it never writes.
    @raise SystemExit when no ancestor is a root.
    """
    if is_root is None:
        return _MODULE.find_project_root(start)
    root = walk_up(start or Path.cwd(), is_root)
    if root is None:
        raise SystemExit(PROFILE.not_a_project_message())
    return root

#: Maps each field to its environment variable; `sa config` reads the reverse map.
_ENV_OVERRIDES = PROFILE.env_overrides()


@dataclass
class Config(StackConfig):
    """Resolved, platform-specific tool configuration for the SushiAI build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths) from
    :class:`ToolConfig` and the shared-tree toolchain resolution from
    :class:`StackConfig`. ``cxx`` empty means "discover it": the bundled clang++
    under the shared toolchain if present, else a clang++ on PATH. ``sushiruntime_dir`` and
    ``sushiblas_dir`` name the checkouts the packages of those libraries are built from; empty
    means the sibling checkout or the copy ``sa setup`` fetches.
    """

    sushiruntime_dir: str = ""

    sushiblas_dir: str = ""

    # Run defaults.
    target_bin: str = "sushiai_example"


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    return _MODULE.load(Config)

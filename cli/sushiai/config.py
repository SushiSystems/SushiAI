# config.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the layered configuration of the SushiAI CLI.

Precedence, low to high: built-in defaults, config.toml, config.local.toml, SA_* variables.
The reasoning is in cli/README.md under "config.py".
"""

from __future__ import annotations

from dataclasses import dataclass

from sushicore.module_config import ModuleConfig
from sushicore.profile import ModuleProfile
from sushicore.stack_config import StackConfig

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

find_project_root = _MODULE.find_project_root
config_dir = _MODULE.config_dir

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

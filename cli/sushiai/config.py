# config.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the layered configuration of the SushiAI CLI.

Precedence, low to high: built-in defaults, config.toml, config.local.toml, SA_* variables.
The reasoning is in cli/README.md under "config.py".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    under the shared toolchain if present, else a clang++ on PATH.
    """

    # Empty means ../sushiruntime under the project root, as CMakeLists.txt defaults it.
    sushiruntime_dir: str = ""

    # Empty means ../sushiblas under the project root, as CMakeLists.txt defaults it.
    sushiblas_dir: str = ""

    # Run defaults.
    target_bin: str = "sushiai_example"


    def runtime_dir(self, root: Path) -> Path:
        """Resolve the SushiRuntime checkout this build links against.

        @param root The project root.
        @return The runtime directory, defaulting to ``<root>/../sushiruntime``
                (which, inside a SushiStack workspace, is the sibling module).
        """
        return self.sibling_dir(root, "sushiruntime", self.sushiruntime_dir)

    def blas_dir(self, root: Path) -> Path:
        """Resolve the SushiBLAS checkout SushiAI builds against.

        @param root The SushiAI project root.
        @return The SushiBLAS directory, defaulting to ``<root>/../sushiblas``.
        """
        return self.sibling_dir(root, "sushiblas", self.sushiblas_dir)


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    return _MODULE.load(Config)

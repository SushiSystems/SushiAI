"""Layered configuration loading for the SushiAI CLI.

Precedence (low to high):

    built-in defaults -> config.toml -> config.local.toml -> SA_* env vars

SushiAI is a head of the stack, like SushiEngine: it consumes the shared
SushiStack SYCL toolchain and vcpkg tree rather than provisioning its own.
Unlike SushiEngine it depends on TWO siblings directly, SushiRuntime and
SushiBLAS, so both get their own resolved directory below.

Locating the project and layering the files is
:class:`sushicore.module_config.ModuleConfig`'s job; resolving the compiler and
vcpkg out of the shared dependency tree is
:class:`sushicore.stack_config.StackConfig`'s. What is left here is what is
actually SushiAI's: its identity, and the siblings it builds against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sushicore.module_config import ModuleConfig
from sushicore.profile import ModuleProfile
from sushicore.stack_config import StackConfig

#: How SushiAI describes itself to the shared machinery. Stated once here and
#: read by config loading, the console, executable discovery and diagnostics --
#: which is the point: they cannot disagree about what this module is.
PROFILE = ModuleProfile(
    name="SushiAI",
    program="sa",
    env_prefix="SA",
    root_marker="CMakeLists.txt",
    siblings=("sushiruntime", "sushiblas"),
    default_target="sushiai_example",
    # Not prefixed: these name the same checkouts for every CLI that builds
    # against them, so they must read the same variables those CLIs do.
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

#: Field -> environment variable, derived from the profile's prefix. Exposed as
#: a module attribute because `sa config` reports which values came from the
#: environment and needs the reverse map.
_ENV_OVERRIDES = PROFILE.env_overrides()


@dataclass
class Config(StackConfig):
    """Resolved, platform-specific tool configuration for the SushiAI build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths) from
    :class:`ToolConfig` and the shared-tree toolchain resolution from
    :class:`StackConfig`. ``cxx`` empty means "discover it": the bundled clang++
    under the shared toolchain if present, else a clang++ on PATH.
    """

    # The SushiRuntime sibling checkout. Empty means "../sushiruntime" relative
    # to the project root (matching SUSHIRUNTIME_DIR's default in CMakeLists.txt).
    sushiruntime_dir: str = ""

    # The SushiBLAS sibling checkout. Empty means "../sushiblas" relative to the
    # project root (matching SUSHIBLAS_DIR's default in CMakeLists.txt).
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


    def standalone_deps_dir(self, root: Path) -> Path:
        """Outside a workspace, fall back to the runtime sibling's own bundle.

        SushiRuntime is the ultimate owner of the toolchain bundle, so a
        standalone checkout borrows it from the runtime it builds against
        rather than provisioning a second copy.
        """
        return self.runtime_dir(root) / "dependencies"


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    return _MODULE.load(Config)

# library_packages.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Takes the Sushi libraries `sa` links as built packages and puts their run-time folders on the path.

The reasoning is in cli/README.md under "services/library_packages.py".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sushicore.provision.sushi_packages import PackageSet, RunDirs
from sushicore.provision.sushi_packages import configure_arguments  # noqa: F401  (callers take it from here)

from ..config import Config

#: The file of a build tree that names the run-time folders of the packages it linked.
K_RUN_DIRS_FILE = ".sushiai_package_dirs.json"

#: The libraries whose package is built with the distributed layer when SushiAI is.
K_DISTRIBUTED_LIBRARIES = ("sushiruntime", "sushiblas")
K_DISTRIBUTED_FLAG = "--distributed"

#: The build types whose library packages are built as release.
K_RELEASE_PACKAGE_TYPES = ("asan", "minsizerel")
K_PACKAGE_BUILD_TYPE = "release"


def package_build_type(build_type: str) -> str:
    """Return the build type the library packages are built in for a SushiAI *build_type*.

    A library CLI's ``--type`` does not take ``asan`` or ``minsizerel``, so a build of either
    links release packages.
    """
    return K_PACKAGE_BUILD_TYPE if build_type in K_RELEASE_PACKAGE_TYPES else build_type


def package_options(distributed: bool) -> Mapping[str, tuple[str, ...]]:
    """Return the extra `build` arguments each library takes for this SushiAI build.

    The distribution module needs the distributed layer of SushiRuntime and the cluster module of
    SushiBLAS, so ``--distributed`` of SushiAI becomes ``--distributed`` of both packages.
    """
    if not distributed:
        return {}
    return {name: (K_DISTRIBUTED_FLAG,) for name in K_DISTRIBUTED_LIBRARIES}


def resolve(cfg: Config, root: Path, build_type: str, compiler: str, distributed: bool,
            report: Callable[[str], None], warn: Callable[[str], None]) -> PackageSet:
    """Return a built package of SushiLog, SushiProfiler, SushiRuntime and SushiBLAS.

    A missing package is built once through the library's own CLI.

    Args:
        cfg: The resolved configuration.
        root: The project root.
        build_type: The build type of this configure, such as ``release``.
        compiler: The compiler of this build; the packages are built with it.
        distributed: Whether the distribution module is built.
        report: Receives one ``reused`` or ``built`` line per library.
        warn: Receives the warnings of the resolver.

    Raises:
        sushicore.provision.sushi_packages.PackageError: A package cannot be chosen.
    """
    return cfg.sushi_packages(root, package_build_type(build_type), package_options(distributed),
                              report=report, warn=warn, compiler=compiler)


#: The run-time folders of the packages a tree linked, kept in the tree for the commands that run it.
_RUN_DIRS = RunDirs(K_RUN_DIRS_FILE)


def record_run_dirs(tree: Path, dirs: Sequence[str]) -> None:
    """Write the run-time folders of the packages *tree* linked, for the commands that run it."""
    _RUN_DIRS.save(tree, dirs)


def read_run_dirs(tree: Path) -> list[str]:
    """Return the folders :func:`record_run_dirs` wrote for *tree*, none when it wrote none."""
    return _RUN_DIRS.read(tree)


def prepend_run_dirs(cfg: Config, env: dict[str, str], dirs: Sequence[str]) -> None:
    """Put *dirs* first on the search path of *env*, where a binary finds the package libraries."""
    _RUN_DIRS.prepend(env, dirs, cfg.is_windows)

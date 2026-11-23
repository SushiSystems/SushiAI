# library_packages.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Takes the Sushi libraries `sa` links as built packages and puts their run-time folders on the path.

The reasoning is in cli/README.md under "services/library_packages.py".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sushicore.build_tree import package_build_type
from sushicore.provision.sushi_packages import PackageSet, RunDirs

from ..config import Config
from .sanitizer import Sanitizer
from .sycl_targets import sycl_targets_define

#: The file of a build tree that names the run-time folders of the packages it linked.
K_RUN_DIRS_FILE = ".sushiai_package_dirs.json"

#: The libraries whose package is built with the distributed layer when SushiAI is.
K_DISTRIBUTED_LIBRARIES = ("sushiruntime", "sushiblas")
K_DISTRIBUTED_FLAG = "--distributed"

#: The library that takes the device target list, and the flag that carries it.
K_BLAS = "sushiblas"
K_DEFINE_FLAG = "-D"

#: The library whose CLI takes ThreadSanitizer as its build type rather than as a flag.
K_RUNTIME = "sushiruntime"

#: The libraries whose CLI takes the sanitizer as ``--sanitize thread``.
K_SANITIZE_FLAG_LIBRARIES = ("sushilog", "sushiprofiler", "sushiblas")

#: The package build type of a ThreadSanitizer tree; TSan needs the symbols to name a race.
K_THREAD_PACKAGE_TYPE = "relwithdebinfo"


def package_options(distributed: bool, sycl_targets: str = "",
                    sanitizer: Sanitizer | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return the extra `build` arguments each library takes for this SushiAI build.

    The distribution module needs the distributed layer of SushiRuntime and the cluster module of
    SushiBLAS, so ``--distributed`` of SushiAI becomes ``--distributed`` of both packages. A
    device target list becomes ``-D SB_SYCL_TARGETS=<list>`` of the SushiBLAS package, so another
    list is another package and an equal one is reused. A ThreadSanitizer tree needs the four
    packages built with it, because TSan cannot see a synchronisation edge that runs through an
    uninstrumented library: SushiLog, SushiProfiler and SushiBLAS take ``--sanitize thread`` and
    SushiRuntime takes ``--type tsan``, which follows the ``--type`` the builder already passed.
    """
    options: dict[str, tuple[str, ...]] = {}
    if distributed:
        options.update({name: (K_DISTRIBUTED_FLAG,) for name in K_DISTRIBUTED_LIBRARIES})
    if sycl_targets:
        define = (K_DEFINE_FLAG, *sycl_targets_define(sycl_targets))
        options[K_BLAS] = (*options.get(K_BLAS, ()), *define)
    if sanitizer is Sanitizer.thread:
        for name in K_SANITIZE_FLAG_LIBRARIES:
            options[name] = (*options.get(name, ()), "--sanitize", "thread")
        options[K_RUNTIME] = (*options.get(K_RUNTIME, ()), "--type", "tsan")
    return options


def package_type(build_type: str, sanitizer: Sanitizer | None) -> str:
    """Return the build type the packages of a tree are built as.

    A ThreadSanitizer tree takes :data:`K_THREAD_PACKAGE_TYPE`; any other tree takes
    :func:`sushicore.build_tree.package_build_type` of its type.
    """
    if sanitizer is Sanitizer.thread:
        return K_THREAD_PACKAGE_TYPE
    return package_build_type(build_type)


def resolve(cfg: Config, root: Path, build_type: str, compiler: str, distributed: bool,
            sycl_targets: str, report: Callable[[str], None],
            warn: Callable[[str], None], sanitizer: Sanitizer | None = None) -> PackageSet:
    """Return a built package of SushiLog, SushiProfiler, SushiRuntime and SushiBLAS.

    A missing package is built once through the library's own CLI.

    Args:
        cfg: The resolved configuration.
        root: The project root.
        build_type: The build type of this configure, such as ``release``.
        compiler: The compiler of this build; the packages are built with it.
        distributed: Whether the distribution module is built.
        sycl_targets: The ahead-of-time device targets of the SushiBLAS package, or ''.
        report: Receives one ``reused`` or ``built`` line per library.
        warn: Receives the warnings of the resolver.
        sanitizer: The sanitizer set of this tree; a ThreadSanitizer tree takes instrumented packages.

    Raises:
        sushicore.provision.sushi_packages.PackageError: A package cannot be chosen.
    """
    return cfg.sushi_packages(root, package_type(build_type, sanitizer),
                              package_options(distributed, sycl_targets, sanitizer),
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

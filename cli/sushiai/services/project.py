# project.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs SushiAI's build, test, run, demo, clean and docs commands through cmake and ctest.

The reasoning is in cli/README.md under "services/project.py".
"""

from __future__ import annotations

import enum
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from sushicore.cmake_cache import cached_value
from sushicore.cmake_driver import CMakeDriver
from sushicore.proc import Runner
from sushicore.toolchain_args import c_compiler_for as _resolved_c_compiler
from sushicore.toolchain_args import vcpkg_prefix as _vcpkg_prefix

from .. import console
from .. import memory_limit
from ..config import Config, PROFILE, find_project_root, load_config
from ..env import load_build_env
from . import discovery


class BuildType(str, enum.Enum):
    release = "release"
    debug = "debug"
    relwithdebinfo = "relwithdebinfo"


class Suite(str, enum.Enum):
    # These are the CTest labels tests/CMakeLists.txt assigns from the GTest suite prefix.
    unit = "unit"
    regression = "regression"
    integration = "integration"
    functional = "functional"   # umbrella: unit + regression + integration
    all = "all"

    # package is neither a CTest label nor part of all; see test_package.
    package = "package"


# Suite.all and Suite.package are absent on purpose; see cli/README.md.
_SUITE_LABEL_REGEX = {
    Suite.unit: "^unit$",
    Suite.regression: "^regression$",
    Suite.integration: "^integration$",
    Suite.functional: "^(unit|integration|regression)$",
}

# Each demo name maps to an executable of its own; see cli/README.md.
_DEMO_TARGETS = {
    "mlp": "sushiai_demo_mlp",
}

_CMAKE_BUILD_TYPE = {
    BuildType.release: "Release",
    BuildType.debug: "Debug",
    BuildType.relwithdebinfo: "RelWithDebInfo",
}


class DefineError(ValueError):
    """Reports a -D entry that is not VAR=VALUE."""


def _build_dir(root: Path) -> Path:
    """Returns the build tree of the project at *root*."""
    return root / "build"


_RUNNER = Runner(console, PROFILE.program, catch_interrupt=True)
_DRIVER = CMakeDriver(console, _RUNNER)


def _check_siblings(cfg: Config, root: Path) -> int:
    """Fail early (with guidance) when a sibling checkout is missing.

    SushiAI cannot configure without both: SushiRuntime provides the SYCL
    target CMake pulls in, SushiBLAS is where all compute is drawn from.
    """
    runtime = cfg.runtime_dir(root)
    if not (runtime / "CMakeLists.txt").is_file():
        console.error(
            f"SushiRuntime sibling not found at: {runtime}\n"
            f"  - clone it next to this repo, or\n"
            f"  - set sushiruntime_dir in cli/config.local.toml, or SUSHIRUNTIME_DIR in the env.")
        return 1
    blas = cfg.blas_dir(root)
    if not (blas / "CMakeLists.txt").is_file():
        console.error(
            f"SushiBLAS sibling not found at: {blas}\n"
            f"  - clone it next to this repo, or\n"
            f"  - set sushiblas_dir in cli/config.local.toml, or SUSHIBLAS_DIR in the env.")
        return 1
    return 0


def _toolchain_args(cfg: Config, root: Path) -> list[str]:
    """The compiler and host-tool flags every configure in this tree must share.

    Used by the project's own configure and by the package consumer's (see
    :func:`test_package`). They are factored out rather than written twice
    because the consumer configure is precisely where a missing one costs an
    hour to diagnose: the public headers need the SYCL-aware clang++ to parse
    at all, clang-cl's CMake platform module insists on a resource compiler,
    and vcpkg's toolchain file is what puts hwloc where find_package can see
    it. A copy of this list would drift the moment one of the two configures
    gained a flag, and the package suite would then start failing for reasons
    that have nothing to do with packaging.

    CMAKE_PREFIX_PATH is deliberately absent: it is the one setting the two
    configures legitimately disagree on, since the consumer must also see the
    install prefix. Each caller sets its own. So are SUSHIRUNTIME_DIR and
    SUSHIBLAS_DIR, for a stronger reason - pointing the consumer at the sibling
    source trees is exactly the shortcut the package suite exists to forbid.

    @param cfg The resolved configuration.
    @param root The project root.
    @return The shared -D flags, in the order cmake should receive them.
    """
    cxx = cfg.resolved_compiler(root)
    args = [
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_C_COMPILER={_resolved_c_compiler(cxx)}",
    ]
    if cfg.is_windows:
        if cfg.ninja_exe:
            args.append(f"-DCMAKE_MAKE_PROGRAM={cfg.expand(cfg.ninja_exe)}")
        if cfg.pkgconf_exe:
            args.append(f"-DPKG_CONFIG_EXECUTABLE={cfg.expand(cfg.pkgconf_exe)}")
        if cfg.rc_exe:
            args.append(f"-DCMAKE_RC_COMPILER={cfg.expand(cfg.rc_exe)}")

    vcpkg = cfg.resolved_vcpkg(root)
    if vcpkg:
        args += [
            f"-DCMAKE_TOOLCHAIN_FILE={vcpkg}/scripts/buildsystems/vcpkg.cmake",
            f"-DVCPKG_ROOT={vcpkg}",
        ]
        if cfg.is_windows:
            args.append(f"-DVCPKG_TARGET_TRIPLET={cfg.vcpkg_triplet}")

    return args


def _define_args(defines: Sequence[str] | None) -> list[str]:
    """Returns one -D argument per entry of *defines*.

    Raises:
        DefineError: An entry is not VAR=VALUE.
    """
    for entry in defines or ():
        if "=" not in entry:
            raise DefineError(
                f"-D expects VAR=VALUE, got {entry!r}. "
                "Use VAR= to clear a variable.")
    return [f"-D{entry}" for entry in defines or ()]


def _remove_tree(path: Path) -> bool:
    """Removes the directory *path* and returns whether it is gone, reporting a failure."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return True
    except OSError as error:
        console.error(
            f"Could not remove {path}: {error}. "
            "Close the program that holds it open and run the command again.")
        return False
    return True


def _clean_build_tree(build_dir: Path) -> int:
    """Removes *build_dir* through the driver and returns 1 when it is still there."""
    _DRIVER.clean_tree(build_dir)
    if build_dir.exists():
        console.error(
            f"Could not remove {build_dir}. "
            "Close the program that holds it open and run the command again.")
        return 1
    return 0


def _configure_args(cfg: Config, root: Path, build_dir: Path, build_type: str,
                    define_args: Sequence[str] = ()) -> list[str]:
    """Returns the cmake configure command, with *define_args* after every default."""
    args = [
        _DRIVER.cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DSUSHIRUNTIME_DIR={cfg.runtime_dir(root)}",
        f"-DSUSHIBLAS_DIR={cfg.blas_dir(root)}",
    ]
    args += _toolchain_args(cfg, root)
    if cfg.is_windows and (prefix := _vcpkg_prefix(cfg, root)):
        args.append(f"-DCMAKE_PREFIX_PATH={prefix}")

    # A caller's -D comes last, so no default set above outranks it.
    return args + list(define_args)


def build(build_type: BuildType, clean: bool = False,
          defines: Sequence[str] | None = None) -> int:
    """Configures and builds the project; a malformed -D entry returns 2 with build/ untouched."""
    console.header("Project Build")
    try:
        define_args = _define_args(defines)
    except DefineError as error:
        console.error(str(error))
        return 2
    root = find_project_root()
    cfg = load_config()
    cmake_build_type = _CMAKE_BUILD_TYPE[build_type]
    build_dir = _build_dir(root)

    if (rc := _check_siblings(cfg, root)) != 0:
        return rc

    if clean and (rc := _clean_build_tree(build_dir)) != 0:
        return rc

    console.info(f"Compiler: {cfg.resolved_compiler(root)}")
    console.info(f"SushiRuntime: {cfg.runtime_dir(root)}")
    console.info(f"SushiBLAS:    {cfg.blas_dir(root)}")

    env = load_build_env(cfg, build_dir)

    fresh = _DRIVER.needs_configure(build_dir, cfg.generator, root=root,
                                    expect={"CMAKE_BUILD_TYPE": cmake_build_type})
    if fresh:
        console.info(f"Configuring CMake... (type={build_type.value})")
        if not _remove_tree(build_dir):
            return 1
    else:
        console.info("Reconfiguring in place...")
    configure_args = _configure_args(cfg, root, build_dir, cmake_build_type, define_args)

    rc = _DRIVER.configure(configure_args, cwd=root, env=env)
    if rc != 0:
        console.error("CMake configure failed.")
        return rc
    build_dir.mkdir(parents=True, exist_ok=True)

    console.info("Building...")
    rc = _DRIVER.compile(cfg, build_dir, cwd=root, env=env, config=cmake_build_type)
    if rc == 0:
        console.success("Build completed successfully!")
    else:
        console.error("Build failed.")
    return rc


def _enter_memory_limit(limit_gb: float | None) -> bool:
    """Applies the memory ceiling of a test or a run; returns False when it is refused."""
    try:
        limit = memory_limit.resolve_limit_bytes(
            limit_gb, dict(os.environ), memory_limit.physical_memory_bytes())
        if memory_limit.apply_limit(limit):
            console.info(f"Memory limit: {limit / memory_limit.K_BYTES_PER_GB:.1f} GB")
    except memory_limit.LimitError as error:
        console.error(str(error))
        return False
    return True


def test(suite: Suite = Suite.functional, filter: str | None = None, repeat: int = 0,
         memory_limit_gb: float | None = None) -> int:
    """Run the test suite via CTest labels.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    the binary directly: sa run sushiai_tests -- --gtest_filter=Version.*

    ``package`` is the one suite that is not a CTest label; it delegates to
    :func:`test_package`, which builds a separate project.
    """
    console.header("Project Test")
    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        console.error("build/ not found. Run `sa build` first.")
        return 1

    env = load_build_env(cfg, build_dir)
    if not _enter_memory_limit(memory_limit_gb):
        return 2

    if suite is Suite.package:
        if filter or repeat:
            # Both are ctest knobs, and this suite runs no ctest.
            console.error("--filter and --repeat do not apply to the package suite.")
            return 2
        return test_package(cfg, root, build_dir, env)

    if suite is Suite.all:
        # all covers every CTest test, which leaves the package suite out.
        console.info(
            "'all' covers every CTest test. The package suite is not among them "
            "(it builds a separate project): run `sa test --suite package`.")

    return _DRIVER.ctest_run(
        cfg, build_dir, env,
        label_regex=None if suite is Suite.all else _SUITE_LABEL_REGEX[suite],
        filter=filter, repeat=repeat)


def _deploy_consumer_dlls(cfg: Config, root: Path, prefix: Path, dest: Path) -> bool:
    """Put the shared libraries the consumer needs beside its executable.

    In-tree targets get these from a POST_BUILD copy (see tests/CMakeLists.txt
    and applications/examples/CMakeLists.txt); a real out-of-tree consumer would deploy them
    itself, and this function is this CLI standing in for that consumer.
    Without them the binary exits with 0xC0000135 before reaching main, which
    looks like a package failure and is not one.

    Two sources, and the second one is the easy mistake. Everything in the
    prefix's bin/ is taken rather than a named list, since the prefix holds
    exactly what this project installed. hwloc is then taken from the vcpkg
    tree on top: SushiRuntime links it PRIVATE, so it is absent from the CMake
    *interface* and a consumer never names it - but PRIVATE is a link-time
    statement, not a load-time one, and hwloc-15.dll is in sushiruntime.dll's
    import table all the same. The loader does not read
    INTERFACE_LINK_LIBRARIES.

    @param cfg The resolved configuration.
    @param root The project root.
    @param prefix The install prefix the package was installed to.
    @param dest The directory holding the consumer executable.
    @return False when a library could not be copied; the failure is already reported.
    """
    if not cfg.is_windows:
        return True  # ELF consumers resolve via RPATH; nothing to stage.

    sources = list((prefix / "bin").glob("*.dll"))
    if vcpkg_prefix := _vcpkg_prefix(cfg, root):
        sources += list(Path(vcpkg_prefix, "bin").glob("hwloc*.dll"))

    for dll in sources:
        try:
            shutil.copy2(dll, dest / dll.name)
        except OSError as error:
            console.error(
                f"Could not copy {dll.name} to {dest}: {error}. "
                "The consumer was not run.")
            return False
    console.info(f"Deployed {len(sources)} shared libraries beside the consumer.")
    return True


def _consumer_run_env(cfg: Config, root: Path, env: dict[str, str]) -> dict[str, str]:
    """The build env with the vcpkg installed bin removed from PATH.

    ``load_build_env`` prepends that directory precisely so sushiruntime.dll
    can find hwloc (see env.py). That is right for every in-tree binary and
    wrong for this one: with it on PATH the consumer loads whether or not
    :func:`_deploy_consumer_dlls` staged anything, so the deployment half of
    this suite would assert nothing and pass. A third party has no vcpkg tree
    on PATH, and the point of the package suite is to stop being a third party
    in name only.

    Only that one entry is dropped. The bundled SYCL toolchain's bin stays -
    a consumer built with this compiler genuinely needs its redistributables,
    and removing it would be testing the toolchain's installer, not this
    package.

    @param cfg The resolved configuration.
    @param root The project root.
    @param env The snapshotted build environment.
    @return A copy of *env* with the vcpkg bin filtered out of PATH.
    """
    vcpkg_prefix = _vcpkg_prefix(cfg, root)
    if not cfg.is_windows or not vcpkg_prefix:
        return env

    vcpkg_bin = Path(vcpkg_prefix, "bin")
    stripped = dict(env)
    for key, value in env.items():
        if key.upper() != "PATH":
            continue
        kept = [
            entry for entry in value.split(os.pathsep)
            if not entry or Path(entry) != vcpkg_bin
        ]
        stripped[key] = os.pathsep.join(kept)
    return stripped


def test_package(cfg: Config, root: Path, build_dir: Path, env: dict[str, str]) -> int:
    """Install the package and build tests/package/ against it, as a consumer would.

    This is the only test in the repo that exercises what ``find_package(SushiAI)``
    actually hands a third party, and it is the reason it cannot be a CTest label:
    a package that works only when the source tree happens to be next to it is not
    a package, so the consumer must configure from an install prefix with no access
    to this tree. It is also the only thing that would have caught a public header
    missing from the SushiAI.h umbrella, since every in-tree test includes the
    individual headers directly.

    One prefix is enough for all three modules: SushiBLAS and SushiRuntime are
    add_subdirectory'd into this build (see cmake/BLAS.cmake), so their own
    install rules run as part of this project's install and their config
    packages land in the same prefix the find_dependency calls in
    SushiAIConfig.cmake then resolve from.

    Both intermediates live under build/, so `sa clean` reclaims them and nothing
    lands in the source tree.

    @param cfg The resolved configuration.
    @param root The project root.
    @param build_dir The project's own build tree, which must already be built.
    @param env The snapshotted build environment.
    @return 0 when the consumer builds and runs, non-zero otherwise.
    """
    build_type = cached_value(build_dir, "CMAKE_BUILD_TYPE") or "Release"
    prefix = build_dir / "package" / "prefix"
    consumer = build_dir / "package" / "consumer"

    # The prefix and the consumer directory start from nothing; see cli/README.md.
    for stale in (prefix, consumer):
        if not _remove_tree(stale):
            return 1

    console.info(f"Installing to {prefix}...")
    rc = _RUNNER.run([_DRIVER.cmake(cfg), "--install", str(build_dir),
                      "--prefix", str(prefix), "--config", build_type], root, env)
    if rc != 0:
        console.error("Install failed; the package could not be staged.")
        return rc

    prefix_path = str(prefix)
    if cfg.is_windows and (vcpkg_prefix := _vcpkg_prefix(cfg, root)):
        prefix_path = f"{prefix_path};{vcpkg_prefix}"

    console.info("Configuring the out-of-tree consumer...")
    configure = [
        _DRIVER.cmake(cfg), "-S", str(root / "tests" / "package"), "-B", str(consumer),
        "-G", cfg.generator,
        # The consumer leaves some shared toolchain flags unread; see cli/README.md.
        "--no-warn-unused-cli",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_PREFIX_PATH={prefix_path}",
    ]
    configure += _toolchain_args(cfg, root)
    rc = _DRIVER.configure(configure, cwd=root, env=env)
    if rc != 0:
        console.error(
            "The consumer could not configure against the installed package.\n"
            "  - a FATAL_ERROR above about leaked flags means the exported "
            "interface carries development-only options; see "
            "cmake/SushiAIBuildPolicy.cmake.\n"
            "  - a failed find_dependency means a sibling's config package did "
            "not land in the prefix.")
        return rc

    console.info("Building the consumer...")
    rc = _DRIVER.compile(cfg, consumer, cwd=root, env=env, config=build_type)
    if rc != 0:
        console.error(
            "The consumer failed to compile against the installed headers.\n"
            "  - a missing header means it is either not covered by "
            "install(DIRECTORY) in cmake/SushiAIModule.cmake, or not reachable from "
            "the modules/layers/umbrella/include/SushiAI/SushiAI.h umbrella.")
        return rc

    exe = discovery.match_by_name(consumer, "sushiai_package_consumer")
    if exe is None:
        console.error("Built the consumer but could not find its executable.")
        return 1

    if not _deploy_consumer_dlls(cfg, root, prefix, exe.parent):
        return 1

    console.info(f"Executing: {exe.name}")
    rc = _RUNNER.run([str(exe)], consumer, _consumer_run_env(cfg, root, env))
    if rc == 0:
        console.success("Package consumer built and ran against the install prefix.")
    else:
        console.error(
            "The package consumer failed at runtime.\n"
            "  - exit code 0xC0000135 (-1073741515) means the loader could not "
            "find a DLL, not that the package is broken: something in the "
            "install's import graph is not staged by _deploy_consumer_dlls.")
    return rc


def run(target: str | None = None, sort: bool = False,
        app_args: list[str] | None = None, memory_limit_gb: float | None = None) -> int:
    """Runs a built executable from the project root, forwarding *app_args* to it."""
    console.header("Project Run")
    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        console.error("build/ not found. Run `sa build` first.")
        return 1

    env = load_build_env(cfg, build_dir)

    exe = discovery.resolve(build_dir, console, target=target, sort=sort, default=cfg.target_bin)
    if exe is None:
        return 1

    if not _enter_memory_limit(memory_limit_gb):
        return 2

    console.info(f"Executing: {exe.name}")
    console.console.print(f"[muted]{exe}[/muted]")
    return _RUNNER.run([str(exe), *(app_args or [])], root, env)


def demo_target(name: str) -> str | None:
    """Returns the executable that is the demo *name*, or None for an unknown demo."""
    return _DEMO_TARGETS.get(name)


def is_built(target: str) -> bool:
    """Returns whether the executable *target* exists in this checkout's build tree."""
    try:
        build_dir = _build_dir(find_project_root())
    except SystemExit:
        return False
    return build_dir.is_dir() and discovery.match_by_name(build_dir, target) is not None


def demo(name: str, app_args: Sequence[str] | None = None) -> int:
    """Runs the showcase demo *name* through :func:`run_app`; an unknown name returns 2."""
    target = demo_target(name)
    if target is None:
        console.header(f"SushiAI Demo: {name}")
        available = ", ".join(sorted(_DEMO_TARGETS))
        console.error(f"Unknown demo '{name}'. Available: {available}.")
        return 2
    return run_app(target, f"SushiAI Demo: {name}", list(app_args or ()))


def run_app(target: str, title: str, app_args: Sequence[str]) -> int:
    """Runs the built program *target* from the project root, forwarding *app_args* to it.

    Returns:
        1 when build/ or the program is missing, otherwise the program's own exit code.
    """
    console.header(title)

    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        console.error("build/ not found. Run `sa build` first.")
        return 1

    exe = discovery.match_by_name(build_dir, target)
    if exe is None:
        console.error(f"'{target}' is not built. Run `sa build` first.")
        return 1

    env = load_build_env(cfg, build_dir)
    console.info(f"Executing: {exe.name}")
    console.console.print(f"[muted]{exe}[/muted]")
    return _RUNNER.run([str(exe), *app_args], root, env)


def clean() -> int:
    """Removes the build tree and returns 1 when it is still there afterwards."""
    console.header("Project Clean")
    return _clean_build_tree(_build_dir(find_project_root()))


def doxygen() -> int:
    """Builds the API reference from .config/doxygen/Doxyfile into build/docs/api-site."""
    console.header("Doxygen Generation")
    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    env = load_build_env(cfg, build_dir)
    (build_dir / "docs").mkdir(parents=True, exist_ok=True)
    return _DRIVER.doxygen(cfg, root / ".config" / "doxygen" / "Doxyfile", root, env)

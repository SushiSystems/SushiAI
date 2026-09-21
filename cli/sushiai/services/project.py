"""Project build / test / run / demo / clean / doxygen logic.

This module holds SushiAI's own build policy — which cache variables it passes, which
demo targets and test suites it has — and drives cmake and ctest through
``sushicore.cmake_driver.CMakeDriver`` with the snapshotted build env. SushiAI configures
against the SushiRuntime sibling's bundled clang++ and vcpkg (via SushiBLAS's own
transitive resolution), so the configure args resolve those from the sibling checkouts
(or the config overrides) rather than provisioning anything - mirrors SushiEngine's `se`
CLI, not SushiBLAS's `sb` (which still models an older, pre-SB-0/SB-2 oneAPI
build).

tests/CMakeLists.txt registers tests via gtest_discover_tests with CTest
labels (unit / integration / regression) assigned from the GTest suite-name
prefix convention, matching sushiblas/sushiengine — see the ``Suite`` enum
below.
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
from ..config import Config, PROFILE, find_project_root, load_config
from ..env import load_build_env
from . import discovery


class BuildType(str, enum.Enum):
    release = "release"
    debug = "debug"
    relwithdebinfo = "relwithdebinfo"


class Suite(str, enum.Enum):
    # Fine-grained CTest labels (assigned in tests/CMakeLists.txt from the GTest
    # suite-name prefix convention).
    unit = "unit"
    regression = "regression"
    integration = "integration"
    functional = "functional"   # umbrella: unit + regression + integration
    all = "all"

    # Not a CTest label, and not part of `all`: an out-of-tree consumer of the
    # INSTALLED package, which has to be installed and configured on its own
    # before it can run at all. See :func:`test_package`.
    package = "package"


# Each suite maps to a CTest label-selection regex (`ctest -L`). The umbrella
# (functional) is an alternation over its sub-labels rather than a label of its
# own, because a baked-in umbrella label does not survive gtest_discover_tests on
# current CMake (see tests/CMakeLists.txt). `all` selects everything (no -L).
# `package` is absent on purpose - it runs no CTest at all.
_SUITE_LABEL_REGEX = {
    Suite.unit: "^unit$",
    Suite.regression: "^regression$",
    Suite.integration: "^integration$",
    Suite.functional: "^(unit|integration|regression)$",
}

# `sa demo <name>` -> the executable that IS that demo. A demo is a separate
# binary from an example on purpose: it is a supported entry point with a
# stable flag and output contract, while examples/example.cpp is a teaching
# file people are meant to edit. See examples/demo_mlp.cpp's header comment.
_DEMO_TARGETS = {
    "mlp": "sushiai_demo_mlp",
}

_CMAKE_BUILD_TYPE = {
    BuildType.release: "Release",
    BuildType.debug: "Debug",
    BuildType.relwithdebinfo: "RelWithDebInfo",
}


def _build_dir(root: Path) -> Path:
    return root / "build"


_RUNNER = Runner(console, PROFILE.program)
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


def _configure_args(cfg: Config, root: Path, build_dir: Path, build_type: str,
                    defines: Sequence[str] | None = None) -> list[str]:
    args = [
        _DRIVER.cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DSUSHIRUNTIME_DIR={cfg.runtime_dir(root)}",
        f"-DSUSHIBLAS_DIR={cfg.blas_dir(root)}",
    ]
    args += _toolchain_args(cfg, root)
    if cfg.is_windows and (prefix := _vcpkg_prefix(cfg, root)):
        args.append(f"-DCMAKE_PREFIX_PATH={prefix}")

    # Caller-supplied cache variables are appended last, after every default
    # this function sets, so that an explicit -D is never silently outranked by
    # one of them.
    for entry in defines or ():
        if "=" not in entry:
            raise ValueError(
                f"-D expects VAR=VALUE, got {entry!r}. "
                "Use VAR= to clear a variable.")
        args.append(f"-D{entry}")

    return args


def build(build_type: BuildType, clean: bool = False,
          defines: Sequence[str] | None = None) -> int:
    console.header("Project Build")
    root = find_project_root()
    cfg = load_config()
    cmake_build_type = _CMAKE_BUILD_TYPE[build_type]
    build_dir = _build_dir(root)

    if (rc := _check_siblings(cfg, root)) != 0:
        return rc

    if clean:
        _DRIVER.clean_tree(build_dir)

    console.info(f"Compiler: {cfg.resolved_compiler(root)}")
    console.info(f"SushiRuntime: {cfg.runtime_dir(root)}")
    console.info(f"SushiBLAS:    {cfg.blas_dir(root)}")

    env = load_build_env(cfg, build_dir)

    fresh = _DRIVER.needs_configure(build_dir, cfg.generator, root=root,
                                    expect={"CMAKE_BUILD_TYPE": cmake_build_type})
    if fresh:
        console.info(f"Configuring CMake... (type={build_type.value})")
        if build_dir.is_dir():
            shutil.rmtree(build_dir, ignore_errors=True)
    else:
        console.info("Reconfiguring in place...")
    try:
        configure_args = _configure_args(cfg, root, build_dir, cmake_build_type, defines)
    except ValueError as exc:
        console.error(str(exc))
        return 2

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


def test(suite: Suite = Suite.functional, filter: str | None = None, repeat: int = 0) -> int:
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

    if suite is Suite.package:
        if filter or repeat:
            # Both are ctest knobs, and this suite runs no ctest. Refusing
            # beats accepting a flag and silently doing nothing with it.
            console.error("--filter and --repeat do not apply to the package suite.")
            return 2
        return test_package(cfg, root, build_dir, env)

    if suite is Suite.all:
        # `all` means every CTest test, which is not the same as every test this
        # repo has. Saying so out loud beats a name that quietly under-delivers.
        console.info(
            "'all' covers every CTest test. The package suite is not among them "
            "(it builds a separate project): run `sa test --suite package`.")

    return _DRIVER.ctest_run(
        cfg, build_dir, env,
        label_regex=None if suite is Suite.all else _SUITE_LABEL_REGEX[suite],
        filter=filter, repeat=repeat)


def _deploy_consumer_dlls(cfg: Config, root: Path, prefix: Path, dest: Path) -> None:
    """Put the shared libraries the consumer needs beside its executable.

    In-tree targets get these from a POST_BUILD copy (see tests/CMakeLists.txt
    and examples/CMakeLists.txt); a real out-of-tree consumer would deploy them
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
    """
    if not cfg.is_windows:
        return  # ELF consumers resolve via RPATH; nothing to stage.

    sources = list((prefix / "bin").glob("*.dll"))
    if vcpkg_prefix := _vcpkg_prefix(cfg, root):
        sources += list(Path(vcpkg_prefix, "bin").glob("hwloc*.dll"))

    for dll in sources:
        shutil.copy2(dll, dest / dll.name)
    console.info(f"Deployed {len(sources)} shared libraries beside the consumer.")


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

    # Both start from nothing. A stale prefix would let a header deleted since
    # the last run keep satisfying an include; a stale consumer directory would
    # let a DLL staged by an earlier run - or by hand - keep satisfying the
    # loader, which is how a deployment regression hides. Found the hard way:
    # a hand-copied hwloc-15.dll from a manual run months earlier was still
    # sitting there, making the run step prove nothing.
    for stale in (prefix, consumer):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

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
        # The consumer is CXX-only and never invokes vcpkg's own installer, so
        # a couple of the shared toolchain flags legitimately go unread here.
        # Silencing exactly that warning keeps a clean run genuinely clean;
        # anything else cmake has to say still comes through.
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
            "install(DIRECTORY) in src/CMakeLists.txt, or not reachable from "
            "the include/SushiAI/SushiAI.h umbrella.")
        return rc

    exe = discovery.match_by_name(consumer, "sushiai_package_consumer")
    if exe is None:
        console.error("Built the consumer but could not find its executable.")
        return 1

    _deploy_consumer_dlls(cfg, root, prefix, exe.parent)

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
        app_args: list[str] | None = None) -> int:
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

    console.info(f"Executing: {exe.name}")
    console.console.print(f"[muted]{exe}[/muted]")
    return _RUNNER.run([str(exe), *(app_args or [])], root, env)


def demo(name: str, app_args: Sequence[str] | None = None) -> int:
    """Run a showcase demo, forwarding *app_args* to it verbatim.

    The flag surface lives in the demo program, not here: duplicating it in
    Typer would give two places to change and one to forget, and the demo's
    own ``--help`` is then the only reference anybody has to read.

    Run from the repo root, so a demo's relative data path (the MLP demo looks
    for ``data/mnist``) means the same thing wherever ``sa`` was invoked from.
    """
    console.header(f"SushiAI Demo: {name}")

    target = _DEMO_TARGETS.get(name)
    if target is None:
        available = ", ".join(sorted(_DEMO_TARGETS))
        console.error(f"Unknown demo '{name}'. Available: {available}.")
        return 2

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
    return _RUNNER.run([str(exe), *(app_args or [])], root, env)


def run_app(target: str, title: str, app_args: Sequence[str]) -> int:
    """Run one of the shipped app binaries, forwarding *app_args* verbatim.

    The same locate-build-then-exec shape as :func:`demo`, factored out rather
    than repeated: both need the build directory, the resolved environment and
    the repo root as the working directory, and a second copy would be a second
    place for the "run `sa build` first" advice to go stale.

    Run from the repo root, so a configuration path a caller typed relative to
    the checkout means the same thing wherever ``sushiai`` was invoked from.
    Paths *inside* a configuration file resolve against that file's own
    directory, which is the library's rule and not this one's.
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
    console.header("Project Clean")
    _DRIVER.clean_tree(_build_dir(find_project_root()))
    return 0


def doxygen() -> int:
    console.header("Doxygen Generation")
    root = find_project_root()
    cfg = load_config()
    env = load_build_env(cfg, _build_dir(root))
    doxyfile = root / "Doxyfile"
    if not doxyfile.exists():
        doxyfile = root / "docs" / "Doxyfile"
    (root / "build" / "docs").mkdir(parents=True, exist_ok=True)
    return _DRIVER.doxygen(cfg, doxyfile, root, env)

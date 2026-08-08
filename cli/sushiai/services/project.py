"""Project build / test / run / demo / clean / doxygen logic.

The cmake / ctest invocations are issued directly via subprocess with the
snapshotted build env. SushiAI configures against the SushiRuntime sibling's
bundled clang++ and vcpkg (via SushiBLAS's own transitive resolution), so the
configure args resolve those from the sibling checkouts (or the config
overrides) rather than provisioning anything - mirrors SushiEngine's `se`
CLI, not SushiBLAS's `sb` (which still models an older, pre-SB-0/SB-2 oneAPI
build).

tests/CMakeLists.txt registers tests via gtest_discover_tests with CTest
labels (unit / integration / regression) assigned from the GTest suite-name
prefix convention, matching sushiblas/sushiengine — see the ``Suite`` enum
below.
"""

from __future__ import annotations

import enum
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from .. import console
from ..config import Config, find_project_root, load_config
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


def _cmake(cfg: Config) -> str:
    """The cmake executable: the configured path if set, else bare 'cmake'."""
    return cfg.expand(cfg.cmake_exe) if cfg.cmake_exe else "cmake"


def _ctest(cfg: Config) -> str:
    """The ctest executable: the configured path if set, else bare 'ctest'."""
    return cfg.expand(cfg.ctest_exe) if cfg.ctest_exe else "ctest"


def _resolve_exe(name: str, env: dict[str, str]) -> str:
    """Return the full path to *name* from env PATH, or *name* itself as fallback.

    Using the full path sidesteps Windows PATH case-sensitivity bugs when a
    plain-dict env is passed to subprocess (os.environ uses 'Path', overlays may
    use 'PATH' — subprocess sees both and picks unpredictably).
    """
    env_path = next((v for k, v in env.items() if k.upper() == "PATH"), None)
    found = shutil.which(name, path=env_path) or shutil.which(name)
    return found or name


def _run(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    resolved = list(cmd)
    resolved[0] = _resolve_exe(cmd[0], env)
    console.command(subprocess.list2cmdline(resolved))
    try:
        return subprocess.run(resolved, cwd=str(cwd), env=env).returncode
    except FileNotFoundError:
        console.error(
            f"Executable not found: '{cmd[0]}'.\n"
            f"  - it is not on PATH and no explicit path is configured.\n"
            f"  - set its path in config.local.toml (e.g. cmake_exe / ctest_exe / ninja_exe), or\n"
            f"  - run `sa config` to see what the CLI resolved.")
        return 1


def _run_drained(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    """Run *cmd* while actively draining its output, echoing it line by line.

    Unlike :func:`_run` (which lets the child inherit our stdout), this pipes the
    child's stdout/stderr and reads them here. That matters for ``ctest`` on
    Windows: its ``gtest_discover_tests`` step spawns the test binary to enumerate
    cases, and when ctest's stdout is an inherited, slowly-drained pipe the
    discovery child intermittently stalls and registers nothing. Draining the
    pipe ourselves keeps enumeration reliable.
    """
    resolved = list(cmd)
    resolved[0] = _resolve_exe(cmd[0], env)
    console.command(subprocess.list2cmdline(resolved))
    try:
        proc = subprocess.Popen(
            resolved, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
    except FileNotFoundError:
        console.error(f"Executable not found: '{cmd[0]}'. Run `sa config`.")
        return 1
    assert proc.stdout is not None
    for line in proc.stdout:
        console.console.print(line.rstrip("\n"), highlight=False, soft_wrap=True)
    return proc.wait()


def _configured_build_type(build_dir: Path) -> Optional[str]:
    """The CMAKE_BUILD_TYPE baked into build_dir's cache, or None if unconfigured."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    prefix = "CMAKE_BUILD_TYPE:"
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _needs_configure(build_dir: Path, generator: str, build_type: str = None) -> bool:
    if not build_dir.is_dir():
        return True
    sentinel = "build.ninja" if generator == "Ninja" else "Makefile"
    if not (build_dir / sentinel).is_file():
        return True
    # Single-config generators (Ninja, Make) bake CMAKE_BUILD_TYPE into the
    # existing cache; `cmake --build --config X` is silently ignored for them,
    # so a stale tree must be reconfigured rather than reused as-is.
    if build_type is not None and _configured_build_type(build_dir) != build_type:
        return True
    return False


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


def _resolved_c_compiler(cxx: str) -> str:
    """The C compiler to pair with *cxx* for CMake's C-language probe.

    SushiAI's root CMakeLists.txt enables C (transitively - SushiBLAS's own
    project() call declares LANGUAGES CXX C, and that's inherited the moment
    it's add_subdirectory'd), so a C compiler must actually work, not just be
    cached. A "clang++"-named binary hardcodes C++ mode by personality - it
    fails outright on a .c file ("The CMAKE_C_COMPILER is set to a C++
    compiler") even though the same toolchain ships a working "clang" sibling
    in the same directory. A multi-language driver like icx-cl.exe has no such
    split (it dispatches on file extension correctly), so it's reused as-is.
    Found by actually running `sa build` fresh - the previous unconditional
    "reuse cxx for both slots" (copied from se's CLI, itself copied from sb's)
    was never exercised against a clang++-personality binary before.
    """
    path = Path(cxx)
    stem = path.stem
    if stem.lower() == "clang++":
        sibling = path.with_name(stem[:-2] + path.suffix)  # clang++[.exe] -> clang[.exe]
        if sibling.is_file():
            return str(sibling)
    return cxx


def _vcpkg_prefix(cfg: Config, root: Path) -> str:
    """The vcpkg installed-tree prefix for this triplet, or '' when unavailable.

    Only used on Windows, where the bundled vcpkg tree is what find_package
    uses to locate GoogleTest, nlohmann_json and hwloc. On Linux the vcpkg
    toolchain file alone puts them in reach, which is why the callers below
    guard on ``is_windows`` rather than on this returning something.

    @param cfg The resolved configuration.
    @param root The project root.
    @return An absolute prefix path, or '' when no vcpkg root is configured.
    """
    vcpkg = cfg.resolved_vcpkg(root)
    return f"{vcpkg}/installed/{cfg.vcpkg_triplet}" if vcpkg else ""


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
        _cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
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
        clean_tree(root)

    console.info(f"Compiler: {cfg.resolved_compiler(root)}")
    console.info(f"SushiRuntime: {cfg.runtime_dir(root)}")
    console.info(f"SushiBLAS:    {cfg.blas_dir(root)}")

    env = load_build_env(cfg, build_dir)

    fresh = _needs_configure(build_dir, cfg.generator, cmake_build_type)
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

    rc = _run(configure_args, env, cwd=root)
    if rc != 0:
        console.error("CMake configure failed.")
        return rc
    build_dir.mkdir(parents=True, exist_ok=True)

    console.info("Building...")
    rc = _run([_cmake(cfg), "--build", str(build_dir), "--config", cmake_build_type],
              env, cwd=root)
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

    cmd = [_ctest(cfg), "--test-dir", str(build_dir), "--output-on-failure"]
    if suite != Suite.all:
        cmd += ["-L", _SUITE_LABEL_REGEX[suite]]
    else:
        # `all` means every CTest test, which is not the same as every test this
        # repo has. Saying so out loud beats a name that quietly under-delivers.
        console.info(
            "'all' covers every CTest test. The package suite is not among them "
            "(it builds a separate project): run `sa test --suite package`.")
    if filter:
        # gtest_discover_tests registers tests as "Suite.Case"; ctest -R filters
        # those names directly (a richer alternative to --gtest_filter).
        cmd += ["-R", filter]
    if repeat > 0:
        cmd += ["--repeat", f"until-fail:{repeat}"]
        console.info(f"Repeating each test up to {repeat}x (stop on first failure).")
    return _run_drained(cmd, env, cwd=build_dir)


def _deploy_consumer_dlls(cfg: Config, root: Path, prefix: Path, dest: Path) -> None:
    """Put the shared libraries the consumer needs beside its executable.

    In-tree targets get these from a POST_BUILD copy (see tests/CMakeLists.txt
    and examples/CMakeLists.txt); a real out-of-tree consumer would deploy them
    itself, and this function is this CLI standing in for that consumer.
    Without them the binary exits with 0xC0000135 before reaching main, which
    looks like a package failure and is not one.

    Everything in the prefix's bin/ is taken rather than a named list: the
    prefix holds exactly what this project installed, and naming
    sushiruntime.dll here would be a fourth place to remember when the set
    changes. hwloc is not among them - it is linked PRIVATE by SushiRuntime and
    never reaches a consumer, which is why the in-tree copies do not stage it
    either.

    @param cfg The resolved configuration.
    @param root The project root.
    @param prefix The install prefix the package was installed to.
    @param dest The directory holding the consumer executable.
    """
    if not cfg.is_windows:
        return  # ELF consumers resolve via RPATH; nothing to stage.

    dlls = list((prefix / "bin").glob("*.dll"))
    for dll in dlls:
        shutil.copy2(dll, dest / dll.name)
    console.info(f"Deployed {len(dlls)} shared libraries beside the consumer.")


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
    build_type = _configured_build_type(build_dir) or "Release"
    prefix = build_dir / "package" / "prefix"
    consumer = build_dir / "package" / "consumer"

    # A stale prefix would let a header deleted since the last run keep
    # satisfying an include, so the install starts from nothing.
    if prefix.is_dir():
        shutil.rmtree(prefix, ignore_errors=True)

    console.info(f"Installing to {prefix}...")
    rc = _run([_cmake(cfg), "--install", str(build_dir),
               "--prefix", str(prefix), "--config", build_type], env, cwd=root)
    if rc != 0:
        console.error("Install failed; the package could not be staged.")
        return rc

    prefix_path = str(prefix)
    if cfg.is_windows and (vcpkg_prefix := _vcpkg_prefix(cfg, root)):
        prefix_path = f"{prefix_path};{vcpkg_prefix}"

    console.info("Configuring the out-of-tree consumer...")
    configure = [
        _cmake(cfg), "-S", str(root / "tests" / "package"), "-B", str(consumer),
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
    rc = _run(configure, env, cwd=root)
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
    rc = _run([_cmake(cfg), "--build", str(consumer), "--config", build_type],
              env, cwd=root)
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
    rc = _run([str(exe)], env, cwd=consumer)
    if rc == 0:
        console.success("Package consumer built and ran against the install prefix.")
    else:
        console.error("The package consumer failed at runtime.")
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

    if sort:
        exe = discovery.select_interactive(build_dir)
    elif target:
        exe = discovery.match_by_name(build_dir, target)
        if exe is None:
            console.error(f"Executable matching '{target}' not found.")
            return 1
    else:
        exe = discovery.match_by_name(build_dir, cfg.target_bin)
        if exe is None:
            console.error(f"Default target '{cfg.target_bin}' not found.")
            return 1

    if exe is None:
        return 1

    console.info(f"Executing: {exe.name}")
    console.console.print(f"[dim]{exe}[/dim]")
    return _run([str(exe), *(app_args or [])], env, cwd=root)


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
    console.console.print(f"[dim]{exe}[/dim]")
    return _run([str(exe), *(app_args or [])], env, cwd=root)


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
    console.console.print(f"[dim]{exe}[/dim]")
    return _run([str(exe), *app_args], env, cwd=root)


def clean_tree(root: Path) -> None:
    build_dir = _build_dir(root)
    if build_dir.is_dir():
        console.info(f"Removing {build_dir}...")
        shutil.rmtree(build_dir, ignore_errors=True)
        console.success("Build directory cleaned.")
    else:
        console.info("build/ does not exist, nothing to clean.")


def clean() -> int:
    console.header("Project Clean")
    clean_tree(find_project_root())
    return 0


def doxygen() -> int:
    console.header("Doxygen Generation")
    root = find_project_root()
    cfg = load_config()
    env = load_build_env(cfg, _build_dir(root))
    doxyfile = root / "Doxyfile"
    if not doxyfile.is_file():
        console.error("Doxyfile not found at repo root.")
        return 1

    doxy = cfg.expand(cfg.doxygen_exe) if cfg.doxygen_exe else "doxygen"
    if _resolve_exe(doxy, env) == doxy and not Path(doxy).is_file():
        console.error(
            "Doxygen is not installed or not on PATH.\n"
            "  - Windows: winget install DimitriVanHeesch.Doxygen\n"
            "  - Linux:   apt-get install -y doxygen graphviz\n"
            "  - macOS:   brew install doxygen graphviz\n"
            "  - or set doxygen_exe in config.local.toml to an existing doxygen binary.")
        return 1

    (root / "docs").mkdir(exist_ok=True)
    return _run([doxy, "Doxyfile"], env, cwd=root)

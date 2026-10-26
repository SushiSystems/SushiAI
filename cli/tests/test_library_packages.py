# test_library_packages.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that `sa` consumes the Sushi libraries as packages and puts their folders on the path."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from sushicore.provision.sushi_packages import PackageError

from sushiai.config import Config
from sushiai.services import library_packages, project
from sushiai.services.project import BuildType

pytestmark = pytest.mark.medium

K_RUNTIME_CONFIG = "/store/sushiruntime/abc/prefix/lib/cmake/SushiRuntime"
K_BLAS_CONFIG = "/store/sushiblas/def/prefix/lib/cmake/SushiBLAS"
K_RUN_DIRS = ["/store/sushiruntime/abc/prefix/lib", "/store/sushiblas/def/prefix/lib"]


class _Packages:
    """Stands in for the resolved packages of the closure."""

    def definitions(self) -> list[str]:
        """Returns the config folder definitions."""
        return [f"-DSushiBLAS_DIR={K_BLAS_CONFIG}", f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}"]

    def run_dirs(self, platform: str) -> list[str]:
        """Returns the run-time folders."""
        return list(K_RUN_DIRS)


class _Config:
    """Holds the settings the build reads and records what it asks the resolver."""

    generator = "Ninja"
    is_windows = False
    platform = "linux"
    target_bin = "sushiai_example"

    def __init__(self, fail: bool = False) -> None:
        self.asked: list[dict] = []
        self.fail = fail

    def sushi_definitions(self, root: Path, packaged=None) -> list[str]:
        """Returns no checkout definition; every pin is a package."""
        return []

    def sushi_packages(self, root, build_type, options=None, report=print, warn=print,
                       compiler=""):
        """Records the request and returns the packages, or fails like a refused closure."""
        self.asked.append({"build_type": build_type, "options": dict(options or {}),
                           "compiler": compiler})
        if self.fail:
            raise PackageError("sushiblas has no checkout")
        return _Packages()

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


class _Driver:
    """Records each configure and the environment it ran under."""

    def __init__(self) -> None:
        self.configured: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    def cmake(self, cfg) -> str:
        """Returns the cmake name."""
        return "cmake"

    def needs_configure(self, build_dir, generator, **_) -> bool:
        """Reports the tree as reusable."""
        return False

    def configure(self, args, cwd, env=None) -> int:
        """Records the arguments and the environment of one configure."""
        self.configured.append(list(args))
        self.environments.append(env)
        return 0

    def compile(self, cfg, build_dir, cwd, env, **options) -> int:
        """Succeeds."""
        return 0


class _Console:
    """Records every line the service prints, by kind."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.console = self

    def __getattr__(self, kind: str):
        """Returns a printer that records its text under *kind*."""
        return lambda text="", **_: self.lines.append((kind, str(text)))


@pytest.fixture
def parts(tmp_path, monkeypatch):
    """Returns a scratch checkout with the service's collaborators replaced."""
    cfg, driver, console = _Config(), _Driver(), _Console()
    monkeypatch.setattr(project, "console", console)
    monkeypatch.setattr(project, "_DRIVER", driver)
    monkeypatch.setattr(project, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(project, "load_config", lambda: cfg)
    monkeypatch.setattr(project, "load_build_env",
                        lambda cfg, tree: {"LD_LIBRARY_PATH": "/old"})
    return tmp_path, cfg, driver, console


@pytest.mark.parametrize(("requested", "expected"),
                         [("debug", "debug"), ("release", "release"), ("asan", "release"),
                          ("minsizerel", "release")])
def test_a_sanitizer_or_size_build_links_release_packages(requested, expected):
    assert library_packages.package_build_type(requested) == expected


def test_distributed_becomes_the_distributed_option_of_both_packages():
    assert library_packages.package_options(True) == {
        "sushiruntime": ("--distributed",), "sushiblas": ("--distributed",)}
    assert library_packages.package_options(False) == {}


def test_resolve_asks_for_the_closure_with_the_compiler_and_the_package_type():
    cfg = _Config()

    library_packages.resolve(cfg, Path("."), "asan", "clang++", True, "spir64", print, print)

    assert cfg.asked == [{"build_type": "release", "compiler": "clang++",
                          "options": {"sushiruntime": ("--distributed",),
                                      "sushiblas": ("--distributed", "-D",
                                                    "SB_SYCL_TARGETS=spir64")}}]


def test_the_run_dirs_are_written_read_and_put_first_on_the_path(tmp_path):
    env = {"LD_LIBRARY_PATH": "/old"}

    library_packages.record_run_dirs(tmp_path / "build", K_RUN_DIRS)
    library_packages.prepend_run_dirs(Config(platform="linux"), env,
                                      library_packages.read_run_dirs(tmp_path / "build"))

    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]


def test_a_tree_that_recorded_nothing_has_no_run_dirs(tmp_path):
    assert library_packages.read_run_dirs(tmp_path) == []


def test_the_windows_run_dirs_lead_path():
    env = {"PATH": "C:/old"}

    library_packages.prepend_run_dirs(Config(platform="windows"), env, ["C:/pkg/bin"])

    assert env["PATH"].startswith("C:/pkg/bin")


def test_the_configure_passes_each_package_and_no_checkout_directory(parts):
    root, cfg, driver, console = parts

    assert project.build(BuildType.release) == 0

    arguments = driver.configured[0]
    assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in arguments
    assert f"-DSushiBLAS_DIR={K_BLAS_CONFIG}" in arguments
    assert not [item for item in arguments
                if item.startswith(("-DSUSHIRUNTIME_DIR", "-DSUSHIBLAS_DIR", "-DSUSHILOG_DIR"))]


def test_the_build_names_its_compiler_and_type_to_the_resolver(parts):
    root, cfg, driver, console = parts

    assert project.build(BuildType.debug) == 0

    assert cfg.asked == [{"build_type": "debug", "options": {}, "compiler": "clang++"}]


def test_a_distributed_build_builds_the_packages_distributed(parts):
    root, cfg, driver, console = parts

    assert project.build(BuildType.release, distributed=True) == 0

    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}


def test_a_build_that_leaves_distributed_out_keeps_what_the_tree_has(parts):
    root, cfg, driver, console = parts
    (root / "build").mkdir()
    (root / "build" / "CMakeCache.txt").write_text("SA_ENABLE_DISTRIBUTED:BOOL=ON\n")

    assert project.build(BuildType.release) == 0

    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}


def test_the_run_dirs_lead_the_build_environment_and_a_later_run(parts):
    root, cfg, driver, console = parts

    assert project.build(BuildType.release) == 0

    assert driver.environments[0]["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]
    assert library_packages.read_run_dirs(root / "build") == K_RUN_DIRS
    assert project._run_env(cfg, root / "build")["LD_LIBRARY_PATH"].split(os.pathsep) == [
        *K_RUN_DIRS, "/old"]


def test_a_refused_closure_is_one_error_and_no_configure(parts):
    root, cfg, driver, console = parts
    cfg.fail = True

    assert project.build(BuildType.release) == 1

    assert driver.configured == []
    assert [text for kind, text in console.lines if kind == "error"] == [
        "sushiblas has no checkout"]


def test_a_program_runs_under_the_recorded_run_dirs(parts, monkeypatch):
    root, cfg, driver, console = parts
    library_packages.record_run_dirs(root / "build", K_RUN_DIRS)
    seen = {}

    class _Runner:
        """Records the environment of the program it runs."""

        def run(self, cmd, cwd, env=None) -> int:
            """Keeps *env* and succeeds."""
            seen["env"] = env
            return 0

    monkeypatch.setattr(project, "_RUNNER", _Runner())
    monkeypatch.setattr(project.discovery, "match_by_name",
                        lambda tree, name: Path(tree) / name)

    assert project.run_app("sushiai_train", "Train", []) == 0

    assert seen["env"]["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]


def test_the_consumer_runs_without_the_package_folders_on_its_path():
    cfg = SimpleNamespace(is_windows=True, resolved_vcpkg=lambda root: "", vcpkg_triplet="x")
    env = {"PATH": os.pathsep.join(["C:/pkg/bin", "C:/other"])}

    stripped = project._consumer_run_env(cfg, Path("."), env, ["C:/pkg/bin"])

    assert stripped["PATH"] == "C:/other"


def test_sycl_targets_go_to_the_blas_package_alone(parts):
    root, cfg, driver, console = parts

    assert project.build(BuildType.release, sycl_targets="spir64;nvidia_gpu_sm_86") == 0

    assert cfg.asked[0]["options"] == {
        "sushiblas": ("-D", "SB_SYCL_TARGETS=spir64;nvidia_gpu_sm_86")}
    assert not [item for item in driver.configured[0] if "SB_SYCL_TARGETS" in item]


def test_a_build_that_leaves_the_targets_out_keeps_what_the_tree_has(parts):
    root, cfg, driver, console = parts
    (root / "build").mkdir()
    (root / "build" / ".sushiai_sycl_targets").write_text("spir64")

    assert project.build(BuildType.release) == 0

    assert cfg.asked[0]["options"] == {"sushiblas": ("-D", "SB_SYCL_TARGETS=spir64")}


def test_an_empty_target_list_clears_the_targets_of_the_package(parts):
    root, cfg, driver, console = parts
    (root / "build").mkdir()
    (root / "build" / "CMakeCache.txt").write_text("SB_SYCL_TARGETS:STRING=spir64\n")

    assert project.build(BuildType.release, sycl_targets="") == 0

    assert cfg.asked[0]["options"] == {}


def test_the_package_suite_resolves_again_and_hands_the_consumer_the_packages(parts, monkeypatch):
    root, cfg, driver, console = parts
    build = root / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Debug\nSA_ENABLE_DISTRIBUTED:BOOL=ON\n")
    commands: list[list[str]] = []

    class _Runner:
        """Records each command and succeeds."""

        def run(self, cmd, cwd, env=None) -> int:
            """Keeps *cmd*."""
            commands.append(list(cmd))
            return 0

    monkeypatch.setattr(project, "_RUNNER", _Runner())
    monkeypatch.setattr(project.discovery, "match_by_name",
                        lambda tree, name: Path(tree) / name)

    assert project.test_package(cfg, root, build, {}) == 0

    assert cfg.asked[0]["build_type"] == "debug"
    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}
    consumer = driver.configured[0]
    assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in consumer
    assert f"-DSushiBLAS_DIR={K_BLAS_CONFIG}" in consumer


def test_a_refused_closure_stops_the_package_suite_before_the_consumer(parts, monkeypatch):
    root, cfg, driver, console = parts
    cfg.fail = True
    (root / "build").mkdir()

    class _Runner:
        """Succeeds without running anything."""

        def run(self, cmd, cwd, env=None) -> int:
            """Succeeds."""
            return 0

    monkeypatch.setattr(project, "_RUNNER", _Runner())
    assert project.test_package(cfg, root, root / "build", {}) == 1

    assert driver.configured == []


def test_the_consumer_gets_the_package_libraries_beside_its_executable(tmp_path, monkeypatch):
    prefix, dest, package = tmp_path / "prefix", tmp_path / "dest", tmp_path / "pkg" / "bin"
    (prefix / "bin").mkdir(parents=True)
    dest.mkdir()
    package.mkdir(parents=True)
    (package / "sushiruntime.dll").write_text("", encoding="utf-8")
    monkeypatch.setattr(project, "_vcpkg_prefix", lambda cfg, root: None)
    monkeypatch.setattr(project, "console", _Console())

    assert project._deploy_consumer_dlls(SimpleNamespace(is_windows=True), tmp_path, prefix,
                                         dest, [str(package)]) is True

    assert (dest / "sushiruntime.dll").is_file()

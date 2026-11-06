# test_library_packages.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that `sa` consumes the Sushi libraries as packages and puts their folders on the path."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from sushicore import build_admission
from sushicore.build_admission import BuildStoppedError
from sushicore.errors import ConfigError
from sushicore.proc_memory import RecordedCommand, RecordingRunner

from recorded_tools import (
    OfflinePackages, Recorded, offline_config, recorded_tools, write_executable)
from sushiai.config import Config
from sushiai.services import build_trees, library_packages, project
from sushiai.services.project import BuildType
from sushiai.services.sanitizer import Sanitizer

pytestmark = pytest.mark.medium

K_RUNTIME_CONFIG = "/store/sushiruntime/abc/prefix/lib/cmake/SushiRuntime"
K_BLAS_CONFIG = "/store/sushiblas/def/prefix/lib/cmake/SushiBLAS"
K_RUN_DIRS = ["/store/sushiruntime/abc/prefix/lib", "/store/sushiblas/def/prefix/lib"]
K_DEFINITIONS = [f"-DSushiBLAS_DIR={K_BLAS_CONFIG}", f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}"]
K_ENVIRONMENT = {"LD_LIBRARY_PATH": "/old"}


class _ConsumerBuildingRunner(RecordingRunner):
    """A runner whose build of the package consumer leaves the consumer's executable behind."""

    def run(self, cmd, cwd, env=None) -> int:
        """Records *cmd*, and writes the consumer program when *cmd* builds the consumer."""
        if "--build" in cmd and Path(cmd[cmd.index("--build") + 1]).name == "consumer":
            write_executable(Path(cmd[cmd.index("--build") + 1]) / "sushiai_package_consumer")
        return super().run(cmd, cwd, env)


def _tree(root: Path, build_type: str = "release") -> Path:
    """Returns the default build tree of *build_type* under *root*."""
    return build_trees.build_dir(root, build_type)


def _recorded(refuse: bool = False, runner_type=RecordingRunner):
    """Returns a configuration whose closure resolves, or is refused, and the tools over it."""
    cfg = offline_config(packages=OfflinePackages(K_DEFINITIONS, K_RUN_DIRS), refuse=refuse)
    return cfg, recorded_tools(cfg, K_ENVIRONMENT, runner_type)


@pytest.fixture
def parts(project_root):
    """Returns a scratch checkout, its configuration and the recorded tools over them."""
    cfg, recorded = _recorded()
    return project_root, cfg, recorded


def _configures(recorded: Recorded) -> list[RecordedCommand]:
    """Returns the cmake configure commands that ran, in order."""
    return [command for command in recorded.runner.commands
            if "-S" in command.argv and "-B" in command.argv]


def _compiles(recorded: Recorded) -> list[RecordedCommand]:
    """Returns the cmake build commands that ran, in order."""
    return recorded.commands_with("--build")


def _jobs_of(command: RecordedCommand) -> int:
    """Returns the --parallel count of a cmake build command."""
    return int(command.argv[command.argv.index("--parallel") + 1])


def _configured_tree(root: Path, build_type: str = "release") -> Path:
    """Returns a tree of *build_type* that holds a generated build, so a build reuses it."""
    tree = _tree(root, build_type)
    tree.mkdir(parents=True)
    (tree / "build.ninja").write_text("", encoding="utf-8")
    return tree


@pytest.mark.parametrize(("requested", "expected"),
                         [("debug", "release"), ("release", "release"), ("asan", "release"),
                          ("minsizerel", "release"), ("relwithdebinfo", "relwithdebinfo")])
def test_a_tree_links_the_package_type_sushicore_names_for_its_build_type(requested, expected):
    cfg = offline_config()

    library_packages.resolve(cfg, Path("."), requested, "clang++", False, "", print, print)

    assert cfg.asked[0]["build_type"] == expected


def test_distributed_becomes_the_distributed_option_of_both_packages():
    assert library_packages.package_options(True) == {
        "sushiruntime": ("--distributed",), "sushiblas": ("--distributed",)}
    assert library_packages.package_options(False) == {}


def test_resolve_asks_for_the_closure_with_the_compiler_and_the_package_type():
    cfg = offline_config()

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
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.release) == 0

    arguments = _configures(recorded)[0].argv
    assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in arguments
    assert f"-DSushiBLAS_DIR={K_BLAS_CONFIG}" in arguments
    assert not [item for item in arguments
                if item.startswith(("-DSUSHIRUNTIME_DIR", "-DSUSHIBLAS_DIR", "-DSUSHILOG_DIR"))]


def test_the_build_names_its_compiler_and_type_to_the_resolver(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.debug) == 0

    assert cfg.asked == [{"build_type": "release", "options": {}, "compiler": "clang++"}]


def test_a_distributed_build_builds_the_packages_distributed(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.release, distributed=True) == 0

    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}


def test_a_build_that_leaves_distributed_out_keeps_what_the_tree_has(parts):
    root, cfg, recorded = parts
    _configured_tree(root)
    (_tree(root) / "CMakeCache.txt").write_text("SA_ENABLE_DISTRIBUTED:BOOL=ON\n")

    assert project.build(recorded.tools, BuildType.release) == 0

    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}


def test_the_run_dirs_lead_the_build_environment_and_a_later_run(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.release) == 0

    configure_env = _configures(recorded)[0].env
    assert configure_env["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]
    assert library_packages.read_run_dirs(_tree(root)) == K_RUN_DIRS
    run_env = project._run_env(recorded.tools, cfg, _tree(root))
    assert run_env["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]


def test_a_refused_closure_is_one_error_and_no_configure(parts):
    root, cfg, recorded = parts
    cfg.refuse = True

    assert project.build(recorded.tools, BuildType.release) == 1

    assert _configures(recorded) == []
    assert recorded.errors() == ["sushiblas has no checkout"]


def test_a_program_runs_under_the_recorded_run_dirs(parts):
    root, cfg, recorded = parts
    library_packages.record_run_dirs(_tree(root, "debug"), K_RUN_DIRS)
    write_executable(_tree(root, "debug") / "sushiai_train")

    assert project.run_app(recorded.tools, "sushiai_train", "Train", []) == 0

    program = recorded.runner.commands[-1]
    assert program.env["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]


def test_the_consumer_runs_without_the_package_folders_on_its_path():
    cfg = SimpleNamespace(is_windows=True, resolved_vcpkg=lambda root: "", vcpkg_triplet="x")
    env = {"PATH": os.pathsep.join(["pkg/bin", "other"])}

    stripped = project._consumer_run_env(cfg, Path("."), env, ["pkg/bin"])

    assert stripped["PATH"] == "other"


def test_sycl_targets_go_to_the_blas_package_alone(parts):
    root, cfg, recorded = parts

    targets = "spir64;nvidia_gpu_sm_86"
    assert project.build(recorded.tools, BuildType.release, sycl_targets=targets) == 0

    assert cfg.asked[0]["options"] == {
        "sushiblas": ("-D", "SB_SYCL_TARGETS=spir64;nvidia_gpu_sm_86")}
    assert not [item for item in _configures(recorded)[0].argv if "SB_SYCL_TARGETS" in item]


def test_a_build_that_leaves_the_targets_out_keeps_what_the_tree_has(parts):
    root, cfg, recorded = parts
    _configured_tree(root)
    (_tree(root) / ".sushiai_sycl_targets").write_text("spir64")

    assert project.build(recorded.tools, BuildType.release) == 0

    assert cfg.asked[0]["options"] == {"sushiblas": ("-D", "SB_SYCL_TARGETS=spir64")}


def test_an_empty_target_list_clears_the_targets_of_the_package(parts):
    root, cfg, recorded = parts
    _configured_tree(root)
    (_tree(root) / "CMakeCache.txt").write_text("SB_SYCL_TARGETS:STRING=spir64\n")

    assert project.build(recorded.tools, BuildType.release, sycl_targets="") == 0

    assert cfg.asked[0]["options"] == {}


def test_the_package_suite_resolves_again_and_hands_the_consumer_the_packages(project_root):
    cfg, recorded = _recorded(runner_type=_ConsumerBuildingRunner)
    build = project_root / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Debug\nSA_ENABLE_DISTRIBUTED:BOOL=ON\n")

    assert project.test_package(recorded.tools, cfg, project_root, build, {}) == 0

    assert cfg.asked[0]["build_type"] == "release"
    assert set(cfg.asked[0]["options"]) == {"sushiruntime", "sushiblas"}
    consumer = _configures(recorded)[0].argv
    assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in consumer
    assert f"-DSushiBLAS_DIR={K_BLAS_CONFIG}" in consumer


def test_a_refused_closure_stops_the_package_suite_before_the_consumer(project_root):
    cfg, recorded = _recorded(refuse=True)
    (project_root / "build").mkdir()

    assert project.test_package(recorded.tools, cfg, project_root, project_root / "build", {}) == 1

    assert _configures(recorded) == []


def test_the_consumer_gets_the_package_libraries_beside_its_executable(tmp_path):
    prefix, dest, package = tmp_path / "prefix", tmp_path / "dest", tmp_path / "pkg" / "bin"
    (prefix / "bin").mkdir(parents=True)
    dest.mkdir()
    package.mkdir(parents=True)
    (package / "sushiruntime.dll").write_text("", encoding="utf-8")
    recorded = recorded_tools()

    assert project._deploy_consumer_dlls(recorded.console, offline_config("windows"), tmp_path,
                                         prefix, dest, [str(package)]) is True

    assert (dest / "sushiruntime.dll").is_file()


def test_the_default_build_type_is_debug():
    assert project.K_DEFAULT_TYPE is BuildType.debug


def test_a_plain_build_configures_the_debug_tree(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, project.K_DEFAULT_TYPE) == 0

    arguments = _configures(recorded)[0].argv
    assert arguments[arguments.index("-B") + 1] == str(_tree(root, "debug"))
    assert "-DCMAKE_BUILD_TYPE=Debug" in _configures(recorded)[0].argv


def test_a_configure_of_another_type_into_an_existing_tree_is_refused(parts):
    root, cfg, recorded = parts
    _tree(root, "debug").mkdir(parents=True)
    (_tree(root, "debug") / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")

    with pytest.raises(ConfigError):
        project.build(recorded.tools, BuildType.debug)

    assert _configures(recorded) == []


def test_a_define_that_sets_the_build_type_is_refused(parts):
    root, cfg, recorded = parts

    with pytest.raises(ConfigError):
        project.build(recorded.tools, BuildType.debug, defines=["CMAKE_BUILD_TYPE=Release"])

    assert _configures(recorded) == []


def test_a_stopped_gate_refuses_the_build_before_any_configure(parts):
    root, cfg, recorded = parts
    stop = build_admission.stop_file(_tree(root, "debug"))
    stop.parent.mkdir(parents=True)
    stop.write_text("")

    with pytest.raises(BuildStoppedError):
        project.build(recorded.tools, BuildType.debug)

    assert _configures(recorded) == []


def test_the_configure_carries_both_launcher_entries(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.debug) == 0

    arguments = _configures(recorded)[0].argv
    assert [item for item in arguments if item.startswith("-DCMAKE_CXX_COMPILER_LAUNCHER=")]
    assert [item for item in arguments if item.startswith("-DCMAKE_CXX_LINKER_LAUNCHER=")]


def test_the_compile_runs_the_hardware_threads_when_no_jobs_are_given(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.debug) == 0

    assert _jobs_of(_compiles(recorded)[0]) == build_admission.hardware_threads()


def test_the_compile_takes_the_jobs_the_user_gave(parts):
    root, cfg, recorded = parts

    assert project.build(recorded.tools, BuildType.debug, jobs=3) == 0

    assert _jobs_of(_compiles(recorded)[0]) == 3


def test_a_sanitizer_build_configures_its_own_tree(parts):
    root, cfg, recorded = parts

    sanitizer = Sanitizer.address_undefined
    assert project.build(recorded.tools, BuildType.debug, sanitizer=sanitizer) == 0

    arguments = _configures(recorded)[0].argv
    assert arguments[arguments.index("-B") + 1] == str(root / "build" / "asan-debug")
    assert "-DSA_SANITIZE=address,undefined" in arguments

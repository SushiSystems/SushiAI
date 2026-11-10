# test_project.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the build policy: what it refuses, and what it reports when the disk refuses."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from sushicore.cmake_driver import CMakeDriver
from sushicore.console_memory import RecordingConsole
from sushicore.proc_memory import RecordingRunner

from sushiai.services import project
from sushiai.services.toolbox import default_toolbox

from recorded_tools import Recorded, offline_config, recorded_tools, write_executable


class _StuckDriver(CMakeDriver):
    """A driver whose clean leaves the tree in place, as a file another program holds open does."""

    def clean_tree(self, build_dir: Path) -> bool:
        """Leaves *build_dir* alone and reports that it is still there."""
        return False


@pytest.fixture
def recorded() -> Recorded:
    """Returns a toolbox that records its console and its commands."""
    return recorded_tools()


@pytest.fixture
def tree(project_root: Path) -> Path:
    """Returns a project root that holds the default build tree and a file beside it."""
    (project_root / "build" / "default-debug").mkdir(parents=True)
    (project_root / "build" / "CMakeCache.txt").write_text("", encoding="utf-8")
    return project_root


def _stuck(recorded: Recorded) -> Recorded:
    """Returns *recorded* with a driver that cannot remove a tree."""
    driver = _StuckDriver(recorded.console, recorded.runner)
    return dataclasses.replace(recorded, tools=dataclasses.replace(recorded.tools, driver=driver))


@pytest.mark.medium
def test_build_rejects_a_malformed_define_before_it_removes_the_tree(recorded, tree):
    """`sa build --clean -D BAD` exits 2 and leaves build/ as it was."""
    assert project.build(recorded.tools, project.K_DEFAULT_TYPE, clean=True,
                         defines=["SA_BUILD_TESTS"]) == 2
    assert recorded.runner.commands == []
    assert (tree / "build" / "CMakeCache.txt").is_file()
    assert "SA_BUILD_TESTS" in recorded.errors()[0]


def _same(recorded: Recorded) -> Recorded:
    """Returns *recorded* unchanged, whose driver removes a tree."""
    return recorded


@pytest.mark.medium
@pytest.mark.parametrize(("rig", "survives", "exit_code"), [(_stuck, True, 1), (_same, False, 0)],
                         ids=["tree-survives", "tree-gone"])
def test_clean_reports_a_tree_it_could_not_remove(rig, survives, exit_code, recorded, tree):
    """A build tree that survives the removal is an error naming it with exit code 1."""
    rigged = rig(recorded)

    assert project.clean(rigged.tools) == exit_code

    assert (tree / "build").exists() == survives
    assert bool(recorded.errors()) == survives
    assert (str(tree / "build") in " ".join(recorded.errors())) == survives


@pytest.mark.medium
def test_the_package_suite_stops_on_a_stale_prefix_it_could_not_remove(recorded, tree):
    """A prefix left by an earlier run that cannot be removed ends the suite before install."""
    prefix = tree / "build" / "package" / "prefix"
    prefix.parent.mkdir(parents=True)
    prefix.write_text("a file where a folder was", encoding="utf-8")
    assert project.test_package(recorded.tools, offline_config(), tree, tree / "build", {}) == 1
    assert recorded.runner.commands == []
    assert str(prefix) in recorded.errors()[0]


@pytest.mark.medium
def test_staging_a_library_that_cannot_be_copied_is_one_error(recorded, tmp_path):
    """A library that cannot be copied is reported with its name instead of raised."""
    prefix, dest = tmp_path / "prefix", tmp_path / "dest"
    (prefix / "bin" / "sushiai.dll").mkdir(parents=True)
    dest.mkdir()
    cfg = offline_config("windows")
    assert project._deploy_consumer_dlls(recorded.console, cfg, tmp_path, prefix, dest) is False
    assert len(recorded.errors()) == 1 and "sushiai.dll" in recorded.errors()[0]


@pytest.mark.medium
def test_an_interrupted_child_ends_in_exit_code_130(tmp_path, monkeypatch):
    """Ctrl+C while a child runs becomes exit code 130, not an exception."""
    def interrupt(*_args, **_kwargs):
        """Raises what Ctrl+C raises in the parent."""
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    assert default_toolbox().runner.run(["sushiai_example"], tmp_path) == 130


@pytest.mark.medium
@pytest.mark.parametrize("target", ["sushiai_train", "sushiai_demo_mlp"])
def test_a_program_is_not_built_outside_a_checkout(target, tmp_path, monkeypatch):
    """is_built answers False, without exiting, where there is no project at all."""
    monkeypatch.chdir(tmp_path)
    assert project.is_built(target) is False


@pytest.mark.small
def test_an_unknown_demo_names_no_program():
    """demo_target answers None for a demo that does not exist."""
    assert project.demo_target("no-such-demo") is None


def _ctest(recorded: Recorded) -> list[str]:
    """Returns the words of the one ctest command that was run."""
    ran = recorded.commands_with("--output-on-failure")
    assert len(ran) == 1, recorded.runner.commands
    return list(ran[0].argv)


def _value_of(words: list[str], option: str) -> str | None:
    """Returns the word after *option*, or None when *option* is absent."""
    return words[words.index(option) + 1] if option in words else None


@pytest.mark.medium
@pytest.mark.parametrize("jobs", [3, None])
def test_jobs_reach_ctest_as_given(jobs, recorded, tree):
    """--jobs reaches ctest unchanged, and None leaves the count to ctest."""
    assert project.test(recorded.tools, jobs=jobs, memory_limit_gb=0.0) == 0
    assert _value_of(_ctest(recorded), "--parallel") == (None if jobs is None else str(jobs))


@pytest.mark.medium
@pytest.mark.parametrize(("suite", "include", "exclude"), [
    (project.Suite.gating, "^(small|medium)$", "^quarantined$"),
    (project.Suite.large, "^(large)$", "^quarantined$"),
    (project.Suite.all, "^(small|medium|large)$", None),
])
def test_a_suite_reaches_ctest_as_the_labels_of_the_shared_command(suite, include, exclude,
                                                                    recorded, tree):
    """Each suite name becomes the shared label patterns, and every run asks ctest for a JUnit file."""
    assert project.test(recorded.tools, suite, memory_limit_gb=0.0) == 0
    words = _ctest(recorded)
    assert (_value_of(words, "-L"), _value_of(words, "-LE")) == (include, exclude)
    assert Path(_value_of(words, "--output-junit")).name == "junit.xml"


class _FailingShardRunner(RecordingRunner):
    """A runner whose ctest leaves the files of a run in which one case of one shard failed."""

    def __init__(self, console, tree: Path, shard: str) -> None:
        """Keep the tree the run writes its results into and the shard that fails."""
        super().__init__(console)
        self._tree = tree
        self._shard = shard

    def run_drained(self, cmd, cwd, env=None) -> int:
        """Records *cmd*; a ctest run writes its JUnit file and the shard's result file and fails."""
        code = super().run_drained(cmd, cwd, env)
        if "--output-junit" not in cmd:
            return code
        junit = Path(cmd[cmd.index("--output-junit") + 1])
        junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_text(
            f'<testsuite><testcase name="{self._shard}" status="fail"><failure/></testcase></testsuite>',
            encoding="utf-8")
        results = self._tree / "test_results" / "gtest"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"{self._shard}.xml").write_text(
            '<testsuites><testsuite><testcase name="One" classname="Small_A"><failure/></testcase>'
            "</testsuite></testsuites>", encoding="utf-8")
        return 1


@pytest.mark.medium
def test_a_failed_shard_reruns_by_gtest_filter_on_its_program_and_not_by_ctest_over_the_shard(tree):
    """sa hands the shared command a run_program, so only the failing case of a failed shard reruns."""
    build = tree / "build" / "default-debug"
    shard = "sushiai_tests.small_shard_0of1"
    program = write_executable(build / "bin" / "sushiai_tests")
    (build / "CTestTestfile.cmake").write_text(
        f'add_test([=[{shard}]=] "{program.as_posix()}")\n', encoding="utf-8")
    console = RecordingConsole()
    runner = _FailingShardRunner(console, build, shard)
    base = recorded_tools()
    tools = dataclasses.replace(base.tools, console=console, runner=runner,
                                driver=CMakeDriver(console, runner))

    assert project.test(tools, memory_limit_gb=0.0) == 1

    commands = [command.argv for command in runner.commands]
    reruns = [argv for argv in commands if argv[0] == str(program)]
    assert [argv[1] for argv in reruns] == ["--gtest_filter=Small_A.One"]
    assert len([argv for argv in commands if "--output-on-failure" in argv]) == 1

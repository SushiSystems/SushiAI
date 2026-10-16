# test_project.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the build policy: what it refuses, and what it reports when the disk refuses."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sushiai.services import project
from sushiai.services.project import BuildType

pytestmark = pytest.mark.medium


class _Console:
    """Records every line the service prints, by kind."""

    def __init__(self) -> None:
        """Starts with nothing recorded."""
        self.lines: list[tuple[str, str]] = []
        self.console = self

    def __getattr__(self, kind: str):
        """Returns a printer that records its text under *kind*."""
        return lambda text="", **_: self.lines.append((kind, str(text)))

    def errors(self) -> list[str]:
        """Returns the text of every error line."""
        return [text for kind, text in self.lines if kind == "error"]


class _Driver:
    """Stands in for the CMake driver and records what it was asked to do."""

    def __init__(self, removes: bool) -> None:
        """Remembers whether clean_tree really removes the directory."""
        self.calls: list[str] = []
        self._removes = removes

    def clean_tree(self, build_dir: Path) -> None:
        """Records the call and removes the tree when told to."""
        self.calls.append("clean_tree")
        if self._removes:
            shutil.rmtree(build_dir)

    def ctest_run(self, cfg, build_dir, env, **options) -> int:
        """Records the options of the ctest run and reports success."""
        self.calls.append("ctest_run")
        self.ctest_options = options
        return 0


class _Runner:
    """Stands in for the process runner and records each command."""

    def __init__(self) -> None:
        """Starts with nothing run."""
        self.commands: list[list[str]] = []

    def run(self, cmd, cwd, env=None) -> int:
        """Records the command and reports success."""
        self.commands.append(list(cmd))
        return 0


@pytest.fixture
def console(monkeypatch) -> _Console:
    """Replaces the service's console with a recorder."""
    recorder = _Console()
    monkeypatch.setattr(project, "console", recorder)
    return recorder


@pytest.fixture
def tree(tmp_path, monkeypatch) -> Path:
    """Returns a project root that holds a build tree with one file in it."""
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "CMakeCache.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(project, "find_project_root", lambda: tmp_path)
    return tmp_path


def _refuse(*_args, **_kwargs):
    """Fails the way a locked file does on Windows."""
    raise PermissionError(13, "Access is denied")


def test_build_rejects_a_malformed_define_before_it_removes_the_tree(console, tree, monkeypatch):
    """`sa build --clean -D BAD` exits 2 and leaves build/ as it was."""
    driver = _Driver(removes=True)
    monkeypatch.setattr(project, "_DRIVER", driver)
    assert project.build(BuildType.release, clean=True, defines=["SA_BUILD_TESTS"]) == 2
    assert driver.calls == []
    assert (tree / "build" / "CMakeCache.txt").is_file()
    assert "SA_BUILD_TESTS" in console.errors()[0]


def test_clean_reports_a_tree_it_could_not_remove(console, tree, monkeypatch):
    """A build tree that survives the removal is an error naming it, and exit code 1."""
    monkeypatch.setattr(project, "_DRIVER", _Driver(removes=False))
    assert project.clean() == 1
    assert str(tree / "build") in console.errors()[0]


def test_clean_succeeds_when_the_tree_is_gone(console, tree, monkeypatch):
    """A removed build tree is exit code 0 and no error."""
    monkeypatch.setattr(project, "_DRIVER", _Driver(removes=True))
    assert project.clean() == 0
    assert console.errors() == []


def test_the_package_suite_stops_on_a_stale_prefix_it_could_not_remove(
        console, tree, monkeypatch):
    """A prefix left by an earlier run that cannot be removed ends the suite before install."""
    prefix = tree / "build" / "package" / "prefix"
    prefix.mkdir(parents=True)
    runner = _Runner()
    monkeypatch.setattr(project, "_RUNNER", runner)
    monkeypatch.setattr(shutil, "rmtree", _refuse)
    cfg = SimpleNamespace(is_windows=True)
    assert project.test_package(cfg, tree, tree / "build", {}) == 1
    assert runner.commands == []
    assert str(prefix) in console.errors()[0]


def test_staging_a_library_that_cannot_be_copied_is_one_error(console, tmp_path, monkeypatch):
    """A locked DLL is reported with its name instead of raised."""
    prefix, dest = tmp_path / "prefix", tmp_path / "dest"
    (prefix / "bin").mkdir(parents=True)
    dest.mkdir()
    (prefix / "bin" / "sushiai.dll").write_text("", encoding="utf-8")
    monkeypatch.setattr(project, "_vcpkg_prefix", lambda cfg, root: None)
    monkeypatch.setattr(shutil, "copy2", _refuse)
    cfg = SimpleNamespace(is_windows=True)
    assert project._deploy_consumer_dlls(cfg, tmp_path, prefix, dest) is False
    assert len(console.errors()) == 1 and "sushiai.dll" in console.errors()[0]


def test_an_interrupted_child_ends_in_exit_code_130(tmp_path, monkeypatch):
    """Ctrl+C while a child runs becomes exit code 130, not an exception."""
    def interrupt(*_args, **_kwargs):
        """Raises what Ctrl+C raises in the parent."""
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    assert project._RUNNER.run(["sushiai_example"], tmp_path) == 130


@pytest.mark.parametrize("target", ["sushiai_train", "sushiai_demo_mlp"])
def test_a_program_is_not_built_outside_a_checkout(target, tmp_path, monkeypatch):
    """is_built answers False, without exiting, where there is no project at all."""
    monkeypatch.chdir(tmp_path)
    assert project.is_built(target) is False


def test_the_mlp_demo_names_its_program():
    """demo_target maps a demo's name to the executable that is that demo."""
    assert project.demo_target("mlp") == "sushiai_demo_mlp"
    assert project.demo_target("no-such-demo") is None


@pytest.mark.parametrize("jobs", [3, None])
def test_jobs_reach_ctest_as_given(jobs, console, tree, monkeypatch):
    """--jobs reaches ctest_run unchanged, and None leaves the count to ctest."""
    driver = _Driver(removes=False)
    monkeypatch.setattr(project, "_DRIVER", driver)
    monkeypatch.setattr(project, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(project, "load_build_env", lambda cfg, build_dir: {})
    monkeypatch.setattr(project, "_enter_memory_limit", lambda limit: True)

    assert project.test(jobs=jobs) == 0
    assert driver.ctest_options["jobs"] == jobs

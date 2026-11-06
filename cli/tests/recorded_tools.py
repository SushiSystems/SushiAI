# recorded_tools.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Builds a toolbox whose console and runner record, so a service runs with no process and no terminal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sushicore.cmake_driver import CMakeDriver
from sushicore.console_memory import RecordingConsole
from sushicore.proc_memory import RecordedCommand, RecordingRunner
from sushicore.provision.sushi_packages import PackageError

from sushiai.config import Config
from sushiai.services.toolbox import Toolbox


class OfflinePackages:
    """Stands in for the resolved packages of the closure.

    Args:
        definitions: The ``-D<Package>_DIR`` entries the configure receives.
        run_dirs: The run-time folders of the packages.
    """

    def __init__(self, definitions: list[str], run_dirs: list[str]) -> None:
        """Keep what the closure resolved to."""
        self._definitions = definitions
        self._run_dirs = run_dirs

    def definitions(self) -> list[str]:
        """Returns the config folder definitions."""
        return list(self._definitions)

    def run_dirs(self, platform: str) -> list[str]:
        """Returns the run-time folders."""
        return list(self._run_dirs)


class OfflineConfig(Config):
    """A configuration that searches no disk: a fixed compiler, no vcpkg tree and recorded packages.

    Attributes:
        asked: One entry per request to resolve the packages, in the order they came.
        packages: What the resolver answers.
        refuse: Whether the resolver refuses the closure.
    """

    asked: list[dict]
    packages: OfflinePackages
    refuse: bool

    def sushi_definitions(self, root: Path, packaged=None) -> list[str]:
        """Returns no checkout definition; every pin is a package."""
        return []

    def sushi_packages(self, root, build_type, options=None, report=print, warn=print,
                       compiler=""):
        """Records the request, then returns the packages or refuses like a closure that failed."""
        self.asked.append({"build_type": build_type, "options": dict(options or {}),
                           "compiler": compiler})
        if self.refuse:
            raise PackageError("sushiblas has no checkout")
        return self.packages

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


def offline_config(platform: str = "linux", packages: OfflinePackages | None = None,
                   refuse: bool = False, **settings: str) -> OfflineConfig:
    """Returns an :class:`OfflineConfig` for *platform* that answers the resolver with *packages*.

    *settings* are further fields of the configuration, such as ``doxygen_exe``.
    """
    config = OfflineConfig(platform=platform, **settings)
    config.asked = []
    config.packages = packages or OfflinePackages([], [])
    config.refuse = refuse
    return config


@dataclass(frozen=True, slots=True)
class Recorded:
    """A toolbox together with the recorders behind it.

    Attributes:
        tools: What a service is handed.
        runner: Records each command the service ran.
        console: Records each line the service printed.
    """

    tools: Toolbox
    runner: RecordingRunner
    console: RecordingConsole

    def commands_with(self, word: str) -> list[RecordedCommand]:
        """Returns the recorded commands whose words include *word*, in the order they ran."""
        return [command for command in self.runner.commands if word in command.argv]

    def errors(self) -> list[str]:
        """Returns the text of every error line."""
        return self.console.texts("error")


def recorded_tools(config: Config | None = None,
                   environment: Mapping[str, str] | None = None,
                   runner_type: type[RecordingRunner] = RecordingRunner) -> Recorded:
    """Returns a toolbox that records its console and its commands.

    Args:
        config: What the toolbox loads as the configuration; a Linux one when None.
        environment: What the toolbox gives as the build environment; a fresh copy for each call.
        runner_type: The recording runner to build, for a test whose commands leave files behind.
    """
    console = RecordingConsole()
    runner = runner_type(console)
    loaded = config if config is not None else Config(platform="linux")
    tools = Toolbox(
        console=console,
        runner=runner,
        driver=CMakeDriver(console, runner),
        load_config=lambda: loaded,
        load_build_env=lambda cfg, build_dir: dict(environment or {}),
    )
    return Recorded(tools, runner, console)


def write_executable(path: Path) -> Path:
    """Writes an empty file that the build-tree discovery takes for a program, and returns it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    program = path.with_suffix(".exe") if path.suffix == "" else path
    program.write_text("", encoding="utf-8")
    program.chmod(0o755)
    return program


def doxygen_ready(root: Path, exit_code: int) -> Recorded:
    """Returns recorded tools for a checkout *root* where `doxygen` is found and exits *exit_code*."""
    doxyfile = root / ".config" / "doxygen" / "Doxyfile"
    doxyfile.parent.mkdir(parents=True)
    doxyfile.write_text("", encoding="utf-8")
    program = write_executable(root / "bin" / "doxygen")
    recorded = recorded_tools(offline_config(doxygen_exe=str(program)))
    recorded.runner.script(exit_code)
    return recorded

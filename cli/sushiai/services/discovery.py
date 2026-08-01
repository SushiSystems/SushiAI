"""Locate built executables under a build tree."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from rich.prompt import IntPrompt
from rich.table import Table

from .. import console

_MAX_DEPTH = 4
_EXCLUDE_DIRS = {"CMakeFiles", "vcpkg_installed", "_deps", "sushiruntime", "sushiblas"}
_SKIP_SUFFIXES = {
    ".cmake", ".ninja", ".log", ".txt", ".json", ".a", ".o",
    ".cmake_install", ".lib", ".pdb", ".obj", ".dll", ".so",
}
_IS_WINDOWS = platform.system().lower() == "windows"


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name.startswith("."):
        return False
    # *.so / *.so.1 style shared objects on Linux.
    if ".so" in name:
        return False
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return False
    if _IS_WINDOWS:
        return path.suffix.lower() == ".exe"
    # Linux: regular file with an execute bit and no library suffix.
    return os.access(path, os.X_OK)


def find_executables(build_root: Path) -> list[Path]:
    """Return candidate executables under ``build_root`` (depth-limited)."""
    if not build_root.is_dir():
        return []
    found: list[Path] = []
    root_depth = len(build_root.parts)
    for dirpath, dirnames, filenames in os.walk(build_root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
        for fname in filenames:
            candidate = Path(dirpath) / fname
            if _is_executable(candidate):
                found.append(candidate)
    return sorted(found, key=lambda p: p.name.lower())


def select_interactive(build_root: Path) -> Path | None:
    """Show a table of executables and prompt the user to pick one."""
    exes = find_executables(build_root)
    if not exes:
        console.error("No executables found. Build the project first.")
        return None

    table = Table(title="Available executables", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Path", style="dim")
    for i, exe in enumerate(exes, 1):
        table.add_row(str(i), exe.name, str(exe.relative_to(build_root.parent)))
    console.console.print(table)

    choice = IntPrompt.ask(
        "Select a number", choices=[str(i) for i in range(1, len(exes) + 1)]
    )
    return exes[choice - 1]


def match_by_name(build_root: Path, query: str) -> Path | None:
    """Exact-name match, falling back to substring match."""
    exes = find_executables(build_root)
    for exe in exes:
        if exe.name == query or exe.stem == query:
            return exe
    for exe in exes:
        if query.lower() in exe.name.lower():
            return exe
    return None

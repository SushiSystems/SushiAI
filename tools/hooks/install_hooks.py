# install_hooks.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Points git at the tracked hooks folder of a repository, so every clone runs the same hooks.

Usage: python tools/hooks/install_hooks.py [repository root]

The repository's setup command runs it once per clone.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

K_HOOKS_PATH = "tools/hooks"
K_EXIT_DONE = 0
K_EXIT_FAILED = 1


@dataclass(frozen=True, slots=True)
class Outcome:
    """Holds how an installation ended: the exit code, and what to print on stdout and on stderr."""

    code: int
    out: str
    err: str


def hooks_path_command(root: Path) -> list[str]:
    """Returns the git command that sets core.hooksPath of the repository at root."""
    return ["git", "-C", str(root), "config", "core.hooksPath", K_HOOKS_PATH]


def root_of(arguments: Sequence[str], current: Path) -> Path:
    """Returns the repository named by the first argument, or *current* when there is none."""
    return Path(arguments[0]) if arguments else current


def outcome_of(root: Path, accepted: bool) -> Outcome:
    """Returns what to report once git has accepted the hooks path of root or refused it."""
    if not accepted:
        return Outcome(K_EXIT_FAILED, "", f"cannot set core.hooksPath in {root}; is it a git repository?")
    return Outcome(K_EXIT_DONE, f"git now runs the hooks in {K_HOOKS_PATH}", "")


def install(root: Path) -> bool:
    """Sets core.hooksPath of the repository at root to the tracked hooks folder; returns whether git accepted it."""
    completed = subprocess.run(hooks_path_command(root), capture_output=True, text=True)
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Installs the hooks of the repository named on the command line, or of the current folder."""
    arguments = sys.argv[1:] if argv is None else argv
    root = root_of(arguments, Path.cwd())
    outcome = outcome_of(root, install(root))
    if outcome.out:
        print(outcome.out)
    if outcome.err:
        print(outcome.err, file=sys.stderr)
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())

# install_hooks.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Points git at the tracked hooks folder of a repository, so every clone runs the same hooks.

Usage: python tools/hooks/install_hooks.py [repository root]

The repository's setup command runs it once per clone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

K_HOOKS_PATH = "tools/hooks"
K_EXIT_DONE = 0
K_EXIT_FAILED = 1


def install(root: Path) -> bool:
    """Sets core.hooksPath of the repository at root to the tracked hooks folder; returns whether git accepted it."""
    completed = subprocess.run(
        ["git", "-C", str(root), "config", "core.hooksPath", K_HOOKS_PATH], capture_output=True, text=True)
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Installs the hooks of the repository named on the command line, or of the current folder."""
    arguments = sys.argv[1:] if argv is None else argv
    root = Path(arguments[0]) if arguments else Path.cwd()
    if not install(root):
        print(f"cannot set core.hooksPath in {root}; is it a git repository?", file=sys.stderr)
        return K_EXIT_FAILED
    print(f"git now runs the hooks in {K_HOOKS_PATH}")
    return K_EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())

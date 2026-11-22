# pre_commit.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Runs every checker on the repository in place, when the disk holds what the commit will hold.

Usage: python tools/hooks/pre_commit.py [repository root]

Git runs it through the pre-commit hook. Nothing is written to disk: a staged file whose disk copy
differs from the index refuses the commit, because the checkers read the disk.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

K_CHECK_ALL = Path("tools") / "check_all.py"
K_EXIT_REFUSED = 1
K_SKIP_HISTORY = "--skip-history"


def absolute_git_dir(root: Path, environ: Mapping[str, str]) -> Path | None:
    """Returns the git directory of the repository at *root*, or None when git cannot name it."""
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--absolute-git-dir"],
        capture_output=True, text=True, env=dict(environ), check=False)
    return Path(done.stdout.strip()) if done.returncode == 0 and done.stdout.strip() else None


def index_file(git_dir: Path, environ: Mapping[str, str]) -> Path:
    """Returns the index git hands the hook: `commit -a` and `-o` give their own."""
    given = environ.get("GIT_INDEX_FILE")
    if not given:
        return git_dir / "index"
    path = Path(given)
    return path if path.is_absolute() else Path(environ.get("GIT_PREFIX", ".")).resolve() / path


def checked_environment(git_dir: Path, root: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """Returns the environment `check_all.py` runs in: git reads the index the commit holds."""
    return {
        **environ,
        "GIT_DIR": str(git_dir),
        "GIT_WORK_TREE": str(root),
        "GIT_INDEX_FILE": str(index_file(git_dir, environ)),
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def unstaged_files(root: Path, environ: Mapping[str, str]) -> list[str] | None:
    """Returns the staged files whose disk copy differs from the index, or None when git fails."""
    done = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "--no-renames", "-z"],
        capture_output=True, text=True, env=dict(environ), check=False)
    if done.returncode != 0:
        return None
    return [name for name in done.stdout.split("\0") if name]


def staged(root: Path, path: Path, environ: Mapping[str, str]) -> bool:
    """Returns whether the index holds *path*."""
    done = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path.as_posix()],
        capture_output=True, text=True, env=dict(environ), check=False)
    return done.returncode == 0


def refusal(root: Path, environ: Mapping[str, str]) -> str | None:
    """Returns why the repository at *root* cannot be checked in place, or None when it can."""
    differing = unstaged_files(root, environ)
    if differing is None:
        return "git cannot read the index"
    if differing:
        names = "\n  ".join(differing)
        return (f"these files differ on disk from the index; stage or restore them:\n  {names}")
    if not staged(root, K_CHECK_ALL, environ):
        return f"the index holds no {K_CHECK_ALL.as_posix()}"
    return None


def run(root: Path, environ: Mapping[str, str] | None = None) -> int:
    """Returns the exit code of `check_all.py` run in place on the repository at *root*.

    The checkers that read every commit are left out: a commit is not in the history yet, and the
    `commit-msg` hook judges its message and its pages.
    """
    environ = dict(os.environ if environ is None else environ)
    git_dir = absolute_git_dir(root, environ)
    if git_dir is None:
        print(f"pre-commit: {root} is not a git repository", file=sys.stderr)
        return K_EXIT_REFUSED
    problem = refusal(root, environ)
    if problem is not None:
        print(f"pre-commit: {problem}", file=sys.stderr)
        return K_EXIT_REFUSED
    tree = root.resolve()
    return subprocess.run([sys.executable, str(tree / K_CHECK_ALL), str(tree), K_SKIP_HISTORY],
                          env=checked_environment(git_dir, tree, environ), check=False).returncode


def main(argv: list[str] | None = None) -> int:
    """Runs the checks on the repository named by the argument, or on the current folder."""
    arguments = sys.argv[1:] if argv is None else argv
    return run(Path(arguments[0]) if arguments else Path.cwd())


if __name__ == "__main__":
    sys.exit(main())

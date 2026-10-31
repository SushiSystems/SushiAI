# repo_builder.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Builds throwaway git repositories for the tests of the checkers that read history."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits.history import Commit  # noqa: E402


class Repo:
    """Holds a git repository in a temporary folder; call close() when the test ends."""

    def __init__(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        self.git("init", "-q")

    def close(self) -> None:
        """Deletes the repository."""
        self._folder.cleanup()

    def git(self, *arguments: str) -> str:
        """Runs git in the repository and returns its output."""
        return subprocess.run(["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
                              capture_output=True, text=True, check=True).stdout

    def write(self, name: str, text: str) -> None:
        """Writes a file below the root, with its folders."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def remove(self, name: str) -> None:
        """Deletes a file below the root."""
        (self.root / name).unlink()

    def commit(self, message: str) -> Commit:
        """Commits every change and returns the commit, with its parent when it has one."""
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        sha = self.git("rev-parse", "HEAD").strip()
        parent = self.git("rev-parse", f"{sha}^").strip() if self._has_parent() else ""
        return Commit(self.root, sha, parent, message)

    def _has_parent(self) -> bool:
        """Returns whether HEAD has a parent commit."""
        return len(self.git("rev-list", "--parents", "-n", "1", "HEAD").split()) > 1

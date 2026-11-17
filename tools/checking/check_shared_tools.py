# check_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks the shared tools of a repository against its lock, `tools/shared_tools.lock`.

The lock names the SushiCore tag the files came from and the SHA-256 of each file as it was
written. SushiCore is the source of the tools and holds no lock, so it passes without one.

Usage: python tools/checking/check_shared_tools.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_LOCK = Path("tools/shared_tools.lock")
K_SOURCE_MARKER = Path("source/sushicore/shared_tools")
K_NO_LOCK = (
    "no tools/shared_tools.lock; run python -m sushicore.shared_tools sync . "
    "--tag v<the version cli/pyproject.toml requires>"
)


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds one repository root, its lock text when it has one, and whether it is SushiCore."""

    root: Path
    is_source: bool
    lock_text: str | None


def _digest(content: bytes) -> str:
    """Returns the SHA-256 of some bytes as hex, the form the lock stores."""
    return hashlib.sha256(content).hexdigest()


def _locked_files(text: str) -> dict[str, str]:
    """Returns the `[files]` table of a lock; raises ValueError when it holds none."""
    files = tomllib.loads(text).get("files")
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(value, str) for path, value in files.items()
    ):
        raise ValueError("no [files] table of path = sha256 strings")
    return files


def _read(root: Path) -> Repository:
    """Returns the subject of one repository root, reading its lock when it has one."""
    lock = root / K_LOCK
    return Repository(
        root,
        (root / K_SOURCE_MARKER).is_dir(),
        lock.read_text(encoding="utf-8") if lock.is_file() else None,
    )


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns one subject per repository root; a file among the paths stands for its folder."""
    return [_read(path if path.is_dir() else path.parent) for path in paths]


def rule_lock_present(repository: Repository) -> Iterator[Issue]:
    """Yields an issue when a repository other than SushiCore holds no lock."""
    if repository.lock_text is None and not repository.is_source:
        yield Issue(repository.root / K_LOCK, 1, K_NO_LOCK)


def rule_lock_parses(repository: Repository) -> Iterator[Issue]:
    """Yields an issue when the lock is not TOML or holds no `[files]` table of strings."""
    if repository.lock_text is None:
        return
    try:
        _locked_files(repository.lock_text)
    except ValueError as error:
        yield Issue(repository.root / K_LOCK, 1, f"not a readable shared tools lock: {error}")


def rule_shared_file_matches(repository: Repository) -> Iterator[Issue]:
    """Yields one issue per locked file that is missing or whose bytes differ from its hash."""
    if repository.lock_text is None:
        return
    try:
        files = _locked_files(repository.lock_text)
    except ValueError:
        return
    for path in sorted(files):
        target = repository.root / path
        if not target.is_file():
            yield Issue(target, 1, "shared file is missing; tools/shared_tools.lock lists it")
        elif _digest(target.read_bytes()) != files[path]:
            yield Issue(target, 1, "shared file differs from its hash in tools/shared_tools.lock")


K_CHECKER = Checker(
    description="Checks the shared tools of a repository against tools/shared_tools.lock.",
    collect=collect,
    rules={
        "rule_lock_present": rule_lock_present,
        "rule_lock_parses": rule_lock_parses,
        "rule_shared_file_matches": rule_shared_file_matches,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

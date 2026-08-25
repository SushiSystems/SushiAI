"""Locate built executables under SushiAI's build tree.

The walk itself is shared -- see :class:`sushicore.discovery.ExecutableIndex`.
``sushiruntime`` and ``sushiblas`` are skipped because sushiai builds
both in-tree, and their executables are not sushiai's to offer.
"""

from __future__ import annotations

from pathlib import Path

from sushicore.discovery import ExecutableIndex

from .. import console

_INDEX = ExecutableIndex(skip_dirs=("sushiruntime", "sushiblas"))

find_executables = _INDEX.find
match_by_name = _INDEX.match


def select_interactive(build_root: Path) -> Path | None:
    """Show a table of executables and prompt the user to pick one."""
    return _INDEX.select(build_root, console)

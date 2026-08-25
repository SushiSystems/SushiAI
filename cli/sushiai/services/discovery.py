"""Locate built executables under SushiAI's build tree.

The walk itself is shared -- see :class:`sushicore.discovery.ExecutableIndex`.
The only thing this module contributes is which directories to prune, and it
does not spell them out either: they are the sibling checkouts PROFILE already
names, because a module that builds a dependency in-tree must not offer that
dependency's executables as its own.
"""

from __future__ import annotations

from pathlib import Path

from sushicore.discovery import ExecutableIndex

from .. import console
from ..config import PROFILE

_INDEX = ExecutableIndex(skip_dirs=PROFILE.sibling_skip_dirs())

find_executables = _INDEX.find
match_by_name = _INDEX.match
resolve = _INDEX.resolve


def select_interactive(build_root: Path) -> Path | None:
    """Show a table of executables and prompt the user to pick one."""
    return _INDEX.select(build_root, console)

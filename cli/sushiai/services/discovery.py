# discovery.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Locates built executables under SushiAI's build tree.

The reasoning is in cli/README.md under "services/discovery.py".
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

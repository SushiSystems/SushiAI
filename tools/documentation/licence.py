# licence.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads the licence a repository's source files carry from tools/settings.py.

The checker of license blocks and the writer of them share this one description, so neither names
a holder, a licence text or an address of its own.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402


@dataclass(frozen=True, slots=True)
class Licence:
    """Holds the holder, the addresses and the licence text of one repository, open and closed."""

    holder: str
    company: str
    site: str
    open_lines: tuple[str, ...]
    closed_lines: tuple[str, ...]
    open_box_lines: tuple[str, ...]
    closed_box_lines: tuple[str, ...]
    closed: bool

    @property
    def copyright_line(self) -> re.Pattern[str]:
        """Returns the pattern of the copyright row of a file."""
        return re.compile(r"^Copyright \(c\) \d{4}-present " + re.escape(self.holder) + "$")

    @property
    def copyright_shape(self) -> str:
        """Returns the copyright row as a message shows it."""
        return f"Copyright (c) <year>-present {self.holder}"

    @property
    def hash_lines(self) -> tuple[str, ...]:
        """Returns the licence rows of the `#` form for the repository's openness."""
        return self.closed_lines if self.closed else self.open_lines

    @property
    def box_lines(self) -> tuple[str, ...]:
        """Returns the licence rows of the boxed form for the repository's openness."""
        return self.closed_box_lines if self.closed else self.open_box_lines

    @property
    def reserved_line(self) -> re.Pattern[str]:
        """Returns the pattern of a row that only this repository's own licence text may hold."""
        pattern = r"^(Licensed under|All rights reserved)"
        return re.compile(f"{pattern}|{re.escape(self.company)}" if self.company else pattern)

    @property
    def own_rows(self) -> frozenset[str]:
        """Returns every licence row the writer owns, open and closed, in both forms."""
        rows = (*self.open_lines, *self.closed_lines, *self.open_box_lines, *self.closed_box_lines)
        return frozenset(row.strip() for row in rows if row.strip())


def load() -> Licence | None:
    """Returns the licence the settings describe, or None when they name no holder."""
    holder: str = getattr(settings, "K_LICENSE_HOLDER", "")
    if not holder:
        return None
    return Licence(
        holder=holder,
        company=getattr(settings, "K_LICENSE_COMPANY", ""),
        site=getattr(settings, "K_LICENSE_SITE", ""),
        open_lines=tuple(getattr(settings, "K_LICENSE_OPEN_LINES", ())),
        closed_lines=tuple(getattr(settings, "K_LICENSE_CLOSED_LINES", ())),
        open_box_lines=tuple(getattr(settings, "K_LICENSE_OPEN_BOX_LINES", ())),
        closed_box_lines=tuple(getattr(settings, "K_LICENSE_CLOSED_BOX_LINES", ())),
        closed=bool(getattr(settings, "K_CLOSED", False)),
    )


K_UNSET_MESSAGE = "tools/settings.py sets no K_LICENSE_HOLDER, so the license block rules cannot run"

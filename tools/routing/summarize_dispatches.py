# summarize_dispatches.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Sums the dispatch lines of every REPORT.md, per model and effort, for the model-routing skill.

Usage: python tools/routing/summarize_dispatches.py [root]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

K_REPORT_NAME = "REPORT.md"
K_REPORT_FOLDERS = ("docs/agent", "docs/archive")
K_LINE = re.compile(
    r"^dispatch: (?P<task>.+?) — (?P<model>\w+) (?P<effort>\w+), (?P<turns>\d+) turns, "
    r"(?P<outcome>passed|failed)(?:, rung (?P<rung>\d+))?\s*$"
)
K_FIRST_RUNG = 1


@dataclass(frozen=True, slots=True)
class Dispatch:
    """Holds one dispatch line of a report."""

    task: str
    model: str
    effort: str
    turns: int
    passed: bool
    rung: int


@dataclass(frozen=True, slots=True)
class Seat:
    """Holds the totals of every dispatch that ran on one model at one effort."""

    model: str
    effort: str
    count: int
    passed: int
    turns: int
    escalated: int

    def format(self) -> str:
        """Returns the seat as one aligned row of the summary."""
        rate = 100 * self.passed // self.count
        return (f"{self.model:<7} {self.effort:<7} {self.count:>5} {rate:>5}% "
                f"{self.turns / self.count:>7.1f} {self.escalated:>9}")


def parse(lines: Iterable[str]) -> Iterator[Dispatch]:
    """Yields every dispatch line among the given lines; other lines are skipped."""
    for line in lines:
        match = K_LINE.match(line.strip())
        if match:
            yield Dispatch(
                match["task"], match["model"], match["effort"], int(match["turns"]),
                match["outcome"] == "passed", int(match["rung"] or K_FIRST_RUNG),
            )


def reports(root: Path) -> Iterator[Path]:
    """Yields every report of agent work below root, live and archived."""
    for name in K_REPORT_FOLDERS:
        folder = root / name
        if folder.is_dir():
            yield from sorted(folder.rglob(K_REPORT_NAME))


def summarize(dispatches: Iterable[Dispatch]) -> list[Seat]:
    """Returns one seat per model and effort, sorted by model then effort."""
    groups: dict[tuple[str, str], list[Dispatch]] = {}
    for dispatch in dispatches:
        groups.setdefault((dispatch.model, dispatch.effort), []).append(dispatch)
    return [
        Seat(model, effort, len(group), sum(item.passed for item in group),
             sum(item.turns for item in group), sum(item.rung > K_FIRST_RUNG for item in group))
        for (model, effort), group in sorted(groups.items())
    ]


def main(argv: list[str] | None = None) -> int:
    """Prints the summary of every dispatch line below a root and returns 0."""
    arguments = sys.argv[1:] if argv is None else argv
    root = Path(arguments[0]) if arguments else Path.cwd()
    dispatches = [
        dispatch for path in reports(root)
        for dispatch in parse(path.read_text(encoding="utf-8", errors="replace").splitlines())
    ]
    if not dispatches:
        print("no dispatch lines found")
        return 0
    print(f"{'model':<7} {'effort':<7} {'runs':>5} {'pass':>6} {'turns':>7} {'escalated':>9}")
    for seat in summarize(dispatches):
        print(seat.format())
    return 0


if __name__ == "__main__":
    sys.exit(main())

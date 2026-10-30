# check_design_format.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the design documents of a repository against the design format of the documentation skill, through the tooling package.

Usage: python tools/documentation/check_design_format.py [repository root] [--report] [--rule NAME]

The rules are those of `design check` in the tooling package's design module that read one document's frame.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, run  # noqa: E402
from documentation.tooling_bridge import design_rule  # noqa: E402

K_DELEGATED = ("sections", "status_length", "phase_table", "no_commit_hashes", "no_execution_record")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of one repository whose design documents are checked."""

    root: Path


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/design tree."""
    return [Repository(path) for path in paths if (path / "docs" / "design").is_dir()]


K_CHECKER = Checker(
    description="Checks the design documents against the design format of the documentation skill.",
    collect=collect,
    rules={f"rule_{name}": design_rule(name, name == K_DELEGATED[0]) for name in K_DELEGATED},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

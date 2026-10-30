# check_workflows.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the workflow files under .github/workflows/ for what GitHub rejects before any job starts.

Usage: python tools/ci/check_workflows.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402

K_WORKFLOW_FOLDER = ".github/workflows"
K_WORKFLOW_SUFFIXES = (".yml", ".yaml")
K_KEY_LINE = re.compile(r"""^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>[^\s#'"\-][^:#]*|"[^"]*"|'[^']*')\s*:(?:\s|$)""")
K_BLOCK_SCALAR = re.compile(r":\s*[|>][+-]?\d*\s*(?:#.*)?$")


@dataclass(frozen=True, slots=True)
class Workflow:
    """Holds one workflow file as its lines."""

    path: Path
    lines: tuple[str, ...]


def collect(paths: list[Path]) -> Iterable[Workflow]:
    """Returns each workflow file under the given repository roots that git tracks."""
    found: list[Path] = []
    for root in (path for path in paths if path.is_dir()):
        folder = root / K_WORKFLOW_FOLDER
        if folder.is_dir():
            found.extend(drop_ignored(root, sorted(p for p in folder.iterdir() if p.suffix in K_WORKFLOW_SUFFIXES)))
    return [Workflow(path, tuple(path.read_text(encoding="utf-8").splitlines())) for path in found]


def rule_duplicate_key(workflow: Workflow) -> Iterator[Issue]:
    """Yields an issue for each mapping key written twice in one block mapping.

    GitHub refuses the whole file for it, so no job starts and the run reads as a file issue;
    YAML libraries accept it and keep the last value, which hides the fault from a parse check.
    """
    scopes: dict[int, set[str]] = {}
    scalar_indent: int | None = None
    for number, line in enumerate(workflow.lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if scalar_indent is not None:
            if indent > scalar_indent:
                continue
            scalar_indent = None
        match = K_KEY_LINE.match(line)
        if match is None:
            continue
        mapping_indent = indent + len(match.group("dash") or "")
        for deeper in [key for key in scopes if key > mapping_indent]:
            del scopes[deeper]
        if match.group("dash"):
            scopes[mapping_indent] = set()
        keys = scopes.setdefault(mapping_indent, set())
        key = match.group("key").strip()
        if key in keys:
            yield Issue(workflow.path, number, f"key '{key}' appears twice in one mapping")
        keys.add(key)
        if K_BLOCK_SCALAR.search(stripped):
            scalar_indent = indent


K_CHECKER = Checker(
    description="Checks workflow files for what GitHub rejects before any job starts.",
    collect=collect,
    rules={
        "rule_duplicate_key": rule_duplicate_key,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

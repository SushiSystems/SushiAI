# check_colour_names.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks that no string of the CLI package names a raw Rich colour in its markup.

Colour comes from the theme of sushicore's console, so a `[green]` in a message bypasses it and
shows in the wrong colour on a terminal with another theme.

Usage: python tools/cli_source/check_colour_names.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_PACKAGE = Path("cli") / "sushiai"
K_RAW_COLOUR = re.compile(r"\[/?(green|red|yellow|cyan|blue|dim)[\] ]")


def collect(paths: list[Path]) -> Iterable[Path]:
    """Returns every Python file of the CLI package of each given repository."""
    return [file for path in paths if (path / K_PACKAGE).is_dir()
            for file in sorted((path / K_PACKAGE).rglob("*.py"))]


def rule_raw_colour_name(file: Path) -> Iterator[Issue]:
    """Yields an issue for each string literal of a file that names a raw Rich colour."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and K_RAW_COLOUR.search(node.value)):
            yield Issue(file, node.lineno, "a string names a raw Rich colour; use the console theme")


K_CHECKER = Checker(
    description="Checks that no string of the CLI package names a raw Rich colour.",
    collect=collect,
    rules={"rule_raw_colour_name": rule_raw_colour_name},
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

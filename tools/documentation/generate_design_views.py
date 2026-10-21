# generate_design_views.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Writes the generated views of the design topics: the topic map, a Designs block per module and a work block per topic.

Usage: python tools/documentation/generate_design_views.py [repository root] [--init] [--check]

A view is written between its begin and end markers and nowhere else. `--init` first adds the markers
where a file has none. `--check` writes nothing and exits 1 when a block is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.design_views import (  # noqa: E402
    K_DESIGNS_BLOCK,
    K_TOPICS_BLOCK,
    expected_blocks,
    insert_block,
    read_block,
    write_block,
)

K_HEADINGS = {K_TOPICS_BLOCK: "## Topics", K_DESIGNS_BLOCK: "## Designs"}
K_EXIT_CLEAN = 0
K_EXIT_STALE = 1


def _parse(argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--init", action="store_true", help="add the markers a file lacks")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 when a block is stale")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Writes or checks every generated block and returns 1 when --check finds a stale one."""
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    root = arguments.root.resolve()
    stale = 0
    for path, name, body in expected_blocks(root):
        if arguments.init and not arguments.check:
            insert_block(path, name, K_HEADINGS.get(name))
        current = read_block(path, name)
        if current is None or current == body:
            continue
        if arguments.check:
            print(f"{path.relative_to(root).as_posix()}: the block '{name}' is stale")
            stale += 1
        else:
            write_block(path, name, body)
    return K_EXIT_STALE if stale else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

# generate_design_views.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
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

from documentation.design_status import read_tree  # noqa: E402
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
    for name, block, body in expected_blocks(read_tree(root)):
        path = root / name
        text = path.read_text(encoding="utf-8-sig")
        if arguments.init and not arguments.check:
            added = insert_block(text, block, K_HEADINGS.get(block))
            if added is not None:
                _write(path, added)
                text = added
        current = read_block(text, block)
        if current is None or current == body:
            continue
        if arguments.check:
            print(f"{name}: the block '{block}' is stale")
            stale += 1
        else:
            _write(path, write_block(text, block, body))
    return K_EXIT_STALE if stale else K_EXIT_CLEAN


def _write(path: Path, text: str) -> None:
    """Writes a view file as UTF-8 with its line endings kept."""
    path.write_text(text, encoding="utf-8", newline="")


if __name__ == "__main__":
    sys.exit(main())

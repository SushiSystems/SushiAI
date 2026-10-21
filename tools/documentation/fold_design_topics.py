# fold_design_topics.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Gives each design document that sits loose in docs/design a folder of its own, with the document as its README.md.

Usage: python tools/documentation/fold_design_topics.py [repository root] [--apply]

Links out of the moved file and every citation of it are rewritten. Nothing is changed without `--apply`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.design_topics import loose_documents  # noqa: E402
from documentation.repair_design_citations import run as repair  # noqa: E402

K_LINK = re.compile(r"\]\((?P<target>[^)#\s]+)(?P<anchor>#[^)\s]*)?\)")
K_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|/)")
K_EXIT_CLEAN = 0
K_EXIT_PENDING = 1


def relink(text: str, old_dir: Path, new_dir: Path) -> str:
    """Returns the text with each relative link rewritten so that it still names the same file from the new folder."""

    def fix(match: re.Match[str]) -> str:
        """Returns one link as it should read from the new folder."""
        target = match["target"]
        if K_EXTERNAL.match(target):
            return match.group(0)
        absolute = (old_dir / target).resolve()
        relative = os.path.relpath(absolute, new_dir).replace(os.sep, "/")
        return f"]({relative}{match['anchor'] or ''})"

    return K_LINK.sub(fix, text)


def _move(root: Path, old: Path, new: Path) -> None:
    """Moves a file with git when the repository is one, and with the file system otherwise."""
    new.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["git", "-C", str(root), "mv", str(old), str(new)], capture_output=True, text=True)
    if done.returncode != 0:
        old.rename(new)


def fold(root: Path, apply: bool) -> dict[str, str]:
    """Folds each loose document and returns the moves, old to new, as paths from the root."""
    moves: dict[str, str] = {}
    for old in loose_documents(root):
        new = old.with_suffix("") / "README.md"
        moves[old.relative_to(root).as_posix()] = new.relative_to(root).as_posix()
        if not apply:
            continue
        text = old.read_text(encoding="utf-8")
        _move(root, old, new)
        new.write_text(relink(text, old.parent, new.parent), encoding="utf-8", newline="")
    if apply and moves:
        repair(root, True, moves)
    return moves


def main(argv: list[str] | None = None) -> int:
    """Prints each document that is or would be folded; returns 1 for a dry run that finds one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--apply", action="store_true", help="move the documents; without it nothing is changed")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    moves = fold(arguments.root.resolve(), arguments.apply)
    for old, new in moves.items():
        print(f"{old} -> {new}{'' if arguments.apply else ' (dry run)'}")
    return K_EXIT_PENDING if moves and not arguments.apply else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

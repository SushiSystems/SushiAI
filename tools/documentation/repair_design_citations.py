# repair_design_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Repairs the citations of design documents that moved when a topic became a folder or a folder became a file.

Usage: python tools/documentation/repair_design_citations.py [repository root] [--apply] [--move OLD NEW]...

A citation whose file is gone but whose topic exists is rewritten to the file that holds the topic; one
the tool cannot place is reported and left alone. `--move` names a file that was moved or merged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import citing_files  # noqa: E402
from documentation.design_topics import topics  # noqa: E402

K_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?P<docs>docs/)?design/(?P<tail>[A-Za-z0-9_]+(?:\.md)?(?:/[A-Za-z0-9_.-]+)*)")
K_LINK = re.compile(r"\]\((?P<target>[A-Za-z0-9_./-]+\.md)(?P<anchor>#[^)\s]*)?\)")
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1


@dataclass(frozen=True, slots=True)
class Repair:
    """Holds one citation, the line it is on and the text that replaces it, or no replacement when it cannot be placed."""

    path: Path
    line: int
    old: str
    new: str | None


def _topic_tail(name: str, known: dict[str, Path], design: Path) -> str | None:
    """Returns the path of a topic's document below docs/design, or None when there is no such topic."""
    document = known.get(name)
    return None if document is None else document.relative_to(design).as_posix()


def _fixed(tail: str, root: Path, known: dict[str, Path]) -> str | None:
    """Returns the tail of the document that now holds what the tail names, or None when it cannot be placed."""
    design = root / "docs" / "design"
    if (design / tail).exists():
        return tail
    name = tail.split("/")[0].removesuffix(".md")
    if tail not in (f"{name}.md", f"{name}/README.md"):
        return None
    return _topic_tail(name, known, design)


def repair_text(
    path: Path, text: str, root: Path, known: dict[str, Path], moves: dict[str, str] | None = None
) -> tuple[str, list[Repair]]:
    """Returns the text with each moved citation rewritten, and one Repair for each citation that was wrong.

    A move names a file below the root, old to new, and is applied whether or not the old file is there.
    """
    found: list[Repair] = []
    design = root / "docs" / "design"
    moved = moves or {}
    base = root.resolve()

    def line_of(match: re.Match[str]) -> int:
        """Returns the line a match starts on."""
        return text.count("\n", 0, match.start()) + 1

    def fix_path(match: re.Match[str]) -> str:
        """Returns the replacement for a path citation and records it when it was wrong."""
        tail = match["tail"].rstrip(".")
        suffix = match["tail"][len(tail):]
        prefix = match["docs"] or ""
        cited = f"{prefix}design/{tail}"
        if f"docs/design/{tail}" in moved:
            new = moved[f"docs/design/{tail}"] if prefix else moved[f"docs/design/{tail}"].removeprefix("docs/")
            found.append(Repair(path, line_of(match), cited, new))
            return f"{new}{suffix}"
        fixed = _fixed(tail, root, known)
        if fixed == tail:
            return match.group(0)
        found.append(Repair(path, line_of(match), cited, None if fixed is None else f"{prefix}design/{fixed}"))
        return match.group(0) if fixed is None else f"{prefix}design/{fixed}{suffix}"

    def fix_link(match: re.Match[str]) -> str:
        """Returns the replacement for a link to a design document and records it when it was wrong."""
        target = (path.parent / match["target"]).resolve()
        below = target.relative_to(base).as_posix() if base in target.parents else ""
        if below in moved:
            replacement = os.path.relpath(root / moved[below], path.parent).replace(os.sep, "/")
            found.append(Repair(path, line_of(match), match["target"], replacement))
            return f"]({replacement}{match['anchor'] or ''})"
        if target.exists() or design.resolve() not in target.parents:
            return match.group(0)
        tail = target.relative_to(design.resolve()).as_posix()
        fixed = _fixed(tail, root, known)
        line = line_of(match)
        if fixed is None:
            found.append(Repair(path, line, match["target"], None))
            return match.group(0)
        replacement = os.path.relpath(design / fixed, path.parent).replace(os.sep, "/")
        found.append(Repair(path, line, match["target"], replacement))
        return f"]({replacement}{match['anchor'] or ''})"

    return K_PATH.sub(fix_path, K_LINK.sub(fix_link, text)), found


def run(root: Path, apply: bool, moves: dict[str, str] | None = None) -> list[Repair]:
    """Repairs, or with apply False only lists, every moved citation below root."""
    known = topics(root / "docs")
    found: list[Repair] = []
    for path in citing_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not moves and "design/" not in text:
            continue
        fixed, repairs = repair_text(path, text, root, known, moves)
        found += repairs
        if apply and fixed != text:
            path.write_text(fixed, encoding="utf-8", newline="")
    return found


def main(argv: list[str] | None = None) -> int:
    """Prints each citation that was or would be repaired and each one that cannot be placed; returns 1 for the latter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--apply", action="store_true", help="write the repairs; without it nothing is written")
    parser.add_argument("--move", nargs=2, action="append", default=[], metavar=("OLD", "NEW"), help="rewrite citations of the file OLD to NEW, paths from the root; repeatable")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = arguments.root.resolve()
    unplaced = 0
    for repair in run(root, arguments.apply, dict(arguments.move)):
        shown = repair.path.relative_to(root).as_posix()
        if repair.new is None:
            print(f"{shown}:{repair.line}: cannot place '{repair.old}'")
            unplaced += 1
        else:
            print(f"{shown}:{repair.line}: '{repair.old}' -> '{repair.new}'{'' if arguments.apply else ' (dry run)'}")
    return K_EXIT_FINDINGS if unplaced else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

# repair_design_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Repairs the citations of design documents that moved, between a file and a folder or into and out of _REVISION.

Usage: python tools/documentation/repair_design_citations.py [repository root] [--apply] [--move OLD NEW]...

A gone citation whose topic exists is rewritten to the topic's file or page; one it cannot place is reported. A
moved page's links are read from its former place. `--move` names a file that was moved or merged.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import citing_files  # noqa: E402
from documentation.design_status import K_REVISION_FOLDER, read_tree  # noqa: E402
from documentation.design_topics import topics  # noqa: E402

K_DESIGN = "docs/design"
K_DESIGN_PREFIX = f"{K_DESIGN}/"
K_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?P<docs>docs/)?design/(?P<tail>[A-Za-z0-9_]+(?:\.md)?(?:/[A-Za-z0-9_.-]+)*)")
K_LINK = re.compile(r"\]\((?P<target>[A-Za-z0-9_./-]+)(?P<anchor>#[^)\s]*)?\)")
K_README = "README.md"
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1


@dataclass(frozen=True, slots=True)
class Repair:
    """Holds one citation, the line it is on and the text that replaces it, or no replacement when it cannot be placed."""

    path: str
    line: int
    old: str
    new: str | None


def relative_name(target: str, start: str) -> str:
    """Returns the path from the folder start to the file target, both root-relative POSIX names, as os.path.relpath would."""
    goal = [part for part in target.split("/") if part not in ("", ".")]
    origin = [part for part in start.split("/") if part not in ("", ".")]
    shared = 0
    while shared < min(len(goal), len(origin)) and goal[shared] == origin[shared]:
        shared += 1
    parts = [".."] * (len(origin) - shared) + goal[shared:]
    return "/".join(parts) if parts else "."


def _joined(folder: str, target: str) -> str:
    """Returns the root-relative name a target names when written in a folder, with its dot segments resolved."""
    return posixpath.normpath(posixpath.join(folder, target))


def is_scanned(path: str, text: str, moves: Mapping[str, str] | None) -> bool:
    """Returns whether a file can hold a citation to repair: a moved file is looked for everywhere, a design file always."""
    return bool(moves) or "design/" in text or path.startswith(K_DESIGN_PREFIX)


def _in_design(tail: str, present: frozenset[str]) -> bool:
    """Returns whether a name below docs/design exists in the repository."""
    return f"{K_DESIGN_PREFIX}{tail}" in present


def _topic_tail(name: str, known: Mapping[str, str]) -> str | None:
    """Returns the path of a topic's document below docs/design, or None when there is no such topic."""
    document = known.get(name)
    return None if document is None else document.removeprefix(K_DESIGN_PREFIX)


def _fixed(tail: str, present: frozenset[str], known: Mapping[str, str]) -> str | None:
    """Returns the tail of the document that now holds what the tail names, or None when it cannot be placed."""
    if _in_design(tail, present):
        return tail
    parts = tail.split("/")
    if parts[0] == K_REVISION_FOLDER:
        parts = parts[1:]
    if not parts:
        return None
    name = parts[0].removesuffix(".md")
    holder = _topic_tail(name, known)
    page = "/".join(parts[1:])
    if holder is None or (page and parts[0] != name):
        return None
    if not page and parts[0] == name and holder.endswith(f"/{K_README}"):
        return holder.removesuffix(f"/{K_README}")
    if page in ("", K_README):
        return holder
    if not holder.endswith(f"/{K_README}"):
        return None
    moved = f"{holder.removesuffix(K_README)}{page}"
    return moved if _in_design(moved, present) else None


def _former_folder(path: str) -> str | None:
    """Returns the folder a design page sat in before it moved into or out of the revision folder."""
    folder = posixpath.dirname(path)
    if not folder.startswith(K_DESIGN_PREFIX):
        return None
    below = folder.removeprefix(K_DESIGN_PREFIX).split("/")
    if below[0] == K_REVISION_FOLDER:
        return "/".join([K_DESIGN, *below[1:]])
    return "/".join([K_DESIGN, K_REVISION_FOLDER, *below])


def repair_text(
    path: str,
    text: str,
    present: frozenset[str],
    known: Mapping[str, str],
    moves: Mapping[str, str] | None = None,
) -> tuple[str, list[Repair]]:
    """Returns the text with each moved citation rewritten, and one Repair for each citation that was wrong.

    The file is named by its root-relative path, present holds every name below the root that exists, and known
    maps each topic to its document. A move names a file below the root, old to new, and is applied whether or
    not the old file is there.
    """
    found: list[Repair] = []
    moved = moves or {}
    folder = posixpath.dirname(path)

    def line_of(match: re.Match[str]) -> int:
        """Returns the line a match starts on."""
        return text.count("\n", 0, match.start()) + 1

    def fix_path(match: re.Match[str]) -> str:
        """Returns the replacement for a path citation and records it when it was wrong."""
        tail = match["tail"].rstrip(".")
        suffix = match["tail"][len(tail):]
        prefix = match["docs"] or ""
        cited = f"{prefix}design/{tail}"
        if f"{K_DESIGN_PREFIX}{tail}" in moved:
            new = moved[f"{K_DESIGN_PREFIX}{tail}"] if prefix else moved[f"{K_DESIGN_PREFIX}{tail}"].removeprefix("docs/")
            found.append(Repair(path, line_of(match), cited, new))
            return f"{new}{suffix}"
        fixed = _fixed(tail, present, known)
        if fixed == tail:
            return match.group(0)
        found.append(Repair(path, line_of(match), cited, None if fixed is None else f"{prefix}design/{fixed}"))
        return match.group(0) if fixed is None else f"{prefix}design/{fixed}{suffix}"

    def fix_link(match: re.Match[str]) -> str:
        """Returns the replacement for a link to a design document and records it when it was wrong."""
        target = _joined(folder, match["target"])
        below = "" if target == ".." or target.startswith("../") else target
        if below in moved:
            replacement = relative_name(moved[below], folder)
            found.append(Repair(path, line_of(match), match["target"], replacement))
            return f"]({replacement}{match['anchor'] or ''})"
        former = None if target in present else _former_folder(path)
        earlier = None if former is None else _joined(former, match["target"])
        if earlier is not None and earlier in present:
            replacement = relative_name(earlier, folder)
            found.append(Repair(path, line_of(match), match["target"], replacement))
            return f"]({replacement}{match['anchor'] or ''})"
        if target in present or not target.startswith(K_DESIGN_PREFIX):
            return match.group(0)
        fixed = _fixed(target.removeprefix(K_DESIGN_PREFIX), present, known)
        line = line_of(match)
        if fixed is None:
            found.append(Repair(path, line, match["target"], None))
            return match.group(0)
        replacement = relative_name(f"{K_DESIGN}/{fixed}", folder)
        found.append(Repair(path, line, match["target"], replacement))
        return f"]({replacement}{match['anchor'] or ''})"

    return K_PATH.sub(fix_path, K_LINK.sub(fix_link, text)), found


def _names(root: Path) -> frozenset[str]:
    """Returns every file and folder below the root as a root-relative POSIX name, git's own folder left out."""
    found: set[str] = set()
    for folder, folders, files in os.walk(root):
        folders[:] = [name for name in folders if name != ".git"]
        found.update(Path(folder, name).relative_to(root).as_posix() for name in (*folders, *files))
    return frozenset(found)


def run(root: Path, apply: bool, moves: dict[str, str] | None = None) -> list[Repair]:
    """Repairs, or with apply False only lists, every moved citation below root."""
    present = _names(root)
    known = topics(read_tree(root))
    found: list[Repair] = []
    for path in citing_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        name = path.relative_to(root).as_posix()
        if not is_scanned(name, text, moves):
            continue
        fixed, repairs = repair_text(name, text, present, known, moves)
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
        if repair.new is None:
            print(f"{repair.path}:{repair.line}: cannot place '{repair.old}'")
            unplaced += 1
        else:
            print(f"{repair.path}:{repair.line}: '{repair.old}' -> '{repair.new}'{'' if arguments.apply else ' (dry run)'}")
    return K_EXIT_FINDINGS if unplaced else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

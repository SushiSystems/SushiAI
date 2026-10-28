# drop_retired_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Replaces each citation of a document that version control keeps and the tree does not with a phrase.

Usage: python tools/documentation/drop_retired_citations.py [repository root] --prefix PATH... [--apply]

A prefix is a folder below docs/ whose files are gone, such as `docs/archive`. A link to one of its files
becomes its text and a path becomes a phrase naming the file. Nothing is written without `--apply`.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import citing_files  # noqa: E402

K_DATE_PREFIX = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}[-_]")
K_KINDS = {"specs": "spec", "plans": "plan", "research": "research note", "notes": "note", "reports": "report"}
K_WORK_FILES = {"SPEC.md": "spec", "PLAN.md": "plan", "REPORT.md": "report"}
K_WORDS = re.compile(r"[-_]+")
K_SUFFIX = "in the version history"
K_REACH = r"(?:(?:\.\./)*docs/|(?:\.\./)+|\./)"
K_RELATIVE_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[A-Za-z0-9_./-]+\.md)(?P<anchor>#[^)\s]*)?\)")
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1


@dataclass(frozen=True, slots=True)
class Drop:
    """Holds one citation that was or would be replaced, with the line it is on and its replacement."""

    path: Path
    line: int
    old: str
    new: str


def phrase(cited: str) -> str:
    """Returns the words that stand for a cited file: its name and its kind, ending with the version history."""
    parts = cited.split("/")
    name = parts[-1]
    if name in K_WORK_FILES and len(parts) >= 2:
        label, kind = parts[-2], K_WORK_FILES[name]
    elif name == "README.md" and len(parts) >= 2:
        label, kind = parts[-2], "document"
    else:
        label, kind = name.removesuffix(".md"), K_KINDS.get(parts[-2], "document") if len(parts) >= 2 else "document"
    words = K_WORDS.sub(" ", K_DATE_PREFIX.sub("", label)).lower().strip().removeprefix("the ")
    return f"the {words} {kind} {K_SUFFIX}"


def _tail(prefix: str) -> str:
    """Returns the part of a prefix below docs/, with its closing slash."""
    return prefix.removeprefix("docs/").strip("/") + "/"


def citation_patterns(prefixes: list[str]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Returns the pattern of a path below any prefix and the pattern of a Markdown link to one."""
    tails = "|".join(re.escape(_tail(prefix)) for prefix in prefixes)
    path = rf"{K_REACH}(?:{tails})[A-Za-z0-9_./-]*\.md"
    link = rf"\[(?P<text>[^\]]*)\]\((?P<path>{path})(?P<anchor>#[^)\s]*)?\)"
    return re.compile(rf"(?P<path>{path})"), re.compile(link)


def _docs_relative(cited: str) -> str:
    """Returns a cited path from the docs/ folder, without the reach in front of it."""
    return re.sub(rf"^{K_REACH}", "", cited)


def drop_text(path: Path, text: str, prefixes: list[str], root: Path | None = None) -> tuple[str, list[Drop]]:
    """Returns the text with each citation of a retired file replaced, and one Drop for each.

    With the repository root, a relative link that resolves into a retired folder is replaced too,
    whatever folder it is written from.
    """
    path_pattern, link_pattern = citation_patterns(prefixes)
    found: list[Drop] = []
    retired = [(root / prefix).resolve() for prefix in prefixes] if root is not None else []

    def line_of(position: int) -> int:
        """Returns the line a position is on."""
        return text.count("\n", 0, position) + 1

    def replace_link(match: re.Match[str]) -> str:
        """Returns the text of a link, or the phrase when the text is itself the path."""
        cited = match["path"]
        words = phrase(_docs_relative(cited))
        inner = path_pattern.search(match["text"])
        new = words if inner else match["text"]
        found.append(Drop(path, line_of(match.start()), match.group(0), new))
        return new

    def replace_path(match: re.Match[str]) -> str:
        """Returns the phrase for a path, taking a code span's backticks with it."""
        words = phrase(_docs_relative(match["path"]))
        found.append(Drop(path, line_of(match.start()), match["path"], words))
        return words

    def replace_relative(match: re.Match[str]) -> str:
        """Returns the text of a relative link that resolves into a retired folder, or the link unchanged."""
        target = (path.parent / match["target"]).resolve()
        if not any(target == folder or folder in target.parents for folder in retired):
            return match.group(0)
        docs = (root / "docs").resolve() if root is not None else target.parent
        cited = target.relative_to(docs).as_posix() if docs in target.parents else target.name
        new = phrase(cited) if match["text"].strip("`") in (match["target"], cited) else match["text"]
        found.append(Drop(path, line_of(match.start()), match.group(0), new))
        return new

    without_links = link_pattern.sub(replace_link, text)
    if retired:
        without_links = K_RELATIVE_LINK.sub(replace_relative, without_links)
    without_spans = re.sub(rf"`({path_pattern.pattern})`", lambda m: replace_path(m), without_links)
    return path_pattern.sub(replace_path, without_spans), found


def run(root: Path, prefixes: list[str], apply: bool) -> list[Drop]:
    """Replaces, or with apply False only lists, every citation of a retired file below root."""
    found: list[Drop] = []
    for path in citing_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fixed, drops = drop_text(path, text, prefixes, root)
        found += drops
        if apply and fixed != text:
            path.write_text(fixed, encoding="utf-8", newline="")
    return found


def main(argv: list[str] | None = None) -> int:
    """Prints each citation that was or would be replaced; returns 1 when it would be and --apply is off."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="repository root; default: .")
    parser.add_argument("--prefix", action="append", required=True, help="a retired folder below docs/; repeatable")
    parser.add_argument("--apply", action="store_true", help="write the replacements; without it nothing is written")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = arguments.root.resolve()
    drops = run(root, arguments.prefix, arguments.apply)
    for drop in drops:
        print(f"{drop.path.relative_to(root).as_posix()}:{drop.line}: '{drop.old}' -> '{drop.new}'")
    return K_EXIT_FINDINGS if drops and not arguments.apply else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

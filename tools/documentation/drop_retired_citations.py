# drop_retired_citations.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Replaces each citation of a document that version control keeps and the tree does not with a phrase.

Usage: python tools/documentation/drop_retired_citations.py [repository root] --prefix PATH... [--apply]

A prefix is a folder below docs/ whose files are gone, such as `docs/archive`. A link to one of its files
becomes its text and a path becomes a phrase naming the file. Nothing is written without `--apply`.
"""

from __future__ import annotations

import argparse
import posixpath
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
K_DOCS = "docs"
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1


@dataclass(frozen=True, slots=True)
class Drop:
    """Holds one citation that was or would be replaced, with the line it is on and its replacement."""

    path: str
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


def citation_patterns(prefixes: list[str]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Returns the pattern of a path below any prefix and the pattern of a Markdown link to one."""
    tails = "|".join(re.escape(_tail(prefix)) for prefix in prefixes)
    path = rf"{K_REACH}(?:{tails})[A-Za-z0-9_./-]*\.md"
    link = rf"\[(?P<text>[^\]]*)\]\((?P<path>{path})(?P<anchor>#[^)\s]*)?\)"
    return re.compile(rf"(?P<path>{path})"), re.compile(link)


def drop_text(
    name: str, text: str, prefixes: list[str], retired: frozenset[str] | None = None
) -> tuple[str, list[Drop]]:
    """Returns the text with each citation of a retired file replaced, and one Drop for each.

    ``name`` is the root-relative POSIX path of the file the text is in. With ``retired``, the
    root-relative folders that version control keeps, a relative link that resolves into one of
    them is replaced too, whatever folder it is written from.
    """
    path_pattern, link_pattern = citation_patterns(prefixes)
    found: list[Drop] = []

    def line_of(position: int) -> int:
        """Returns the line a position is on."""
        return text.count("\n", 0, position) + 1

    def replace_link(match: re.Match[str]) -> str:
        """Returns the text of a link, or the phrase when the text is itself the path."""
        words = phrase(_docs_relative(match["path"]))
        inner = path_pattern.search(match["text"])
        new = words if inner else match["text"]
        found.append(Drop(name, line_of(match.start()), match.group(0), new))
        return new

    def replace_path(match: re.Match[str]) -> str:
        """Returns the phrase for a path, taking a code span's backticks with it."""
        words = phrase(_docs_relative(match["path"]))
        found.append(Drop(name, line_of(match.start()), match["path"], words))
        return words

    def replace_relative(match: re.Match[str]) -> str:
        """Returns the text of a relative link that resolves into a retired folder, or the link unchanged."""
        resolved = _resolved(name, match["target"], retired or frozenset())
        if resolved is None:
            return match.group(0)
        cited = _docs_cited(resolved)
        new = phrase(cited) if match["text"].strip("`") in (match["target"], cited) else match["text"]
        found.append(Drop(name, line_of(match.start()), match.group(0), new))
        return new

    without_links = link_pattern.sub(replace_link, text)
    if retired is not None:
        without_links = K_RELATIVE_LINK.sub(replace_relative, without_links)
    without_spans = re.sub(rf"`({path_pattern.pattern})`", lambda m: replace_path(m), without_links)
    return path_pattern.sub(replace_path, without_spans), found


def run(root: Path, prefixes: list[str], apply: bool) -> list[Drop]:
    """Replaces, or with apply False only lists, every citation of a retired file below root."""
    retired = frozenset(posixpath.normpath(prefix) for prefix in prefixes)
    found: list[Drop] = []
    for path in citing_files(root):
        text = _read_text(path)
        if text is None:
            continue
        fixed, drops = drop_text(path.relative_to(root).as_posix(), text, prefixes, retired)
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
        print(f"{drop.path}:{drop.line}: '{drop.old}' -> '{drop.new}'")
    return K_EXIT_FINDINGS if drops and not arguments.apply else K_EXIT_CLEAN


def _docs_relative(cited: str) -> str:
    """Returns a cited path from the docs/ folder, without the reach in front of it."""
    return re.sub(rf"^{K_REACH}", "", cited)


def _tail(prefix: str) -> str:
    """Returns the part of a prefix below docs/, with its closing slash."""
    return prefix.removeprefix(f"{K_DOCS}/").strip("/") + "/"


def _resolved(name: str, target: str, retired: frozenset[str]) -> str | None:
    """Returns the root-relative path a relative link reaches when that path is in a retired folder, else None."""
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
    if any(resolved == folder or resolved.startswith(f"{folder}/") for folder in retired):
        return resolved
    return None


def _docs_cited(resolved: str) -> str:
    """Returns a root-relative path as a citation names it: from docs/ when it is below docs/, else its file name."""
    return resolved.removeprefix(f"{K_DOCS}/") if resolved.startswith(f"{K_DOCS}/") else posixpath.basename(resolved)


def _read_text(path: Path) -> str | None:
    """Returns the UTF-8 text of a file, or None when it is not UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


if __name__ == "__main__":
    sys.exit(main())

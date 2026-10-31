# docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Finds the documentation pages a change leaves behind, for the hook and the checker that share the rule.

A page names what it describes in a `**Describes:**` line of its first lines. A README also describes
the entries of its own folder. A change to a described file, or a new or removed entry of that folder,
needs the page in the same commit, or a `Docs:` line in the message that says why not.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from commits.history import blob_texts, git  # noqa: E402
from documentation import facts  # noqa: E402

K_SINCE: str = getattr(settings, "K_DOCS_FOLLOW_SINCE", "")
K_IGNORED: tuple[str, ...] = tuple(getattr(
    settings, "K_DOCS_FOLLOW_IGNORED", ("docs/agent", "docs/design", "docs/backlog", "docs/archive", "third_party")))
K_HEAD_LINES = 12
K_DESCRIBES = re.compile(r"^\*\*Describes:\*\*\s*(.+)$")
K_SPAN = re.compile(r"`([^`]+)`")
K_ACKNOWLEDGED = re.compile(r"^Docs:[ \t]*(.+?)[ \t]+unchanged[ \t]+\(\s*(\S.*?)\s*\)[ \t]*$", re.MULTILINE)
K_WAIVED = re.compile(r"^Docs:[ \t]*none[ \t]+\(\s*(\S.*?)\s*\)[ \t]*$", re.MULTILINE)
K_README = "README.md"


@dataclass(frozen=True, slots=True)
class Change:
    """Holds one path a commit touches: its status letter, its path and, for a rename, the old path."""

    status: str
    path: str
    old_path: str = ""


def describes_of(text: str) -> tuple[str, ...]:
    """Returns the paths in the `**Describes:**` line among the first lines of a page."""
    for line in text.splitlines()[:K_HEAD_LINES]:
        found = K_DESCRIBES.match(line)
        if found:
            return tuple(item.strip("/") for item in K_SPAN.findall(found.group(1)))
    return ()


def parse_changes(listing: str) -> list[Change]:
    """Returns the changes of a `git diff --name-status` listing."""
    changes = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].startswith("R"):
            changes.append(Change("R", parts[2], parts[1]))
        elif len(parts) >= 2:
            changes.append(Change(parts[0][0], parts[1]))
    return changes


def acknowledged(message: str) -> tuple[bool, set[str]]:
    """Returns whether the message waives every page, and the pages it says are unchanged."""
    return bool(K_WAIVED.search(message)), {match[0].strip("` ") for match in K_ACKNOWLEDGED.findall(message)}


def _is_ignored(path: str) -> bool:
    """Returns whether a path lies in a folder that no README follows."""
    return any(path == entry or path.startswith(entry + "/") for entry in K_IGNORED)


def _covers(entry: str, path: str) -> bool:
    """Returns whether a described file or folder is the path or holds it."""
    return path == entry or path.startswith(entry + "/")


def _exists(root: Path, revision: str | None, path: str) -> bool:
    """Returns whether a file or folder exists in a revision, or in the index when revision is None."""
    if revision is None:
        return bool((git(root, "ls-files", "--", path) or "").strip())
    return git(root, "cat-file", "-e", f"{revision}:{path}") is not None


def _text(root: Path, revision: str | None, path: str) -> str:
    """Returns a file's text in a revision, or in the index when revision is None; empty when it is absent."""
    return git(root, "show", f"{revision or ''}:{path}") or ""


K_PAGE_CACHE: dict[tuple[Path, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _listing(root: Path, revision: str | None) -> list[tuple[str, str]]:
    """Returns the object hash and the path of every file of a revision, or of the index when revision is None."""
    listing = git(root, "ls-tree", "-r", revision) if revision else git(root, "ls-files", "-s")
    entries = []
    for line in (listing or "").splitlines():
        info, _, name = line.partition("\t")
        fields = info.split()
        if name and len(fields) >= 3:
            entries.append((fields[2] if revision else fields[1], name))
    return entries


def _pages(root: Path, revision: str | None) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Returns every followed markdown page of a revision (or the index) with what it describes and follows.

    A page is read once for each version of it: the pages are cached by object hash and the ones not yet
    seen are read in a single git call.
    """
    entries = [(blob, name) for blob, name in _listing(root, revision)
               if name.endswith(".md") and not _is_ignored(name)]
    missing = sorted({blob for blob, _ in entries if (root, blob) not in K_PAGE_CACHE})
    texts = blob_texts(root, missing)
    for blob in missing:
        text = texts.get(blob, "")
        K_PAGE_CACHE[(root, blob)] = (describes_of(text),
                                      tuple(dict.fromkeys(fact_id for fact_id, _ in facts.parse("", text).follows)))
    return {name: K_PAGE_CACHE[(root, blob)] for blob, name in entries}


def _changed_facts(root: Path, changes: list[Change], before: str | None, after: str | None) -> set[str]:
    """Returns the ids of the facts whose text, or whose existence, differs between the two sides of a change."""
    changed: set[str] = set()
    for change in changes:
        if not change.path.endswith(".md"):
            continue
        old = {block.id: block.body for block in facts.parse("", _text(root, before, change.old_path or change.path)).facts}
        new = {block.id: block.body for block in facts.parse("", _text(root, after, change.path)).facts}
        changed |= {fact_id for fact_id in old.keys() | new.keys() if old.get(fact_id) != new.get(fact_id)}
    return changed


def _moves(change: Change) -> list[tuple[str, bool]]:
    """Returns the paths a change makes appear (True) or disappear (False); an edit makes none."""
    if change.status == "A":
        return [(change.path, True)]
    if change.status == "D":
        return [(change.path, False)]
    if change.status == "R":
        return [(change.path, True), (change.old_path, False)]
    return []


def _entry_changes(root: Path, page: str, before: str | None, after: str | None, changes: list[Change]) -> list[str]:
    """Returns the entries of a README's folder that the changes add or remove whole."""
    folder = PurePosixPath(page).parent.as_posix()
    prefix = "" if folder == "." else folder + "/"
    found: list[str] = []
    for change in changes:
        for path, appears in _moves(change):
            if not path.startswith(prefix) or path == page or _is_ignored(path):
                continue
            entry = prefix + path[len(prefix):].split("/", 1)[0]
            if appears and not _exists(root, before, entry):
                found.append(f"adds {entry}")
            if not appears and not _exists(root, after, entry):
                found.append(f"removes {entry}")
    return list(dict.fromkeys(found))


def follow_problems(root: Path, changes: list[Change], message: str, before: str | None, after: str | None) -> list[str]:
    """Returns the pages a change leaves behind and has not acknowledged.

    *before* and *after* are the revisions on each side of the change; None as *after* means the index.
    """
    waived, unchanged = acknowledged(message)
    if waived:
        return []
    touched = {change.path for change in changes} | {change.old_path for change in changes if change.old_path}
    changed_facts = _changed_facts(root, changes, before, after)
    problems: list[str] = []
    for page, (described, followed) in _pages(root, after).items():
        if page in touched or page in unchanged:
            continue
        reason = _reason(root, page, described, followed, changes, changed_facts, before, after)
        if reason:
            problems.append(f"{page} {reason}; update the page or add `Docs: {page} unchanged (<reason>)`")
    return problems


def _reason(root: Path, page: str, described: tuple[str, ...], followed: tuple[str, ...], changes: list[Change],
            changed_facts: set[str], before: str | None, after: str | None) -> str:
    """Returns why a change leaves a page behind, or an empty text when it does not."""
    for change in changes:
        hits = [entry for entry in described
                if _covers(entry, change.path) or (change.old_path and _covers(entry, change.old_path))]
        if hits:
            return f"describes {hits[0]}, which this commit changes"
    cited = sorted(set(followed) & changed_facts)
    if cited:
        return f"follows the fact {cited[0]}, which this commit changes"
    if PurePosixPath(page).name == K_README:
        for entry in _entry_changes(root, page, before, after, changes):
            return f"lists its folder, and this commit {entry}"
    return ""

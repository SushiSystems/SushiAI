# docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Finds the pages a change leaves behind, for the hook and the checker that share the rule.

A page names what it describes in a `**Describes:**` line; a README also describes its folder's
entries. The rules take values; `read_follow` and `follow_problems` are the shell that reads git.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from commits.history import blob_texts, git  # noqa: E402
from documentation import facts  # noqa: E402

K_SINCE: str = getattr(settings, "K_DOCS_FOLLOW_SINCE", "")
K_IGNORED: tuple[str, ...] = tuple(getattr(settings, "K_DOCS_FOLLOW_IGNORED", (
    "docs/agent", "docs/design", "docs/backlog", "docs/archive", "third_party")))
K_HEAD_LINES = 12
K_DESCRIBES = re.compile(r"^\*\*Describes:\*\*\s*(.+)$")
K_SPAN = re.compile(r"`([^`]+)`")
K_ACKNOWLEDGED = re.compile(
    r"^Docs:[ \t]*(.+?)[ \t]+unchanged[ \t]+\(\s*(\S.*?)\s*\)[ \t]*$", re.MULTILINE)
K_WAIVED = re.compile(r"^Docs:[ \t]*none[ \t]+\(\s*(\S.*?)\s*\)[ \t]*$", re.MULTILINE)
K_README = "README.md"
K_MARKDOWN = ".md"


@dataclass(frozen=True, slots=True)
class Change:
    """Holds one path a commit touches: its status letter, the path and, for a rename, the old one."""

    status: str
    path: str
    old_path: str = ""


@dataclass(frozen=True, slots=True)
class PageFacts:
    """Holds one followed page: its path, the paths it describes and the fact ids it follows."""

    name: str
    described: tuple[str, ...]
    follows: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Versions:
    """Holds the text of one changed markdown page on each side of a change; empty where absent."""

    before: str
    after: str


@dataclass(frozen=True, slots=True)
class Follow:
    """Holds what one commit presents to the rules: its changes, message, pages and trees."""

    changes: tuple[Change, ...]
    message: str
    ignored: tuple[str, ...]
    pages: tuple[PageFacts, ...]
    versions: tuple[Versions, ...]
    tree_before: frozenset[str]
    tree_after: frozenset[str]


def describes_of(text: str) -> tuple[str, ...]:
    """Returns the paths in the `**Describes:**` line among the first lines of a page."""
    for line in text.splitlines()[:K_HEAD_LINES]:
        found = K_DESCRIBES.match(line)
        if found:
            return tuple(item.strip("/") for item in K_SPAN.findall(found.group(1)))
    return ()


def page_facts(name: str, text: str) -> PageFacts:
    """Returns what a page describes and the ids of the facts it follows."""
    follows = tuple(dict.fromkeys(fact_id for fact_id, _ in facts.parse("", text).follows))
    return PageFacts(name, describes_of(text), follows)


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
    return (bool(K_WAIVED.search(message)),
            {match[0].strip("` ") for match in K_ACKNOWLEDGED.findall(message)})


def is_followed_page(name: str, ignored: tuple[str, ...]) -> bool:
    """Returns whether a path is a markdown page that no folder in *ignored* holds."""
    return name.endswith(K_MARKDOWN) and not any(_covers(entry, name) for entry in ignored)


def page_problems(follow: Follow) -> list[str]:
    """Returns the pages a commit leaves behind and has not acknowledged."""
    waived, unchanged = acknowledged(follow.message)
    if waived:
        return []
    touched = {change.path for change in follow.changes}
    touched |= {change.old_path for change in follow.changes if change.old_path}
    changed = _changed_facts(follow.versions)
    problems: list[str] = []
    for page in follow.pages:
        if page.name in touched or page.name in unchanged:
            continue
        reason = _reason(page, follow, changed)
        if reason:
            problems.append(f"{page.name} {reason}; update the page or add "
                            f"`Docs: {page.name} unchanged (<reason>)`")
    return problems


def _covers(entry: str, path: str) -> bool:
    """Returns whether a described file or folder is the path or holds it."""
    return path == entry or path.startswith(entry + "/")


def _present(tree: frozenset[str], entry: str) -> bool:
    """Returns whether a file or folder is in a tree of file paths."""
    return entry in tree or any(path.startswith(entry + "/") for path in tree)


def _moves(change: Change) -> list[tuple[str, bool]]:
    """Returns the paths a change makes appear (True) or disappear (False); an edit makes none."""
    if change.status == "A":
        return [(change.path, True)]
    if change.status == "D":
        return [(change.path, False)]
    if change.status == "R":
        return [(change.path, True), (change.old_path, False)]
    return []


def _changed_facts(versions: tuple[Versions, ...]) -> set[str]:
    """Returns the ids of the facts whose text, or whose existence, differs between the sides."""
    changed: set[str] = set()
    for version in versions:
        old = {block.id: block.body for block in facts.parse("", version.before).facts}
        new = {block.id: block.body for block in facts.parse("", version.after).facts}
        changed |= {fact_id for fact_id in old.keys() | new.keys()
                    if old.get(fact_id) != new.get(fact_id)}
    return changed


def _entry_changes(page: PageFacts, follow: Follow) -> list[str]:
    """Returns the entries of a README's folder that the changes add or remove whole."""
    folder = PurePosixPath(page.name).parent.as_posix()
    prefix = "" if folder == "." else folder + "/"
    found: list[str] = []
    for change in follow.changes:
        for path, appears in _moves(change):
            if not path.startswith(prefix) or path == page.name:
                continue
            if any(_covers(entry, path) for entry in follow.ignored):
                continue
            entry = prefix + path[len(prefix):].split("/", 1)[0]
            if appears and not _present(follow.tree_before, entry):
                found.append(f"adds {entry}")
            if not appears and not _present(follow.tree_after, entry):
                found.append(f"removes {entry}")
    return list(dict.fromkeys(found))


def _reason(page: PageFacts, follow: Follow, changed: set[str]) -> str:
    """Returns why a commit leaves a page behind, or an empty text when it does not."""
    for change in follow.changes:
        hits = [entry for entry in page.described
                if _covers(entry, change.path)
                or (change.old_path and _covers(entry, change.old_path))]
        if hits:
            return f"describes {hits[0]}, which this commit changes"
    cited = sorted(set(page.follows) & changed)
    if cited:
        return f"follows the fact {cited[0]}, which this commit changes"
    if PurePosixPath(page.name).name == K_README:
        for entry in _entry_changes(page, follow):
            return f"lists its folder, and this commit {entry}"
    return ""


K_PAGE_CACHE: dict[tuple[Path, str], PageFacts] = {}


def read_follow(root: Path, changes: list[Change], message: str,
                before: str | None, after: str | None) -> Follow:
    """Reads what a commit presents to the rules from git; None as *after* means the index.

    *before* and *after* are the revisions on each side of the change.
    """
    after_entries = _listing(root, after)
    followed = [(blob, name) for blob, name in after_entries
                if is_followed_page(name, K_IGNORED)]
    return Follow(
        changes=tuple(changes),
        message=message,
        ignored=K_IGNORED,
        pages=_pages(root, followed),
        versions=tuple(
            Versions(_text(root, before, change.old_path or change.path),
                     _text(root, after, change.path))
            for change in changes if change.path.endswith(K_MARKDOWN)
        ),
        tree_before=frozenset(name for _, name in _listing(root, before)),
        tree_after=frozenset(name for _, name in after_entries),
    )


def follow_problems(root: Path, changes: list[Change], message: str,
                    before: str | None, after: str | None) -> list[str]:
    """Returns the pages a change leaves behind and has not acknowledged, read from git.

    *before* and *after* are the revisions on each side of the change; None as *after* means
    the index.
    """
    return page_problems(read_follow(root, changes, message, before, after))


def _listing(root: Path, revision: str | None) -> list[tuple[str, str]]:
    """Returns the object hash and path of each file of a revision, or of the index for None."""
    listing = git(root, "ls-tree", "-r", revision) if revision else git(root, "ls-files", "-s")
    entries = []
    for line in (listing or "").splitlines():
        info, _, name = line.partition("\t")
        fields = info.split()
        if name and len(fields) >= 3:
            entries.append((fields[2] if revision else fields[1], name))
    return entries


def _pages(root: Path, entries: list[tuple[str, str]]) -> tuple[PageFacts, ...]:
    """Returns what each given page describes and follows, reading each version once.

    The pages are cached by object hash, and the ones not yet seen are read in one git call.
    """
    missing = sorted({blob for blob, _ in entries if (root, blob) not in K_PAGE_CACHE})
    texts = blob_texts(root, missing)
    for blob in missing:
        K_PAGE_CACHE[(root, blob)] = page_facts("", texts.get(blob, ""))
    return tuple(replace(K_PAGE_CACHE[(root, blob)], name=name) for blob, name in entries)


def _text(root: Path, revision: str | None, path: str) -> str:
    """Returns a file's text in a revision, or in the index for None; empty when absent."""
    return git(root, "show", f"{revision or ''}:{path}") or ""

# check_docs_layout.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks a repository's docs/ tree and module READMEs against the documentation skill.

Usage: python tools/documentation/check_docs_layout.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import datetime
import posixpath
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
from checking.source_file import read_files  # noqa: E402
from documentation.design_status import (  # noqa: E402
    K_DESIGN_INDEXES,
    K_WORK_FILES_IN_ORDER,
    Tree,
    children,
    design_documents,
    parse_status,
    status_documents,
)

K_DOCS_ENTRIES = frozenset(
    {
        "README.md", "CONTRIBUTING.md", "DOCUMENTATION_STYLE_GUIDE.md", "publish.toml",
        ".docs_truth_allow", "getting_started", "architecture", "guides", "reference",
        "design", "agent", "archive", "backlog",
    }
)
K_REQUIRED_ENTRIES = (
    "README.md", "CONTRIBUTING.md", "DOCUMENTATION_STYLE_GUIDE.md",
    "reference/CHANGELOG.md", "reference/GLOSSARY.md", "design/README.md",
    "backlog/README.md", "backlog/backlog.toml",
)
K_DETAIL_PREFIX = "docs/backlog/detail/"
K_DETAIL_NAME = re.compile(r"^(?:RW|KI)-[1-9]\d*\.md$")
K_WORK_FILES = frozenset(K_WORK_FILES_IN_ORDER)
K_WORK_FOLDER = re.compile(r"^\d{4}_\d{2}_\d{2}_[A-Z0-9]+(?:_[A-Z0-9]+)*$")
K_DOCUMENT_NAME = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*\.md$")
K_FORBIDDEN_KINDS = re.compile(r"ROADMAP|JOURNAL|HANDOVER|MILESTONE|PRIORITY|TODO")
K_DESIGN_LINE_CEILING = 400
K_CLOSED_GRACE_DAYS = 90
K_LINK = re.compile(r'(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
K_LINK_SUFFIX = re.compile(r"[#?]")
K_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:")
K_FENCE = re.compile(r"^(`{3,}|~{3,})(.*)$")
K_CODE_SPAN = re.compile(r"`[^`]*`")
K_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
K_HEADING_LABEL = re.compile(r"\{#([^}]+)\}")
K_ANCHOR_DROPPED = re.compile(r"[^\w\- ]")


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the values one repository's docs/ tree is checked from.

    The shell reads them from disk and the rules take them as given. Names are root-relative
    POSIX paths. ``present`` holds the link targets found on disk, and ``texts`` the text of each
    Markdown file the rules read.
    """

    root: Path
    today: datetime.date
    folders: frozenset[str]
    files: frozenset[str]
    texts: Mapping[str, str]
    present: frozenset[str]
    docs_entries: frozenset[str]
    closed_grace_days: int = K_CLOSED_GRACE_DAYS

    def path_of(self, relative: str) -> Path:
        """Returns the path below the root that a root-relative name stands for."""
        return self.root / relative

    @property
    def tree(self) -> Tree:
        """Returns the listing and the texts of the repository as the design functions read them."""
        return Tree(self.folders, self.files, self.texts)


def _children(repository: Repository, folder: str) -> list[str]:
    """Returns the root-relative name of each folder and file directly inside a folder, sorted."""
    return children(repository.tree, folder)


def _documents(repository: Repository) -> list[str]:
    """Returns each Markdown file under docs/ outside the archive, sorted."""
    return sorted(
        name for name in repository.texts
        if name.startswith("docs/") and not name.startswith("docs/archive/")
    )


def _manual(repository: Repository) -> list[str]:
    """Returns each live document outside the agent work folders and the backlog's detail files."""
    return [
        name for name in _documents(repository)
        if not name.startswith(("docs/agent/", K_DETAIL_PREFIX))
    ]


def _key(relative: str, target: str) -> str:
    """Returns the root-relative path a link target reaches from the document it sits in."""
    joined = posixpath.join(posixpath.dirname(relative), target.replace("\\", "/"))
    return posixpath.normpath(joined)


def _prose(text: str) -> str:
    """Returns a document's text with fenced blocks and code spans blanked, lines kept."""
    kept: list[str] = []
    fence = ""
    for line in text.removeprefix("﻿").splitlines():
        opener = K_FENCE.match(line.lstrip())
        if fence:
            fence = "" if opener and opener.group(1).startswith(fence) else fence
            kept.append("")
        elif opener and opener.group(1)[0] not in opener.group(2):
            fence = opener.group(1)
            kept.append("")
        else:
            kept.append(K_CODE_SPAN.sub("", line))
    return "\n".join(kept)


def _relative_links(text: str) -> Iterator[tuple[int, str]]:
    """Yields the line number and the path part of each relative link outside code."""
    prose = _prose(text)
    for match in K_LINK.finditer(prose):
        if K_SCHEME.match(match.group(1)):
            continue
        target = unquote(K_LINK_SUFFIX.split(match.group(1), maxsplit=1)[0])
        if target:
            yield prose.count("\n", 0, match.start()) + 1, target


def _anchored_links(text: str) -> Iterator[tuple[int, str, str]]:
    """Yields the line, the path part and the anchor of each relative link that has one."""
    prose = _prose(text)
    for match in K_LINK.finditer(prose):
        target, mark, anchor = match.group(1).partition("#")
        if mark and anchor and not K_SCHEME.match(target):
            yield prose.count("\n", 0, match.start()) + 1, unquote(target), unquote(anchor)


def heading_anchors(text: str) -> set[str]:
    """Returns the anchors of a document: each heading as GitHub slugs it, and each {#label}."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fence = ""
    for line in text.removeprefix("﻿").splitlines():
        opener = K_FENCE.match(line.lstrip())
        if fence:
            fence = "" if opener and opener.group(1).startswith(fence) else fence
            continue
        if opener and opener.group(1)[0] not in opener.group(2):
            fence = opener.group(1)
            continue
        heading = K_HEADING.match(line)
        if heading is None:
            continue
        title = heading.group(1).strip()
        label = K_HEADING_LABEL.search(title)
        if label:
            anchors.add(label.group(1))
            title = title[:label.start()].strip()
        slug = K_ANCHOR_DROPPED.sub("", title.replace("`", "")).strip().lower().replace(" ", "-")
        if not slug:
            continue
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def link_probes(documents: Mapping[str, str]) -> tuple[frozenset[str], frozenset[str]]:
    """Returns the paths the live documents' links reach, and the Markdown files among them.

    The shell probes the first set on disk, and reads the second for the headings a link names.
    """
    reached: set[str] = set()
    markdown: set[str] = set()
    for name, text in documents.items():
        if not name.startswith("docs/") or name.startswith("docs/archive/"):
            continue
        reached.update(_key(name, target) for _, target in _relative_links(text))
        for _, target, _anchor in _anchored_links(text):
            key = _key(name, target) if target else name
            if PurePosixPath(key).suffix == ".md":
                markdown.add(key)
    return frozenset(reached), frozenset(markdown)


def _reachable(repository: Repository) -> set[str]:
    """Returns the manual documents docs/README.md reaches without leaving the manual."""
    manual = set(_manual(repository))
    reached: set[str] = set()
    pending = ["docs/README.md"]
    while pending:
        name = pending.pop()
        if name in reached or name not in manual:
            continue
        reached.add(name)
        pending += [_key(name, target) for _, target in _relative_links(repository.texts[name])]
    return reached


def rule_docs_entries(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each entry directly under docs/ that the tree does not name."""
    for name in sorted(repository.docs_entries):
        if name not in K_DOCS_ENTRIES:
            yield Issue(repository.path_of(f"docs/{name}"), 1, "not part of the docs/ tree")


def rule_document_names(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each live document not named in UPPER_SNAKE_CASE."""
    for name in _documents(repository):
        filename = PurePosixPath(name).name
        if name.startswith(K_DETAIL_PREFIX):
            if not K_DETAIL_NAME.match(filename):
                yield Issue(
                    repository.path_of(name), 1,
                    "a backlog detail file is named after its entry, RW-<n>.md or KI-<n>.md",
                )
        elif not K_DOCUMENT_NAME.match(filename):
            yield Issue(repository.path_of(name), 1, "document names are UPPER_SNAKE_CASE.md")


def rule_work_folders(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each malformed work folder or foreign file inside one."""
    for folder in _children(repository, "docs/agent"):
        if folder not in repository.folders or not K_WORK_FOLDER.match(PurePosixPath(folder).name):
            yield Issue(
                repository.path_of(folder), 1,
                "docs/agent/ holds only YYYY_MM_DD_WORK_NAME folders",
            )
            continue
        for entry in _children(repository, folder):
            if PurePosixPath(entry).name not in K_WORK_FILES:
                yield Issue(
                    repository.path_of(entry), 1,
                    "a work folder holds only SPEC.md, PLAN.md and REPORT.md",
                )


def rule_status_lines(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each design document or work folder with a status in no known form."""
    for name in design_documents(repository.tree):
        if name not in repository.files:
            yield Issue(repository.path_of(name), 1, "a split design topic needs its README.md")
    for name in status_documents(repository.tree):
        status = parse_status(repository.texts[name])
        if not status.is_known:
            yield Issue(
                repository.path_of(name), status.line,
                "status must be '**Status:** Open — ...', 'Shipped — <date>', "
                "'Superseded — <date> by ...' or 'Withdrawn — <date>, ...'",
            )


def rule_design_ceiling(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each design document longer than the ceiling."""
    for name in sorted(name for name in repository.texts if name.startswith("docs/design/")):
        count = len(repository.texts[name].splitlines())
        if PurePosixPath(name).name not in K_DESIGN_INDEXES and count > K_DESIGN_LINE_CEILING:
            yield Issue(
                repository.path_of(name), 1,
                f"{count} lines; split the topic into phase files",
            )


def rule_links(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each relative link in a live document that does not resolve."""
    for name in _documents(repository):
        for number, target in _relative_links(repository.texts[name]):
            if _key(name, target) not in repository.present:
                yield Issue(
                    repository.path_of(name), number,
                    f"link target does not exist: {target}",
                )


def rule_link_anchors(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each link whose anchor names no heading of the document it reaches."""
    known: dict[str, set[str]] = {}
    for name in _documents(repository):
        for number, target, anchor in _anchored_links(repository.texts[name]):
            reached = _key(name, target) if target else name
            text = repository.texts.get(reached)
            if PurePosixPath(reached).suffix != ".md" or text is None:
                continue
            if reached not in known:
                known[reached] = heading_anchors(text)
            if anchor not in known[reached]:
                yield Issue(
                    repository.path_of(name), number,
                    f"no heading for the anchor: {target}#{anchor}",
                )


def rule_required_entries(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each required document missing from docs/."""
    for name in K_REQUIRED_ENTRIES:
        if f"docs/{name}" not in repository.files:
            yield Issue(repository.path_of(f"docs/{name}"), 1, "required document is missing")


def rule_reachable_from_index(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each manual document that docs/README.md does not reach."""
    if "docs/README.md" not in repository.files:
        return
    reached = _reachable(repository)
    for name in _manual(repository):
        if name not in reached:
            yield Issue(repository.path_of(name), 1, "not reachable from docs/README.md")


def rule_module_readmes(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each module folder without a README.md."""
    if "modules" not in repository.folders:
        return
    for tier in _children(repository, "modules"):
        if tier not in repository.folders:
            continue
        for module in _children(repository, tier):
            if module in repository.folders and f"{module}/README.md" not in repository.files:
                yield Issue(repository.path_of(module), 1, "module lacks README.md")


def rule_closed_work(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each closed work folder or design that is past the grace.

    A shipped design stays: it is the record of its decisions and graduates in size, not in place.
    """
    today = repository.today
    grace = repository.closed_grace_days
    for name in status_documents(repository.tree):
        status = parse_status(repository.texts[name])
        if status.closed_on is None or (today - status.closed_on).days < grace:
            continue
        path = PurePosixPath(name)
        work = path.parent.parent == PurePosixPath("docs/agent")
        if status.is_shipped and not work:
            continue
        candidate = path.parent if work or path.name == "README.md" else path
        days = (today - status.closed_on).days
        yield Issue(
            repository.path_of(candidate.as_posix()), 1,
            f"closed {days} days ago; delete it, the version history keeps it",
        )


def rule_forbidden_kinds(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each roadmap, journal, handover, milestone, priority or to-do file."""
    for name in _manual(repository):
        if K_FORBIDDEN_KINDS.search(PurePosixPath(name).stem):
            yield Issue(
                repository.path_of(name), 1,
                "this kind of file repeats the backlog or the changelog; see AGENT_WORK.md",
            )


def _relative(root: Path, path: Path) -> str:
    """Returns a path below the root as a root-relative POSIX name."""
    return path.relative_to(root).as_posix()


def _listing(root: Path) -> tuple[set[str], set[str]]:
    """Returns the folders and files under docs/, and the module folders and READMEs."""
    folders = {"docs"}
    files: set[str] = set()
    for path in (root / "docs").rglob("*"):
        if path.is_dir():
            folders.add(_relative(root, path))
        elif path.is_file():
            files.add(_relative(root, path))
    modules = root / "modules"
    if modules.is_dir():
        folders.add("modules")
        for tier in sorted(entry for entry in modules.iterdir() if entry.is_dir()):
            folders.add(_relative(root, tier))
            for module in sorted(entry for entry in tier.iterdir() if entry.is_dir()):
                folders.add(_relative(root, module))
                if (module / "README.md").is_file():
                    files.add(_relative(root, module / "README.md"))
    return folders, files


def _read_texts(root: Path, names: Iterable[str]) -> dict[str, str]:
    """Returns the text of each named file, keyed by the name it was given."""
    named = list(names)
    files = read_files([root / name for name in named], lambda _: True)
    return {name: file.text for name, file in zip(named, files, strict=True)}


def _read_repository(root: Path, today: datetime.date, grace: int) -> Repository:
    """Reads one repository's files into values; the only reader of the disk in this checker."""
    folders, files = _listing(root)
    markdown = sorted(
        _relative(root, path) for path in (root / "docs").rglob("*.md") if path.is_file()
    )
    texts = _read_texts(root, markdown)
    reached, linked = link_probes(texts)
    present = frozenset(name for name in reached if (root / name).exists())
    texts.update(_read_texts(root, sorted(
        name for name in linked if name not in texts and (root / name).is_file()
    )))
    entries = drop_ignored(root, sorted((root / "docs").iterdir()))
    return Repository(
        root=root,
        today=today,
        folders=frozenset(folders),
        files=frozenset(files),
        texts=texts,
        present=present,
        docs_entries=frozenset(entry.name for entry in entries),
        closed_grace_days=grace,
    )


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree, read into values with today's date."""
    grace = int(getattr(settings, "K_CLOSED_GRACE_DAYS", K_CLOSED_GRACE_DAYS))
    today = datetime.date.today()
    return [_read_repository(path, today, grace) for path in paths if (path / "docs").is_dir()]


K_CHECKER = Checker(
    description="Checks the docs/ tree against the documentation skill.",
    collect=collect,
    rules={
        "rule_docs_entries": rule_docs_entries,
        "rule_required_entries": rule_required_entries,
        "rule_document_names": rule_document_names,
        "rule_work_folders": rule_work_folders,
        "rule_status_lines": rule_status_lines,
        "rule_design_ceiling": rule_design_ceiling,
        "rule_links": rule_links,
        "rule_link_anchors": rule_link_anchors,
        "rule_reachable_from_index": rule_reachable_from_index,
        "rule_module_readmes": rule_module_readmes,
        "rule_closed_work": rule_closed_work,
        "rule_forbidden_kinds": rule_forbidden_kinds,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

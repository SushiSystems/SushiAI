# check_docs_layout.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks a repository's docs/ tree and module READMEs against the documentation skill.

Usage: python tools/documentation/check_docs_layout.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_DOCS_ENTRIES = frozenset(
    {
        "README.md", "CONTRIBUTING.md", "DOCUMENTATION_STYLE_GUIDE.md", "publish.toml",
        ".docs_truth_allow", "getting_started", "architecture", "guides", "reference", "design", "agent", "archive",
    }
)
K_DESIGN_INDEXES = frozenset({"README.md", "REMAINING_WORK.md"})
K_REQUIRED_ENTRIES = (
    "README.md", "CONTRIBUTING.md", "DOCUMENTATION_STYLE_GUIDE.md",
    "reference/CHANGELOG.md", "reference/GLOSSARY.md", "reference/KNOWN_ISSUES.md",
    "design/README.md", "design/REMAINING_WORK.md",
)
K_WORK_FILES = frozenset({"SPEC.md", "PLAN.md", "REPORT.md"})
K_WORK_FOLDER = re.compile(r"^\d{4}_\d{2}_\d{2}_[A-Z0-9]+(?:_[A-Z0-9]+)*$")
K_DOCUMENT_NAME = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*\.md$")
K_STATUS_LINE = re.compile(r"^\*\*Status:\*\* \S")
K_SHIPPED = re.compile(r"^\*\*Status:\*\* Shipped")
K_DESIGN_LINE_CEILING = 1500
K_ARCHIVE_AFTER_DAYS = 90
K_SECONDS_PER_DAY = 86_400
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
    """Holds the root of one repository whose documentation is checked."""

    root: Path

    @property
    def docs(self) -> Path:
        """Returns the repository's docs folder."""
        return self.root / "docs"

    def live_documents(self) -> list[Path]:
        """Returns every Markdown file under docs/ outside the archive."""
        archive = self.docs / "archive"
        return [
            path for path in sorted(self.docs.rglob("*.md"))
            if archive not in path.parents
        ]

    def manual_documents(self) -> list[Path]:
        """Returns every live document outside the agent work folders."""
        agent = self.docs / "agent"
        return [path for path in self.live_documents() if agent not in path.parents]

    def design_documents(self) -> list[Path]:
        """Returns every design document, counting a split topic's README as its document."""
        design = self.docs / "design"
        if not design.is_dir():
            return []
        documents = [path for path in sorted(design.glob("*.md")) if path.name not in K_DESIGN_INDEXES]
        documents += [folder / "README.md" for folder in sorted(design.iterdir()) if folder.is_dir()]
        return documents

    def work_folders(self) -> list[Path]:
        """Returns every entry directly under docs/agent/."""
        agent = self.docs / "agent"
        return sorted(agent.iterdir()) if agent.is_dir() else []


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree."""
    return [Repository(path) for path in paths if (path / "docs").is_dir()]


def _first_content_line(path: Path) -> tuple[int, str]:
    """Returns the number and text of the first non-empty line after the title."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines[1:], start=2):
        if line.strip():
            return index, line
    return 1, ""


def _last_commit_age_days(repository: Repository, path: Path) -> float | None:
    """Returns how many days ago a path last changed in git, or None when git cannot say."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository.root), "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stamp = completed.stdout.strip()
    return (time.time() - int(stamp)) / K_SECONDS_PER_DAY if stamp else None


def _prose(path: Path) -> str:
    """Returns a document's text with fenced blocks and code spans blanked, lines kept."""
    kept: list[str] = []
    fence = ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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


def _relative_links(path: Path) -> Iterator[tuple[int, str]]:
    """Yields the line number and the path part of each relative link outside code."""
    text = _prose(path)
    for match in K_LINK.finditer(text):
        if K_SCHEME.match(match.group(1)):
            continue
        target = unquote(K_LINK_SUFFIX.split(match.group(1), maxsplit=1)[0])
        if target:
            yield text.count("\n", 0, match.start()) + 1, target


def _anchored_links(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yields the line, the path part and the anchor of each relative link that has one."""
    text = _prose(path)
    for match in K_LINK.finditer(text):
        target, mark, anchor = match.group(1).partition("#")
        if mark and anchor and not K_SCHEME.match(target):
            yield text.count("\n", 0, match.start()) + 1, unquote(target), unquote(anchor)


def heading_anchors(path: Path) -> set[str]:
    """Returns the anchors of a document: each heading as GitHub slugs it, and each {#label}."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fence = ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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


def _reachable(repository: Repository) -> set[Path]:
    """Returns the manual documents docs/README.md reaches without leaving the manual."""
    manual = {path.resolve() for path in repository.manual_documents()}
    reached: set[Path] = set()
    pending = [(repository.docs / "README.md").resolve()]
    while pending:
        path = pending.pop()
        if path in reached or path not in manual:
            continue
        reached.add(path)
        pending += [(path.parent / target).resolve() for _, target in _relative_links(path)]
    return reached


def rule_docs_entries(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each entry directly under docs/ that the tree does not name."""
    for entry in sorted(repository.docs.iterdir()):
        if entry.name not in K_DOCS_ENTRIES:
            yield Issue(entry, 1, "not part of the docs/ tree")


def rule_document_names(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each live document not named in UPPER_SNAKE_CASE."""
    for path in repository.live_documents():
        if not K_DOCUMENT_NAME.match(path.name):
            yield Issue(path, 1, "document names are UPPER_SNAKE_CASE.md")


def rule_work_folders(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each malformed work folder or foreign file inside one."""
    for folder in repository.work_folders():
        if not folder.is_dir() or not K_WORK_FOLDER.match(folder.name):
            yield Issue(folder, 1, "docs/agent/ holds only YYYY_MM_DD_WORK_NAME folders")
            continue
        for entry in sorted(folder.iterdir()):
            if entry.name not in K_WORK_FILES:
                yield Issue(entry, 1, "a work folder holds only SPEC.md, PLAN.md and REPORT.md")


def rule_status_lines(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each design document without a status line under its title."""
    for path in repository.design_documents():
        if not path.is_file():
            yield Issue(path, 1, "a split design topic needs its README.md")
            continue
        number, line = _first_content_line(path)
        if not K_STATUS_LINE.match(line):
            yield Issue(path, number, "design document must open with '**Status:** ...'")


def rule_design_ceiling(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each design document longer than the ceiling."""
    design = repository.docs / "design"
    for path in sorted(design.rglob("*.md")) if design.is_dir() else []:
        count = len(path.read_text(encoding="utf-8").splitlines())
        if path.name not in K_DESIGN_INDEXES and count > K_DESIGN_LINE_CEILING:
            yield Issue(path, 1, f"{count} lines; split the topic into phase files")


def rule_links(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each relative link in a live document that does not resolve."""
    for path in repository.live_documents():
        for number, target in _relative_links(path):
            if not (path.parent / target).exists():
                yield Issue(path, number, f"link target does not exist: {target}")


def rule_link_anchors(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each link whose anchor names no heading of the document it reaches."""
    known: dict[Path, set[str]] = {}
    for path in repository.live_documents():
        for number, target, anchor in _anchored_links(path):
            reached = (path.parent / target).resolve() if target else path.resolve()
            if reached.suffix != ".md" or not reached.is_file():
                continue
            if reached not in known:
                known[reached] = heading_anchors(reached)
            if anchor not in known[reached]:
                yield Issue(path, number, f"no heading for the anchor: {target}#{anchor}")


def rule_required_entries(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each required document missing from docs/."""
    for name in K_REQUIRED_ENTRIES:
        if not (repository.docs / name).is_file():
            yield Issue(repository.docs / name, 1, "required document is missing")


def rule_reachable_from_index(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each manual document that docs/README.md does not reach."""
    if not (repository.docs / "README.md").is_file():
        return
    reached = _reachable(repository)
    for path in repository.manual_documents():
        if path.resolve() not in reached:
            yield Issue(path, 1, "not reachable from docs/README.md")


def rule_module_readmes(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each module folder without a README.md."""
    modules = repository.root / "modules"
    if not modules.is_dir():
        return
    for tier in sorted(entry for entry in modules.iterdir() if entry.is_dir()):
        for module in sorted(entry for entry in tier.iterdir() if entry.is_dir()):
            if not (module / "README.md").is_file():
                yield Issue(module, 1, "module lacks README.md")


def rule_archive_candidates(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each work folder or shipped design document idle past the limit."""
    candidates = [folder for folder in repository.work_folders() if folder.is_dir()]
    for path in repository.design_documents():
        if path.is_file() and K_SHIPPED.match(_first_content_line(path)[1]):
            candidates.append(path.parent if path.name == "README.md" else path)
    for candidate in candidates:
        age = _last_commit_age_days(repository, candidate)
        if age is not None and age > K_ARCHIVE_AFTER_DAYS:
            yield Issue(candidate, 1, f"idle {int(age)} days; move to docs/archive/")


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
        "rule_archive_candidates": rule_archive_candidates,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

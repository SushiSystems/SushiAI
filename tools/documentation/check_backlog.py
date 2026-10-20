# check_backlog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks REMAINING_WORK.md and KNOWN_ISSUES.md against the backlog format of the documentation skill.

Usage: python tools/documentation/check_backlog.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import K_EXIT_CLEAN, Checker, Issue, run  # noqa: E402
from documentation.design_status import design_documents, read_status  # noqa: E402

K_BACKLOG = Path("docs/design/REMAINING_WORK.md")
K_ISSUES = Path("docs/reference/KNOWN_ISSUES.md")
K_NEXT = "Next"
K_DECISIONS = "Decisions waiting for the owner"
K_NEXT_CEILING = 10
K_ENTRY_CEILING = 25
K_WHERE_CEILING = 5
K_STATES = frozenset({"decision", "open", "blocked", "unmeasured"})
K_ORIGINS = frozenset({"own", "toolchain", "dependency"})
K_COUNTER = re.compile(r"^Next id: (RW|KI)-(\d+)$")
K_ENTRY_HEADING = re.compile(r"^### (RW|KI)-(\d+) \S")
K_NEXT_LINE = re.compile(r"^- (RW-\d+) \S")
K_FIELD = re.compile(r"^- ([A-Z][a-z]+(?: [a-z]+)?): (\S.*)$")
K_CONTINUATION = re.compile(r"^  \S")
K_FINISHED = re.compile(r"^\s*[-*] (?:Done|Completed|Landed|Fixed)[:.]")
K_DATED_SECTION = re.compile(r"\d{4}-\d{2}-\d{2}")
K_FINISHED_SECTION = re.compile(r"\b(?:done|completed|finished|confirmed|landed)\b", re.IGNORECASE)
K_REFERENCE = re.compile(r"^(?:(?P<repository>[a-z0-9][a-z0-9_.-]*) )?(?P<id>(?:RW|KI)-\d+)$")
K_EXTERNAL = re.compile(r"^external: \S")
K_FIX = re.compile(r"^(?:RW-\d+|none, \S.*)$")
K_NO_WORKAROUND = re.compile(r"^none\b", re.IGNORECASE)
K_BACKLOG_REFERENCE = re.compile(r"^RW-\d+$")
K_SIBLING_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
K_CODE_SPAN = re.compile(r"`([^`]+)`")
K_SOURCE = re.compile(r"^`[^`]+`(?: §\d+(?:\.\d+)*)?$")
K_FENCE = re.compile(r"^(`{3,}|~{3,})")
K_SIBLING_FIELDS = frozenset({"Origin", "Upstream", "Blocked by"})


@dataclass(frozen=True, slots=True)
class Field:
    """Holds one field of an entry, its continuation lines joined to its value."""

    name: str
    value: str
    line: int


@dataclass(frozen=True, slots=True)
class Entry:
    """Holds one entry: its ID, the section it sits in, its fields and its lines."""

    ident: str
    heading: str
    section: str
    line: int
    end: int
    fields: tuple[Field, ...]
    body: tuple[tuple[int, str], ...]

    def field(self, name: str) -> Field | None:
        """Returns the named field, or None when the entry lacks it."""
        return next((item for item in self.fields if item.name == name), None)

    def value(self, name: str) -> str:
        """Returns the named field's value, or an empty string."""
        found = self.field(name)
        return found.value if found else ""


@dataclass(frozen=True, slots=True)
class Kind:
    """Describes one of the two files: its place, title, prefix, fields and the fields each entry needs."""

    path: Path
    title: str
    prefix: str
    fields: frozenset[str]
    required: Callable[[Entry], Iterator[str]]
    has_next: bool
    has_decisions: bool


@dataclass(slots=True)
class Document:
    """Holds one parsed file and the shape faults found while reading it."""

    kind: Kind
    path: Path
    counter: tuple[str, int] | None = None
    sections: list[tuple[str, int]] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    next_ids: list[tuple[str, int]] = field(default_factory=list)
    faults: list[Issue] = field(default_factory=list)

    def ids(self) -> set[str]:
        """Returns the ID of every entry."""
        return {entry.ident for entry in self.entries}


def _backlog_needs(entry: Entry) -> Iterator[str]:
    """Yields what a backlog entry lacks or carries against its state."""
    state = entry.value("State")
    if not state:
        yield "State: is required"
    elif state not in K_STATES:
        yield f"State: must be one of {', '.join(sorted(K_STATES))}"
    if not entry.field("Done when"):
        yield "Done when: is required"
    if (state == "blocked") != bool(entry.field("Blocked by")):
        yield "Blocked by: goes with State: blocked and with nothing else"
    if (state == "decision") != (entry.section == K_DECISIONS):
        yield f"State: decision goes in '## {K_DECISIONS}' and nothing else does"


def _issue_needs(entry: Entry) -> Iterator[str]:
    """Yields what a known issue lacks or carries against its origin."""
    origin = entry.value("Origin")
    sibling = bool(origin) and origin not in K_ORIGINS
    if not origin:
        yield "Origin: is required"
    elif sibling and not K_SIBLING_NAME.match(origin):
        yield "Origin: must be own, toolchain, dependency or a sibling repository"
    for name in ("Recognise", "Workaround"):
        if not entry.field(name):
            yield f"{name}: is required"
    if sibling != bool(entry.field("Upstream")):
        yield "Upstream: goes with an Origin: that names a sibling repository, and with nothing else"
    if not sibling and not entry.field("Cause"):
        yield "Cause: is required unless the origin is a sibling"
    if origin == "own":
        for name in ("Where", "Fix"):
            if not entry.field(name):
                yield f"{name}: is required when the origin is own"
    if bool(entry.field("Owner")) != bool(entry.field("Fix by")):
        yield "Owner: and Fix by: go together"


K_BACKLOG_KIND = Kind(
    K_BACKLOG, "# Remaining work", "RW",
    frozenset({"State", "Done when", "Blocked by", "Where", "Source"}), _backlog_needs, True, True,
)
K_ISSUES_KIND = Kind(
    K_ISSUES, "# Known issues", "KI",
    frozenset({"Origin", "Upstream", "Where", "Cause", "Recognise", "Workaround", "Fix", "Owner", "Fix by", "Source"}),
    _issue_needs, False, False,
)


class _Reader:
    """Reads one file line by line into a Document."""

    def __init__(self, kind: Kind, path: Path) -> None:
        self.document = Document(kind, path)
        self.section: str | None = None
        self.heading: tuple[str, int, str] | None = None
        self.skipping = False
        self.lines: list[tuple[int, str]] = []

    def fault(self, line: int, message: str) -> None:
        """Records a shape fault."""
        self.document.faults.append(Issue(self.document.path, line, message))

    def close_entry(self) -> None:
        """Turns the lines gathered under the current heading into an entry."""
        if self.heading is None:
            return
        ident, line, heading = self.heading
        lines = list(self.lines)
        while lines and not lines[-1][1].strip():
            lines.pop()
        fields: list[Field] = []
        for number, text in lines:
            match = K_FIELD.match(text)
            if match:
                fields.append(Field(match[1], match[2], number))
            elif fields and K_CONTINUATION.match(text):
                last = fields[-1]
                fields[-1] = Field(last.name, f"{last.value} {text.strip()}", last.line)
            else:
                break
        for item in fields:
            if item.name not in self.document.kind.fields:
                self.fault(item.line, f"unknown field '{item.name}'")
        field_lines = {item.line for item in fields}
        for number, text in lines:
            if number not in field_lines and K_FINISHED.match(text):
                self.fault(number, "finished work is deleted; the changelog records it")
        end = lines[-1][0] if lines else line
        self.document.entries.append(
            Entry(ident, heading, self.section or "", line, end, tuple(fields), tuple(lines))
        )
        self.heading = None
        self.lines = []

    def read_heading(self, number: int, line: str) -> None:
        """Opens a section or an entry, or records a heading out of shape."""
        if line.startswith("## "):
            self.close_entry()
            self.skipping = False
            self.section = line[3:].strip()
            self.document.sections.append((self.section, number))
        elif line.startswith("### "):
            self.close_entry()
            match = K_ENTRY_HEADING.match(line)
            self.skipping = match is None
            if match is None:
                self.fault(number, "an entry heading reads '### <ID> <title>'")
            elif self.section is None:
                self.fault(number, "an entry sits in a section")
            else:
                self.heading = (f"{match[1]}-{match[2]}", number, line)
        elif number != 1:
            self.fault(number, "only the title, sections and entries are headings")

    def read_line(self, number: int, line: str) -> None:
        """Places one line that is not a heading."""
        counter = K_COUNTER.match(line)
        if counter:
            if self.section is not None or self.document.counter is not None:
                self.fault(number, "the counter is written once, before the first section")
            self.document.counter = (counter[1], int(counter[2]))
        elif self.section == K_NEXT and self.document.kind.has_next:
            match = K_NEXT_LINE.match(line)
            if match:
                self.document.next_ids.append((match[1], number))
            elif line.strip():
                self.fault(number, "## Next holds lines '- RW-<n> <why it comes here>'")
        elif self.heading is not None:
            self.lines.append((number, line))
        elif line.strip() and self.section is not None and not self.skipping:
            self.fault(number, "a section holds entries only")

    def read(self) -> Document:
        """Returns the parsed document."""
        lines = self.document.path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != self.document.kind.title:
            self.fault(1, f"the title is '{self.document.kind.title}'")
        in_fence = False
        for number, line in enumerate(lines, start=1):
            if K_FENCE.match(line):
                in_fence = not in_fence
            if not in_fence and line.startswith("#"):
                self.read_heading(number, line)
            else:
                self.read_line(number, line)
        self.close_entry()
        return self.document


@final
class Repository:
    """Holds one repository's two files, its siblings and what reading them found."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def name(self) -> str:
        """Returns the folder name, which is the name other repositories cite it by."""
        return self.root.name

    @cached_property
    def documents(self) -> dict[str, Document]:
        """Returns each file that exists, by its prefix."""
        found: dict[str, Document] = {}
        for kind in (K_BACKLOG_KIND, K_ISSUES_KIND):
            if (self.root / kind.path).is_file():
                found[kind.prefix] = _Reader(kind, self.root / kind.path).read()
        return found

    @cached_property
    def siblings(self) -> dict[str, Repository]:
        """Returns each folder beside this one that holds a backlog, by its name."""
        parent = self.root.parent
        return {
            folder.name: Repository(folder)
            for folder in sorted(parent.iterdir())
            if folder.is_dir() and folder != self.root and (folder / K_BACKLOG).is_file()
        }

    def entries(self) -> Iterator[tuple[Document, Entry]]:
        """Yields every entry of both files with the file it sits in."""
        for document in self.documents.values():
            for entry in document.entries:
                yield document, entry

    def ids(self, prefix: str) -> set[str]:
        """Returns the IDs of the file with the given prefix."""
        document = self.documents.get(prefix)
        return document.ids() if document else set()

    def resolves(self, reference: str) -> bool | None:
        """Returns whether a reference names an entry, or None for a sibling that is not checked out."""
        match = K_REFERENCE.match(reference)
        if match is None:
            return False
        ident = match["id"]
        owner = match["repository"]
        if owner is None or owner == self.name:
            return ident in self.ids(ident[:2])
        sibling = self.siblings.get(owner)
        return None if sibling is None else ident in sibling.ids(ident[:2])

    def references(self) -> Iterator[tuple[Path, int, str]]:
        """Yields each ID reference the two files make, with where it is written."""
        for document in self.documents.values():
            for ident, line in document.next_ids:
                yield document.path, line, ident
        for document, entry in self.entries():
            blocked = entry.field("Blocked by")
            if blocked and not K_EXTERNAL.match(blocked.value):
                for part in blocked.value.split(", "):
                    if not K_EXTERNAL.match(part):
                        yield document.path, blocked.line, part
            for name in ("Fix", "Upstream"):
                item = entry.field(name)
                if item and K_REFERENCE.match(item.value):
                    yield document.path, item.line, item.value

    def unchecked_references(self) -> int:
        """Returns how many references name a sibling that is not checked out."""
        return sum(1 for _, _, reference in self.references() if self.resolves(reference) is None)

    def sibling_mentions(self) -> Iterator[tuple[Path, int, str, str]]:
        """Yields each entry whose title or body names a sibling outside the fields that cite one."""
        patterns = {name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) for name in self.siblings}
        for document, entry in self.entries():
            cited = {item.line for item in entry.fields if item.name in K_SIBLING_FIELDS}
            text = [entry.heading] + [line for number, line in entry.body if number not in cited]
            for name, pattern in patterns.items():
                if any(pattern.search(line) for line in text):
                    yield document.path, entry.line, entry.ident, name


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder that holds a docs/ tree."""
    return [Repository(path) for path in paths if (path / "docs").is_dir()]


def rule_shape(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a title, counter, section, heading or field out of shape."""
    for document in repository.documents.values():
        kind = document.kind
        yield from document.faults
        if document.counter is None:
            yield Issue(document.path, 1, f"the counter line 'Next id: {kind.prefix}-<n>' is missing")
        elif document.counter[0] != kind.prefix:
            yield Issue(document.path, 1, f"the counter counts {kind.prefix} IDs")
        for index, (name, line) in enumerate(document.sections):
            if name == K_NEXT and (not kind.has_next or index != 0):
                yield Issue(document.path, line, "## Next is the backlog's first section")
            elif name == K_DECISIONS and not kind.has_decisions:
                yield Issue(document.path, line, "the decisions section is the backlog's")
            elif K_DATED_SECTION.search(name) or K_FINISHED_SECTION.search(name):
                yield Issue(document.path, line, "a section is an area: no date, no finished work")
        if len(document.next_ids) > K_NEXT_CEILING:
            yield Issue(document.path, document.next_ids[K_NEXT_CEILING][1], f"## Next holds at most {K_NEXT_CEILING} lines")
        for entry in document.entries:
            for message in kind.required(entry):
                yield Issue(document.path, entry.line, f"{entry.ident}: {message}")
            yield from _value_shapes(document.path, entry)


def _value_shapes(path: Path, entry: Entry) -> Iterator[Issue]:
    """Yields an issue for each field whose value is out of shape."""
    blocked = entry.field("Blocked by")
    if blocked:
        parts = blocked.value.split(", ")
        if not all(K_REFERENCE.match(part) or K_EXTERNAL.match(part) for part in parts) and not K_EXTERNAL.match(blocked.value):
            yield Issue(path, blocked.line, "Blocked by: IDs, or 'external: <reason>'")
    fix = entry.field("Fix")
    if fix and not K_FIX.match(fix.value):
        yield Issue(path, fix.line, "Fix: 'RW-<n>' or 'none, <reason>'")
    upstream = entry.field("Upstream")
    if upstream:
        match = K_REFERENCE.match(upstream.value)
        if match is None or match["repository"] is None or not match["id"].startswith("KI-"):
            yield Issue(path, upstream.line, "Upstream: '<repository> KI-<n>'")
    source = entry.field("Source")
    if source and not K_SOURCE.match(source.value):
        yield Issue(path, source.line, "Source: one path in backticks, then an optional '§<n>'")
    where = entry.field("Where")
    if where and not 1 <= len(K_CODE_SPAN.findall(where.value)) <= K_WHERE_CEILING:
        yield Issue(path, where.line, f"Where: one to {K_WHERE_CEILING} paths in backticks")


def rule_scope(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for an own known issue that has a scheduled fix and no workaround."""
    issues = repository.documents.get(K_ISSUES_KIND.prefix)
    for entry in issues.entries if issues else []:
        fix = entry.value("Fix")
        if entry.value("Origin") == "own" and K_BACKLOG_REFERENCE.match(fix) and K_NO_WORKAROUND.match(entry.value("Workaround")):
            yield Issue(issues.path, entry.line, f"{entry.ident}: a scheduled fix and no workaround is backlog work; move it to {fix}")


def rule_ids(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a duplicate ID, an ID past the counter or an ID of the other file."""
    for document in repository.documents.values():
        seen: set[str] = set()
        for entry in document.entries:
            prefix, number = entry.ident.split("-")
            if prefix != document.kind.prefix:
                yield Issue(document.path, entry.line, f"{entry.ident}: this file holds {document.kind.prefix} IDs")
            elif entry.ident in seen:
                yield Issue(document.path, entry.line, f"{entry.ident}: the ID is used twice")
            elif document.counter is not None and int(number) >= document.counter[1]:
                yield Issue(document.path, entry.line, f"{entry.ident}: at or past the counter; raise 'Next id:'")
            seen.add(entry.ident)


def rule_paths(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each path in Where or Source that does not exist from the root."""
    for document, entry in repository.entries():
        for name in ("Where", "Source"):
            item = entry.field(name)
            for cited in K_CODE_SPAN.findall(item.value) if item else []:
                if cited.startswith(("/", "..")) or not (repository.root / cited).exists():
                    yield Issue(document.path, item.line, f"{name}: '{cited}' does not exist from the repository root")


def rule_references(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each reference that names no entry, here or in a sibling checked out."""
    for path, line, reference in repository.references():
        if repository.resolves(reference) is False:
            yield Issue(path, line, f"'{reference}' names no entry")


def rule_entry_size(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each entry longer than the ceiling."""
    for document, entry in repository.entries():
        size = entry.end - entry.line + 1
        if size > K_ENTRY_CEILING:
            yield Issue(document.path, entry.line, f"{entry.ident}: {size} lines, ceiling {K_ENTRY_CEILING}; move the detail to its Source")


def rule_design_coverage(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each open design document that no backlog entry cites."""
    backlog = repository.documents.get(K_BACKLOG_KIND.prefix)
    if backlog is None:
        return
    cited = {
        span for entry in backlog.entries
        for span in K_CODE_SPAN.findall(entry.value("Source"))
    }
    for path in design_documents(repository.root / "docs"):
        if not path.is_file() or not read_status(path).is_open:
            continue
        relative = path.relative_to(repository.root).as_posix()
        topic = relative.rsplit("/", 1)[0] + "/" if path.name == "README.md" else None
        if relative not in cited and not (topic and any(span.startswith(topic) for span in cited)):
            yield Issue(path, 1, "an open design document is cited by a backlog entry's Source")


K_CHECKER = Checker(
    description="Checks the backlog and the known issues against the documentation skill.",
    collect=collect,
    rules={
        "rule_shape": rule_shape,
        "rule_ids": rule_ids,
        "rule_paths": rule_paths,
        "rule_references": rule_references,
        "rule_scope": rule_scope,
        "rule_entry_size": rule_entry_size,
        "rule_design_coverage": rule_design_coverage,
    },
)


def _branch(repository: Repository) -> str:
    """Returns the branch a sibling is checked out on, or '?' when git cannot say."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository.root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return completed.stdout.strip()


def main(argv: list[str]) -> int:
    """Runs the checker, and under --report lists the siblings read and the entries naming one."""
    repositories: list[Repository] = []

    def collect_once(paths: list[Path]) -> list[Repository]:
        """Returns the repositories of the given folders and keeps them for the closing lines."""
        repositories.extend(collect(paths))
        return repositories

    code = run(Checker(K_CHECKER.description, collect_once, K_CHECKER.rules), argv)
    if "--report" not in argv or code != K_EXIT_CLEAN:
        return code
    for repository in repositories:
        read = ", ".join(f"{name} ({_branch(sibling)})" for name, sibling in repository.siblings.items())
        print(f"siblings read: {read or 'none'}")
        print(f"sibling references not checked: {repository.unchecked_references()}")
        mentions = list(repository.sibling_mentions())
        print(f"entries naming a sibling: {len(mentions)}")
        for path, line, ident, name in mentions:
            print(f"  {path.relative_to(repository.root).as_posix()}:{line}: {ident} names {name}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

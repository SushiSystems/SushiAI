# check_contract.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks the throw sites of the library and of the applications against the class table.

A use of SA_THROW_IF or of another removed name anywhere in the library, the programs or the tests
is a finding. The table is tools/contract/site_classes.tsv.
Usage: python tools/contract/check_contract.py [repository root] [--report] [--rule NAME]
The design is docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md, sections 5 and 6.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_SITE_TABLE = Path("tools/contract/site_classes.tsv")
K_LIBRARY_FOLDERS = ("modules/*/*/include", "modules/*/*/source")
K_PROGRAM_FOLDERS = ("applications",)
K_SITE_FOLDERS = K_LIBRARY_FOLDERS + K_PROGRAM_FOLDERS
K_REMOVED_FOLDERS = K_SITE_FOLDERS + ("tests",)
K_SOURCE_SUFFIXES = (".hpp", ".cpp", ".h", ".inl")
K_COLUMNS = (
    "file", "line", "function", "kind", "message_head", "cause", "mechanism", "open_question",
    "reason",
)
K_KINDS = ("throw",)
K_CAUSES = ("user-content", "outside", "precondition", "invariant", "unclassified")
K_MECHANISMS = ("status", "fatal", "unclassified")
K_HEAD_LENGTH = 60
K_QUESTION_KEYS = re.compile(r"(?:[a-z0-9_]+(?:;[a-z0-9_]+)*)?")
K_THROW_SITE = re.compile(r"\bthrow\b")
K_REMOVED_NAMES = re.compile(
    r"\bSA_THROW_IF\b|\bSA_EXPECT_ERROR\b|\bbridge_throwing\b|\bSushiAI\s*::\s*Error\b"
    r"|failure/error\.hpp|nn/throwing_bridge\.hpp|failure_expectations/throw_expectations\.hpp"
)
K_LITERAL = re.compile(r'\s*"((?:[^"\\\n]|\\.)*)"')
K_OPENERS = "([{"
K_CLOSERS = ")]}"


@dataclass(frozen=True, slots=True)
class Site:
    """Holds one failure site found in a source file."""

    file: str
    line: int
    kind: str
    head: str


@dataclass(frozen=True, slots=True)
class Row:
    """Holds one data row of the class table."""

    number: int
    file: str
    line: int
    function: str
    kind: str
    head: str
    cause: str
    mechanism: str
    question: str
    reason: str


def _blank(text: str) -> str:
    """Returns the text with every character but a newline turned into a space."""
    return re.sub(r"[^\n]", " ", text)


def _literal_end(text: str, start: int) -> int:
    """Returns the index after the string or character literal that opens at start."""
    quote = text[start]
    index = start + 1
    while index < len(text) and text[index] not in (quote, "\n"):
        index += 2 if text[index] == "\\" else 1
    return min(index + 1, len(text))


def _raw_literal_end(text: str, start: int) -> int:
    """Returns the index after the raw string literal whose R sits at start, or -1."""
    paren = text.find("(", start + 2, start + 20)
    if paren < 0:
        return -1
    closing = ")" + text[start + 2:paren] + '"'
    end = text.find(closing, paren)
    return -1 if end < 0 else end + len(closing)


def mask_source(text: str) -> tuple[str, str]:
    """Returns the C++ text twice with comments blanked: literals kept, then blanked.

    Both results keep every offset and line of the input.
    """
    kept: list[str] = []
    code: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        pair = text[index:index + 2]
        after_word = index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_")
        end = index + 1
        if pair == "//":
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            kept.append(_blank(text[index:end]))
            code.append(_blank(text[index:end]))
        elif pair == "/*":
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            kept.append(_blank(text[index:end]))
            code.append(_blank(text[index:end]))
        elif pair == 'R"' and not after_word and _raw_literal_end(text, index) > 0:
            end = _raw_literal_end(text, index)
            kept.append(text[index:end])
            code.append(_blank(text[index:end]))
        elif character == '"' or (character == "'" and not after_word):
            end = _literal_end(text, index)
            kept.append(text[index:end])
            code.append(_blank(text[index:end]))
        else:
            kept.append(character)
            code.append(character)
        index = end
    return "".join(kept), "".join(code)


def message_head(text: str) -> str:
    """Returns the first characters of the adjacent string literals that open text.

    A text that opens with no literal gives its own first characters, whitespace collapsed.
    """
    parts: list[str] = []
    position = 0
    while match := K_LITERAL.match(text, position):
        parts.append(match.group(1).replace('\\"', '"').replace("\\\\", "\\"))
        position = match.end()
    joined = "".join(parts) if parts else " ".join(text.split())
    return joined[:K_HEAD_LENGTH]


def find_sites(file: str, text: str) -> list[Site]:
    """Returns every throw statement of one source text, by line."""
    kept, code = mask_source(text)
    sites: list[Site] = []
    for match in K_THROW_SITE.finditer(code):
        end = code.find(";", match.end())
        statement = kept[match.end():len(kept) if end < 0 else end]
        literal = statement.find('"')
        head = message_head(statement[literal:]) if literal >= 0 else ""
        sites.append(Site(file, code.count("\n", 0, match.start()) + 1, "throw", head))
    return sorted(sites, key=lambda site: site.line)


def find_removed_names(text: str) -> list[tuple[int, str]]:
    """Returns the line and the text of each use of a removed name outside comments and literals."""
    code = mask_source(text)[1]
    return [
        (code.count("\n", 0, match.start()) + 1, match.group(0))
        for match in K_REMOVED_NAMES.finditer(code)
    ]


def _is_source(name: str) -> bool:
    """Returns whether a relative path names a C++ source or header file."""
    return PurePosixPath(name).suffix in K_SOURCE_SUFFIXES


def _folder_of(name: str, patterns: tuple[str, ...]) -> bool:
    """Returns whether a relative path lies at any depth under a folder one pattern names."""
    parts = name.split("/")
    return any(
        len(parts) > len(pattern.split("/"))
        and fnmatch("/".join(parts[:len(pattern.split("/"))]), pattern)
        for pattern in patterns
    )


class SourceTree(Protocol):
    """Gives a repository its source texts and its class table without saying where they live."""

    def texts(self, patterns: tuple[str, ...]) -> dict[str, str]:
        """Returns the text of each source file under the folders, by relative path."""

    def table_text(self) -> str | None:
        """Returns the text of the class table, or None when the tree holds none."""


class DiskTree:
    """Reads the sources and the class table from a folder on disk."""

    def __init__(self, root: Path) -> None:
        """Remembers the root of the tree."""
        self.root = root

    def texts(self, patterns: tuple[str, ...]) -> dict[str, str]:
        """Returns the text of each source file under the folders, by relative path."""
        found: dict[str, str] = {}
        for pattern in patterns:
            for folder in sorted(self.root.glob(pattern)):
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix in K_SOURCE_SUFFIXES:
                        found[path.relative_to(self.root).as_posix()] = path.read_text(
                            encoding="utf-8"
                        )
        return found

    def table_text(self) -> str | None:
        """Returns the text of the class table, or None when the file is absent."""
        table = self.root / K_SITE_TABLE
        return table.read_text(encoding="utf-8") if table.is_file() else None


class MemoryTree:
    """Holds the files of a tree as a mapping of relative path to text."""

    def __init__(self, files: Mapping[str, str]) -> None:
        """Remembers the files, the class table among them under its own path."""
        self.files = dict(files)

    def texts(self, patterns: tuple[str, ...]) -> dict[str, str]:
        """Returns the text of each source file under the folders, by relative path."""
        return {
            name: text
            for name, text in sorted(self.files.items())
            if _is_source(name) and _folder_of(name, patterns)
        }

    def table_text(self) -> str | None:
        """Returns the text of the class table, or None when the tree holds none."""
        return self.files.get(K_SITE_TABLE.as_posix())


def removed_uses_of(sources: Mapping[str, str]) -> dict[str, list[tuple[int, str]]]:
    """Returns the uses of a removed name in each source text that has any."""
    found = {name: find_removed_names(text) for name, text in sources.items()}
    return {name: uses for name, uses in found.items() if uses}


def sites_of(sources: Mapping[str, str]) -> dict[str, list[Site]]:
    """Returns the failure sites of each source text that has any."""
    found = {file: find_sites(file, text) for file, text in sources.items()}
    return {file: sites for file, sites in found.items() if sites}


def rows_of(lines: list[str]) -> dict[str, list[Row]]:
    """Returns the well-formed rows of the class table lines, by file."""
    found: dict[str, list[Row]] = defaultdict(list)
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) == len(K_COLUMNS) and cells[1].isdigit():
            found[cells[0]].append(Row(number, cells[0], int(cells[1]), *cells[2:]))
    return dict(found)


def unmatched_of(
    sites: Mapping[str, list[Site]], rows: Mapping[str, list[Row]]
) -> tuple[list[Site], list[Row]]:
    """Returns the sites with no row and the rows with no site.

    A site and a row match on file, kind and message head; equal heads pair in line order.
    """
    lone_sites: list[Site] = []
    lone_rows: list[Row] = []
    for file in sorted(set(sites) | set(rows)):
        waiting: dict[tuple[str, str], list[Row]] = defaultdict(list)
        for row in sorted(rows.get(file, []), key=lambda row: row.line):
            waiting[(row.kind, row.head)].append(row)
        for site in sites.get(file, []):
            if waiting[(site.kind, site.head)]:
                waiting[(site.kind, site.head)].pop(0)
            else:
                lone_sites.append(site)
        lone_rows.extend(row for held in waiting.values() for row in held)
    return lone_sites, sorted(lone_rows, key=lambda row: row.number)


def table_problems(lines: list[str]) -> list[tuple[int, str]]:
    """Returns the line number and the message of each fault in the class table lines."""
    if not lines:
        return [(1, f"{K_SITE_TABLE.as_posix()} is missing or empty")]
    problems: list[tuple[int, str]] = []
    if tuple(lines[0].split("\t")) != K_COLUMNS:
        problems.append((1, "the header must read " + " ".join(K_COLUMNS)))
    previous = ("", 0)
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(K_COLUMNS) or not cells[1].isdigit():
            problems.append((number, f"a row has {len(K_COLUMNS)} cells and a numeric line"))
            continue
        if cells[3] not in K_KINDS or cells[5] not in K_CAUSES or cells[6] not in K_MECHANISMS:
            problems.append((number, "unknown kind, cause or mechanism"))
        if not K_QUESTION_KEYS.fullmatch(cells[7]):
            problems.append((number, "open_question holds snake_case keys joined by ;"))
        if (cells[0], int(cells[1])) < previous:
            problems.append((number, "rows are sorted by file, then by line"))
        previous = (cells[0], int(cells[1]))
    return problems


def unclassified_notes(rows: Mapping[str, list[Row]]) -> list[tuple[int, str]]:
    """Returns the table line and the note of each row whose cause or mechanism is unclassified."""
    return [
        (row.number, f"{row.file}:{row.line} has no class yet")
        for held in rows.values()
        for row in held
        if "unclassified" in (row.cause, row.mechanism)
    ]


def open_question_notes(rows: Mapping[str, list[Row]]) -> list[tuple[int, str]]:
    """Returns one note per open question, at its first row, with the number of rows it holds."""
    first: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for held in rows.values():
        for row in held:
            for key in filter(None, row.question.split(";")):
                first[key] = min(first.get(key, row.number), row.number)
                counts[key] += 1
    return [(first[key], f"{key}: {counts[key]} rows wait for an answer") for key in sorted(counts)]


class Repository:
    """Holds one repository's failure sites and the rows of its class table."""

    def __init__(self, root: Path, tree: SourceTree | None = None) -> None:
        """Remembers the root and the tree; the tree is read when a rule first asks."""
        self.root = root
        self.table = root / K_SITE_TABLE
        self.tree: SourceTree = DiskTree(root) if tree is None else tree

    @cached_property
    def sources(self) -> dict[str, str]:
        """Returns the text of every source file under the site folders, by relative path."""
        return self.tree.texts(K_SITE_FOLDERS)

    @cached_property
    def removed_uses(self) -> dict[str, list[tuple[int, str]]]:
        """Returns the uses of a removed name in each file of the library, programs and tests."""
        return removed_uses_of(self.tree.texts(K_REMOVED_FOLDERS))

    @cached_property
    def sites(self) -> dict[str, list[Site]]:
        """Returns the sites of each file that has any."""
        return sites_of(self.sources)

    @cached_property
    def table_lines(self) -> list[str]:
        """Returns the lines of the class table, empty when the table is absent."""
        return (self.tree.table_text() or "").splitlines()

    @cached_property
    def rows(self) -> dict[str, list[Row]]:
        """Returns the well-formed rows of the class table, by file."""
        return rows_of(self.table_lines)

    @cached_property
    def unmatched(self) -> tuple[list[Site], list[Row]]:
        """Returns the sites with no row and the rows with no site."""
        return unmatched_of(self.sites, self.rows)


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns one repository per path given."""
    return [Repository(path if path.is_dir() else path.parent) for path in paths]


def rule_table_shape(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a missing table, a wrong header, a malformed or misplaced row."""
    for line, message in table_problems(repository.table_lines):
        yield Issue(repository.table, line, message)


def rule_removed_surface(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each use of SA_THROW_IF, SushiAI::Error or another removed name."""
    for file, uses in repository.removed_uses.items():
        for line, name in uses:
            message = f"'{name}' belongs to the removed throwing surface; use the failure contract"
            yield Issue(repository.root / file, line, message)


def rule_unlisted_site(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each throw site that no row describes."""
    for site in repository.unmatched[0]:
        yield Issue(repository.root / site.file, site.line, f"no row for '{site.head}'")


def rule_stale_row(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each row that matches no site of its file."""
    for row in repository.unmatched[1]:
        yield Issue(repository.table, row.number, f"no site in {row.file} for '{row.head}'")


def note_unclassified(repository: Repository) -> Iterator[Issue]:
    """Yields a note for each row whose cause or mechanism is unclassified."""
    for line, message in unclassified_notes(repository.rows):
        yield Issue(repository.table, line, message)


def note_open_question(repository: Repository) -> Iterator[Issue]:
    """Yields one note per open question, at its first row, with the number of rows it holds."""
    for line, message in open_question_notes(repository.rows):
        yield Issue(repository.table, line, message)


K_CHECKER = Checker(
    description="Checks the throw sites against the class table and bans the removed throwing surface.",
    collect=collect,
    rules={
        "rule_table_shape": rule_table_shape,
        "rule_removed_surface": rule_removed_surface,
        "rule_unlisted_site": rule_unlisted_site,
        "rule_stale_row": rule_stale_row,
    },
)

K_NOTES = {
    "note_unclassified": note_unclassified,
    "note_open_question": note_open_question,
}


def _roots(argv: list[str]) -> list[Path]:
    """Returns the paths on a command line the runner will parse again."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rule")
    return parser.parse_known_args(argv)[0].paths or [Path.cwd()]


def render_notes(repository: Repository) -> list[str]:
    """Returns the printed line of each note a repository raises."""
    lines: list[str] = []
    for name, note in K_NOTES.items():
        for issue in note(repository):
            try:
                shown = issue.path.resolve().relative_to(repository.root.resolve()).as_posix()
            except ValueError:
                shown = issue.path.as_posix()
            lines.append(f"note: {shown}:{issue.line}: {name}: {issue.message}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Prints the notes, which never fail, then runs the rules and returns their exit code."""
    arguments = sys.argv[1:] if argv is None else argv
    for repository in collect([path for path in _roots(arguments) if path.exists()]):
        for line in render_notes(repository):
            print(line)
    return run(K_CHECKER, arguments)


if __name__ == "__main__":
    sys.exit(main())

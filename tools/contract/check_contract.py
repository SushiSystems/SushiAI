# check_contract.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the failure sites of the library and of the applications against the class table.

The table is tools/contract/site_classes.tsv.
Usage: python tools/contract/check_contract.py [repository root] [--report] [--rule NAME]
The design is docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md, sections 5 and 6.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_SITE_TABLE = Path("tools/contract/site_classes.tsv")
K_LIBRARY_FOLDERS = ("modules/*/*/include", "modules/*/*/source")
K_PROGRAM_FOLDERS = ("applications",)
K_SITE_FOLDERS = K_LIBRARY_FOLDERS + K_PROGRAM_FOLDERS
K_LIBRARY_DEPTH = 4
K_SOURCE_SUFFIXES = (".hpp", ".cpp", ".h", ".inl")
K_FROZEN: tuple[str, ...] = ()
K_COLUMNS = (
    "file", "line", "function", "kind", "message_head", "cause", "mechanism", "open_question",
    "reason",
)
K_KINDS = ("macro", "throw")
K_CAUSES = ("user-content", "outside", "precondition", "invariant", "unclassified")
K_MECHANISMS = ("status", "fatal", "unclassified")
K_HEAD_LENGTH = 60
K_QUESTION_KEYS = re.compile(r"(?:[a-z0-9_]+(?:;[a-z0-9_]+)*)?")
K_SITE_MACRO_DEFINITION = re.compile(r"\s*#\s*define\s+SA_THROW_IF\b")
K_MACRO_SITE = re.compile(r"\bSA_THROW_IF\s*\(")
K_THROW_SITE = re.compile(r"\bthrow\b")
K_BRIDGE_CALL = re.compile(r"\bDetail\s*::\s*bridge\s*\(")
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


def _directive_free(masked: str, original: str) -> str:
    """Returns masked with the directive that defines SA_THROW_IF blanked, continuations included.

    Every other directive stays, so a macro that wraps SA_THROW_IF holds a site.
    """
    kept: list[str] = []
    in_directive = False
    for masked_line, original_line in zip(masked.split("\n"), original.split("\n")):
        if in_directive or K_SITE_MACRO_DEFINITION.match(masked_line):
            kept.append(" " * len(masked_line))
            in_directive = original_line.rstrip().endswith("\\")
        else:
            kept.append(masked_line)
    return "\n".join(kept)


def mask_source(text: str) -> tuple[str, str]:
    """Returns the C++ text twice with comments blanked: literals kept, then blanked.

    Both results keep every offset and line of the input and hold no definition of SA_THROW_IF.
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
    return _directive_free("".join(kept), text), _directive_free("".join(code), text)


def _call_end(code: str, open_paren: int) -> int:
    """Returns the index of the bracket that closes the one at open_paren."""
    depth = 0
    for index in range(open_paren, len(code)):
        if code[index] in K_OPENERS:
            depth += 1
        elif code[index] in K_CLOSERS:
            depth -= 1
            if depth == 0:
                return index
    return len(code) - 1


def _second_argument(code: str, open_paren: int, close_paren: int) -> tuple[int, int]:
    """Returns the span of the call's second argument, empty when it has one argument."""
    depth = 0
    commas: list[int] = []
    for index in range(open_paren + 1, close_paren):
        if code[index] in K_OPENERS:
            depth += 1
        elif code[index] in K_CLOSERS:
            depth -= 1
        elif code[index] == "," and depth == 0:
            commas.append(index)
    if not commas:
        return close_paren, close_paren
    return commas[0] + 1, commas[1] if len(commas) > 1 else close_paren


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
    """Returns every SA_THROW_IF call and every throw statement of one source text, by line."""
    kept, code = mask_source(text)
    sites: list[Site] = []
    for match in K_MACRO_SITE.finditer(code):
        close_paren = _call_end(code, match.end() - 1)
        start, end = _second_argument(code, match.end() - 1, close_paren)
        line = code.count("\n", 0, match.start()) + 1
        sites.append(Site(file, line, "macro", message_head(kept[start:end])))
    for match in K_THROW_SITE.finditer(code):
        end = code.find(";", match.end())
        statement = kept[match.end():len(kept) if end < 0 else end]
        literal = statement.find('"')
        head = message_head(statement[literal:]) if literal >= 0 else ""
        sites.append(Site(file, code.count("\n", 0, match.start()) + 1, "throw", head))
    return sorted(sites, key=lambda site: site.line)


def count_bridges(text: str) -> int:
    """Returns how many Detail::bridge( calls one source text holds outside comments."""
    return len(K_BRIDGE_CALL.findall(mask_source(text)[1]))


def is_frozen(file: str) -> bool:
    """Returns whether a file is one the wave may not edit yet."""
    return any(
        file == entry or (entry.endswith("/") and file.startswith(entry)) for entry in K_FROZEN
    )


def module_of(file: str) -> str:
    """Returns the tier and module of a library file, or the folder of a program, from its path."""
    parts = file.split("/")
    folder = "/".join(parts[:K_LIBRARY_DEPTH])
    if any(fnmatch.fnmatchcase(folder, pattern) for pattern in K_LIBRARY_FOLDERS):
        return "/".join(parts[1:3])
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


class Repository:
    """Holds one repository's failure sites and the rows of its class table."""

    def __init__(self, root: Path) -> None:
        """Remembers the root; the tree is read when a rule first asks."""
        self.root = root
        self.table = root / K_SITE_TABLE

    @cached_property
    def sources(self) -> dict[str, str]:
        """Returns the text of every source file under the site folders, by relative path."""
        found: dict[str, str] = {}
        for pattern in K_SITE_FOLDERS:
            for folder in sorted(self.root.glob(pattern)):
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix in K_SOURCE_SUFFIXES:
                        name = path.relative_to(self.root).as_posix()
                        found[name] = path.read_text(encoding="utf-8")
        return found

    @cached_property
    def sites(self) -> dict[str, list[Site]]:
        """Returns the sites of each file that has any."""
        found = {file: find_sites(file, text) for file, text in self.sources.items()}
        return {file: sites for file, sites in found.items() if sites}

    @cached_property
    def table_lines(self) -> list[str]:
        """Returns the lines of the class table, empty when the file is absent."""
        if not self.table.is_file():
            return []
        return self.table.read_text(encoding="utf-8").splitlines()

    @cached_property
    def rows(self) -> dict[str, list[Row]]:
        """Returns the well-formed rows of the class table, by file."""
        found: dict[str, list[Row]] = defaultdict(list)
        for number, line in enumerate(self.table_lines[1:], start=2):
            cells = line.split("\t")
            if len(cells) == len(K_COLUMNS) and cells[1].isdigit():
                found[cells[0]].append(Row(number, cells[0], int(cells[1]), *cells[2:]))
        return dict(found)

    @cached_property
    def unmatched(self) -> tuple[list[Site], list[Row]]:
        """Returns the sites with no row and the rows with no site.

        A site and a row match on file, kind and message head; equal heads pair in line order.
        """
        lone_sites: list[Site] = []
        lone_rows: list[Row] = []
        for file in sorted(set(self.sites) | set(self.rows)):
            waiting: dict[tuple[str, str], list[Row]] = defaultdict(list)
            for row in sorted(self.rows.get(file, []), key=lambda row: row.line):
                waiting[(row.kind, row.head)].append(row)
            for site in self.sites.get(file, []):
                if waiting[(site.kind, site.head)]:
                    waiting[(site.kind, site.head)].pop(0)
                else:
                    lone_sites.append(site)
            lone_rows.extend(row for rows in waiting.values() for row in rows)
        return lone_sites, sorted(lone_rows, key=lambda row: row.number)


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns one repository per path given."""
    return [Repository(path if path.is_dir() else path.parent) for path in paths]


def _macro_surplus(repository: Repository) -> Iterator[tuple[str, int, int, int]]:
    """Yields file, first surplus line, site count and row count where sites exceed rows."""
    for file, sites in repository.sites.items():
        macros = [site for site in sites if site.kind == "macro"]
        allowed = sum(1 for row in repository.rows.get(file, []) if row.kind == "macro")
        if len(macros) > allowed:
            yield file, macros[allowed].line, len(macros), allowed


def rule_table_shape(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for a missing table, a wrong header, a malformed or misplaced row."""
    if not repository.table_lines:
        yield Issue(repository.table, 1, f"{K_SITE_TABLE.as_posix()} is missing or empty")
        return
    if tuple(repository.table_lines[0].split("\t")) != K_COLUMNS:
        yield Issue(repository.table, 1, "the header must read " + " ".join(K_COLUMNS))
    previous = ("", 0)
    for number, line in enumerate(repository.table_lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(K_COLUMNS) or not cells[1].isdigit():
            yield Issue(
                repository.table, number, f"a row has {len(K_COLUMNS)} cells and a numeric line"
            )
            continue
        if cells[3] not in K_KINDS or cells[5] not in K_CAUSES or cells[6] not in K_MECHANISMS:
            yield Issue(repository.table, number, "unknown kind, cause or mechanism")
        if not K_QUESTION_KEYS.fullmatch(cells[7]):
            yield Issue(repository.table, number, "open_question holds snake_case keys joined by ;")
        if (cells[0], int(cells[1])) < previous:
            yield Issue(repository.table, number, "rows are sorted by file, then by line")
        previous = (cells[0], int(cells[1]))


def rule_ratchet(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each unfrozen file that holds more SA_THROW_IF sites than rows."""
    for file, line, count, allowed in _macro_surplus(repository):
        if not is_frozen(file):
            message = f"{count} SA_THROW_IF sites, {allowed} rows; a new site uses the new macros"
            yield Issue(repository.root / file, line, message)


def rule_unlisted_site(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each site of an unfrozen file that no row describes."""
    for site in repository.unmatched[0]:
        if not is_frozen(site.file):
            yield Issue(repository.root / site.file, site.line, f"no row for '{site.head}'")


def rule_stale_row(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each row that matches no site of its file."""
    for row in repository.unmatched[1]:
        yield Issue(repository.table, row.number, f"no site in {row.file} for '{row.head}'")


def note_frozen_growth(repository: Repository) -> Iterator[Issue]:
    """Yields a note for each frozen file whose sites outgrew or left its rows."""
    for file, line, count, allowed in _macro_surplus(repository):
        if is_frozen(file):
            yield Issue(
                repository.root / file, line, f"frozen: {count} SA_THROW_IF sites, {allowed} rows"
            )
    for site in repository.unmatched[0]:
        if is_frozen(site.file):
            yield Issue(repository.root / site.file, site.line, f"frozen: no row for '{site.head}'")


def note_unclassified(repository: Repository) -> Iterator[Issue]:
    """Yields a note for each row whose cause or mechanism is unclassified."""
    for rows in repository.rows.values():
        for row in rows:
            if "unclassified" in (row.cause, row.mechanism):
                yield Issue(repository.table, row.number, f"{row.file}:{row.line} has no class yet")


def note_open_question(repository: Repository) -> Iterator[Issue]:
    """Yields one note per open question, at its first row, with the number of rows it holds."""
    first: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for rows in repository.rows.values():
        for row in rows:
            for key in filter(None, row.question.split(";")):
                first[key] = min(first.get(key, row.number), row.number)
                counts[key] += 1
    for key in sorted(counts):
        yield Issue(repository.table, first[key], f"{key}: {counts[key]} rows wait for an answer")


def note_bridge_count(repository: Repository) -> Iterator[Issue]:
    """Yields one note per module that calls Detail::bridge, and one with the total."""
    counts: Counter[str] = Counter()
    for file, text in repository.sources.items():
        counts[module_of(file)] += count_bridges(text)
    for module in sorted(counts):
        if counts[module]:
            yield Issue(repository.root, 0, f"module {module}: {counts[module]} bridge calls")
    yield Issue(repository.root, 0, f"{sum(counts.values())} bridge calls; the last wave needs 0")


K_CHECKER = Checker(
    description="Checks the failure sites against the class table of the error contract.",
    collect=collect,
    rules={
        "rule_table_shape": rule_table_shape,
        "rule_ratchet": rule_ratchet,
        "rule_unlisted_site": rule_unlisted_site,
        "rule_stale_row": rule_stale_row,
    },
)

K_NOTES = {
    "note_frozen_growth": note_frozen_growth,
    "note_unclassified": note_unclassified,
    "note_open_question": note_open_question,
    "note_bridge_count": note_bridge_count,
}


def _roots(argv: list[str]) -> list[Path]:
    """Returns the paths on a command line the runner will parse again."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rule")
    return parser.parse_known_args(argv)[0].paths or [Path.cwd()]


def main(argv: list[str] | None = None) -> int:
    """Prints the notes, which never fail, then runs the rules and returns their exit code."""
    arguments = sys.argv[1:] if argv is None else argv
    for repository in collect([path for path in _roots(arguments) if path.exists()]):
        for name, note in K_NOTES.items():
            for issue in note(repository):
                try:
                    shown = issue.path.resolve().relative_to(repository.root.resolve()).as_posix()
                except ValueError:
                    shown = issue.path.as_posix()
                print(f"note: {shown}:{issue.line}: {name}: {issue.message}")
    return run(K_CHECKER, arguments)


if __name__ == "__main__":
    sys.exit(main())

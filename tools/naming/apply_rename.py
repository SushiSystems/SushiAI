# apply_rename.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Applies tools/naming/rename_map.tsv to the C++ tree as identifier-token replacements.

Usage: python tools/naming/apply_rename.py [repository root] [--map PATH] [--kind KIND] [--dry-run]
The design is docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md, section 2.13.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

K_RENAME_MAP = Path("tools/naming/rename_map.tsv")
K_CODE_ROOTS = ("include", "src", "tests", "apps", "examples")
K_CODE_SUFFIXES = (".hpp", ".cpp", ".h", ".inl")
K_COLUMNS = (
    "kind", "old_name", "new_name", "declaring_file", "occurrences", "document_occurrences",
    "status", "reason",
)
K_STATUS_RENAME = "rename"
K_KIND_ENUMERATOR = "enumerator"
K_APPLIED_KINDS = ("enumerator", "constant", "interface", "macro")
K_ENUM_KINDS = ("enumerator", "sibling_enumerator", "unscoped_enum", "enum_underlying")
K_EXIT_CLEAN = 0
K_EXIT_UNSAFE = 1
K_EXIT_REFUSED = 2
K_TOKEN = re.compile(
    r"""
      (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^()\\\s]{0,16})\(.*?\)(?P=delimiter)")
    | (?P<literal>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)+')
    | (?P<number>\d[\w.']*)
    | (?P<identifier>[A-Za-z_]\w*)
    | (?P<scope>::)
    | (?P<space>\s+)
    | (?P<other>.)
    """,
    re.S | re.X,
)
K_COMMENT_REFERENCE = re.compile(r"(@(?:ref|c|see)\s+)((?:\w+::)*\w+)")
K_ENUM_KEYWORDS = ("class", "struct")
K_SIBLING_NAMESPACES = ("SushiBLAS", "SushiRuntime", "sb", "sr")
K_OPENERS = "([{"
K_CLOSERS = ")]}"
K_WORDS_BEFORE_GLOBAL_SCOPE = frozenset((
    "and", "case", "class", "co_await", "co_return", "co_yield", "const", "constexpr", "delete",
    "do", "else", "enum", "explicit", "extern", "friend", "goto", "inline", "mutable", "namespace",
    "new", "not", "operator", "or", "private", "protected", "public", "return", "sizeof", "static",
    "struct", "throw", "typedef", "typename", "union", "using", "virtual", "volatile",
))


class RenameError(Exception):
    """Reports a rename that would be unsafe: a taken name, a changed literal, a surviving name."""


class RenameRefusedError(RenameError):
    """Reports a rename that may not start: an unreadable map or uncommitted work in the tree."""


@dataclass(frozen=True, slots=True)
class RenameEntry:
    """Holds one row of the rename map."""

    kind: str
    old_name: str
    new_name: str
    declaring_file: str
    occurrences: int
    document_occurrences: int
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class Token:
    """Holds one lexed piece of a C++ text."""

    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class RenameNames:
    """Holds the lookups one pass over a file needs."""

    enumerators: dict[tuple[str, str], str]
    enums: frozenset[str]
    bare: dict[str, str]
    lone: frozenset[str]
    known: frozenset[str]

    @classmethod
    def from_entries(
        cls, entries: Iterable[RenameEntry], kinds: Iterable[str] | None = None
    ) -> RenameNames:
        """Returns the lookups of a map.

        A row marked rename is applied when kinds is None or names its kind. Every row, applied
        or not, gives its enum and its unqualified old name to the lookups that keep a token.
        """
        wanted = None if kinds is None else frozenset(kinds)
        enumerators: dict[tuple[str, str], str] = {}
        enums: set[str] = set()
        bare: dict[str, str] = {}
        known: set[str] = set()
        for entry in entries:
            qualifier, _, name = entry.old_name.rpartition("::")
            if entry.kind in K_ENUM_KINDS:
                enums.add(qualifier.rpartition("::")[2] or name)
            if not qualifier:
                known.add(name)
            if entry.status != K_STATUS_RENAME or (wanted is not None and entry.kind not in wanted):
                continue
            if entry.kind == K_KIND_ENUMERATOR:
                enumerators[(qualifier, name)] = entry.new_name.rpartition("::")[2]
            else:
                bare[entry.old_name] = entry.new_name
        lone = frozenset(name for _, name in enumerators)
        return cls(enumerators, frozenset(enums), bare, lone, frozenset(known))

    def renamed(
        self, qualifier: str | None, name: str, owner: str | None = None, declares: bool = False
    ) -> tuple[str, str] | None:
        """Returns the map key and the new spelling of one identifier, or None when it stays.

        Args:
            qualifier: The identifier before the `::` that precedes the name, if any.
            name: The identifier itself.
            owner: The enum whose body holds an unqualified name, if any.
            declares: Whether the name stands where an enum body declares an enumerator.
        """
        for enum in (qualifier, owner):
            if enum is not None and (enum, name) in self.enumerators:
                return f"{enum}::{name}", self.enumerators[(enum, name)]
        if declares or qualifier in self.enums or name not in self.bare:
            return None
        return name, self.bare[name]

    def survives(self, qualifier: str | None, name: str, in_enum: bool) -> bool:
        """Returns whether a token left in place spells an applied enumerator no lookup explains.

        A name an enum of the map qualifies, a name inside another enum's body and an unqualified
        old name of another row are explained.
        """
        explained = qualifier in self.enums or in_enum or name in self.known
        return name in self.lone and not explained


@dataclass(slots=True)
class Rewrite:
    """Holds one file's outcome: the new text, the count per old name, what was left."""

    text: str
    counts: Counter[str] = field(default_factory=Counter)
    comment_mentions: int = 0
    unresolved: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Place:
    """Holds where an identifier stands in a qualified name."""

    is_qualified: bool
    qualifier: str | None
    root: str


class QualifierChain:
    """Tracks the qualified name that the code tokens passing by spell."""

    def __init__(self) -> None:
        """Starts before the first token of a text."""
        self._last: Token | None = None
        self._earlier: Token | None = None
        self._root = ""

    def follows(self, text: str) -> bool:
        """Returns whether the token seen last has the given text."""
        return self._last is not None and self._last.text == text

    def place(self, identifier: Token) -> Place:
        """Returns the place of an identifier that comes next.

        A `::` continues a name only after an identifier that is no keyword, or after `>`.
        """
        last, earlier = self._last, self._earlier
        if last is None or last.kind != "scope" or earlier is None:
            return Place(False, None, identifier.text)
        is_name = earlier.kind == "identifier" and earlier.text not in K_WORDS_BEFORE_GLOBAL_SCOPE
        if not is_name and earlier.text != ">":
            return Place(False, None, identifier.text)
        return Place(True, earlier.text if is_name else None, self._root)

    def see(self, token: Token) -> None:
        """Takes the next token that is neither white space nor a comment."""
        if token.kind == "identifier":
            self._root = self.place(token).root
        self._earlier, self._last = self._last, token


class EnumScope:
    """Tracks whether the code tokens passing by stand in the head or the body of an enum."""

    def __init__(self) -> None:
        """Starts outside every enum."""
        self._stage = ""
        self._name: str | None = None
        self._depth = 0
        self._at_item_start = False

    @property
    def in_head(self) -> bool:
        """Returns whether the next token stands between `enum` and its body."""
        return self._stage == "head"

    @property
    def in_body(self) -> bool:
        """Returns whether the next token stands inside the braces of an enum."""
        return self._stage == "body"

    @property
    def owner(self) -> str | None:
        """Returns the name of the enum whose body holds the next token, if it has one."""
        return self._name if self.in_body else None

    @property
    def declares(self) -> bool:
        """Returns whether an identifier that comes next declares an enumerator."""
        return self.in_body and self._depth == 1 and self._at_item_start

    def see(self, token: Token) -> None:
        """Takes the next token that is neither white space nor a comment."""
        if token.kind == "identifier":
            if token.text == "enum":
                self._stage, self._name = "head", None
            elif self.in_head and self._name is None and token.text not in K_ENUM_KEYWORDS:
                self._name = token.text
            self._at_item_start = False
        elif token.kind == "other" and self.in_head and token.text != ":":
            self._stage = "body" if token.text == "{" else ""
            self._depth, self._at_item_start = 1, True
        elif token.kind == "other" and self.in_body:
            self._depth += (token.text in K_OPENERS) - (token.text in K_CLOSERS)
            self._at_item_start = self._depth == 1 and token.text == ","
            self._stage = "body" if self._depth > 0 else ""


def tokenize(text: str) -> Iterator[Token]:
    """Yields the tokens of a C++ text; joined in order they give the text back."""
    for match in K_TOKEN.finditer(text):
        kind = match.lastgroup if match.lastgroup != "delimiter" else "raw"
        yield Token("literal" if kind == "raw" else str(kind), match.group())


def code_only(text: str) -> str:
    """Returns the text with comments and literals blanked, every offset and line kept."""
    return "".join(
        re.sub(r"[^\n]", " ", token.text) if token.kind in ("comment", "literal") else token.text
        for token in tokenize(text)
    )


def literals_of(text: str) -> Counter[str]:
    """Returns how often each string or character literal occurs in a C++ text."""
    return Counter(token.text for token in tokenize(text) if token.kind == "literal")


def parse_rename_map(text: str, origin: str) -> list[RenameEntry]:
    """Returns the rows of a rename map text.

    Args:
        text: The text of the map.
        origin: The name the refusals use for the text.

    Raises:
        RenameRefusedError: The header is not K_COLUMNS or a row is malformed.
    """
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != K_COLUMNS:
        raise RenameRefusedError(f"{origin}: the header must read {' '.join(K_COLUMNS)}")
    entries: list[RenameEntry] = []
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(K_COLUMNS) or not (cells[4].isdigit() and cells[5].isdigit()):
            raise RenameRefusedError(
                f"{origin}:{number}: a row has {len(K_COLUMNS)} cells and two counts"
            )
        entries.append(RenameEntry(*cells[:4], int(cells[4]), int(cells[5]), *cells[6:]))
    return entries


def format_rename_map(entries: Iterable[RenameEntry]) -> str:
    """Returns the rows as tab-separated text under the K_COLUMNS header."""
    lines = ["\t".join(K_COLUMNS)]
    for entry in entries:
        lines.append("\t".join(str(getattr(entry, column)) for column in K_COLUMNS))
    return "\n".join(lines) + "\n"


def write_rename_map(path: Path, entries: Iterable[RenameEntry]) -> None:
    """Writes the rows as a tab-separated file under the K_COLUMNS header."""
    path.write_text(format_rename_map(entries), encoding="utf-8", newline="\n")


class RepositoryFiles(Protocol):
    """Gives the tools the text files of a repository without saying where they live."""

    def texts(
        self, roots: Iterable[str], suffixes: Iterable[str], errors: str = "strict"
    ) -> dict[str, str]:
        """Returns the text of each file below the roots with one of the suffixes, by relative path.

        The roots come in the order given and the files of one root in path order. Line endings
        are kept as they are.
        """

    def read(self, name: str | Path) -> str | None:
        """Returns the text of one file, None when there is none."""

    def write(self, name: str, text: str) -> None:
        """Replaces the text of one file, line endings kept as given."""


class DiskRepository:
    """Reads and writes the files of a repository on disk."""

    def __init__(self, root: Path) -> None:
        """Remembers the root the relative paths hang from."""
        self.root = root

    def texts(
        self, roots: Iterable[str], suffixes: Iterable[str], errors: str = "strict"
    ) -> dict[str, str]:
        """Returns the text of each file below the roots with one of the suffixes, by relative path."""
        wanted = tuple(suffixes)
        found: dict[str, str] = {}
        for top in roots:
            for path in sorted((self.root / top).rglob("*")):
                if path.is_file() and path.suffix in wanted:
                    with path.open(encoding="utf-8", errors=errors, newline="") as stream:
                        found[path.relative_to(self.root).as_posix()] = stream.read()
        return found

    def read(self, name: str | Path) -> str | None:
        """Returns the text of one file, None when the path names no file."""
        path = self.root / name
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write(self, name: str, text: str) -> None:
        """Replaces the text of one file, line endings kept as given."""
        with (self.root / name).open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)


class MemoryRepository:
    """Holds the files of a repository as a mapping of relative path to text."""

    def __init__(self, files: Mapping[str, str]) -> None:
        """Remembers the files."""
        self.files = dict(files)

    def texts(
        self, roots: Iterable[str], suffixes: Iterable[str], errors: str = "strict"
    ) -> dict[str, str]:
        """Returns the text of each file below the roots with one of the suffixes, by relative path."""
        wanted = tuple(suffixes)
        return {
            name: self.files[name]
            for top in roots
            for name in sorted(self.files, key=lambda name: name.split("/"))
            if name.startswith(f"{top}/") and PurePosixPath(name).suffix in wanted
        }

    def read(self, name: str | Path) -> str | None:
        """Returns the text of one file, None when the memory holds none."""
        return self.files.get(Path(name).as_posix())

    def write(self, name: str, text: str) -> None:
        """Replaces the text of one file."""
        self.files[name] = text


def load_rename_map(files: RepositoryFiles, name: str | Path) -> list[RenameEntry]:
    """Returns the rows of the rename map a repository holds under a name.

    Raises:
        RenameRefusedError: The map is absent, its header is not K_COLUMNS or a row is malformed.
    """
    text = files.read(name)
    if text is None:
        raise RenameRefusedError(f"no rename map at {name}")
    return parse_rename_map(text, str(name))


def _rewrite_comment(comment: str, names: RenameNames, outcome: Rewrite) -> str:
    """Returns a comment with the names after @ref, @c and @see renamed; counts the rest."""

    def replace(match: re.Match[str]) -> str:
        qualifier, _, name = match.group(2).rpartition("::")
        last_qualifier = qualifier.rpartition("::")[2] or None
        renamed = names.renamed(last_qualifier, name)
        if renamed is None:
            return match.group()
        outcome.counts[renamed[0]] += 1
        prefix = f"{qualifier}::" if qualifier else ""
        return f"{match.group(1)}{prefix}{renamed[1]}"

    rewritten = K_COMMENT_REFERENCE.sub(replace, comment)
    words = re.findall(r"[A-Za-z_]\w*", rewritten)
    pairs = re.findall(r"(\w+)::(\w+)", rewritten)
    outcome.comment_mentions += sum(1 for word in words if word in names.bare)
    outcome.comment_mentions += sum(1 for pair in pairs if pair in names.enumerators)
    return rewritten


def _resolve(
    token: Token,
    line: int,
    chain: QualifierChain,
    scope: EnumScope,
    names: RenameNames,
    outcome: Rewrite,
) -> str:
    """Returns the spelling one identifier leaves the pass with; records a rename or a survivor."""
    place = chain.place(token)
    if place.is_qualified and place.root in K_SIBLING_NAMESPACES:
        return token.text
    owner = None if place.is_qualified else scope.owner
    declares = scope.declares and not place.is_qualified
    renamed = names.renamed(place.qualifier, token.text, owner, declares)
    if renamed is not None:
        outcome.counts[renamed[0]] += 1
        return renamed[1]
    if names.survives(place.qualifier, token.text, scope.in_body and not place.is_qualified):
        qualified = f"{place.qualifier or ''}::" if place.is_qualified else ""
        outcome.unresolved.append(f"{line}: {qualified}{token.text}")
    return token.text


def rewrite(text: str, names: RenameNames) -> Rewrite:
    """Returns one C++ text with every mapped identifier token renamed.

    Literals, #include lines and names qualified from a sibling's namespace stay as they are;
    a comment changes only after @ref, @c and @see. The outcome lists, by line, each token that
    still spells an applied enumerator and that no qualifier resolves.
    """
    outcome = Rewrite("")
    pieces: list[str] = []
    chain = QualifierChain()
    scope = EnumScope()
    in_include = False
    line = 1
    for token in tokenize(text):
        piece = token.text
        if token.kind == "comment":
            piece = _rewrite_comment(token.text, names, outcome)
        elif token.kind == "space":
            in_include = in_include and "\n" not in token.text
        else:
            if token.kind == "identifier" and not in_include:
                in_include = chain.follows("#") and token.text == "include"
                if not in_include and not scope.in_head:
                    piece = _resolve(token, line, chain, scope, names, outcome)
            chain.see(token)
            scope.see(token)
        pieces.append(piece)
        line += token.text.count("\n")
    outcome.text = "".join(pieces)
    return outcome


def find_collisions(files: RepositoryFiles, entries: Iterable[RenameEntry]) -> list[str]:
    """Returns one line per new name that some file under the code roots already uses as a token."""
    wanted: dict[tuple[str | None, str], str] = {}
    for entry in entries:
        if entry.status == K_STATUS_RENAME:
            qualifier, _, name = entry.new_name.rpartition("::")
            wanted[(qualifier or None, name)] = entry.old_name
    found: list[str] = []
    for shown, text in files.texts(K_CODE_ROOTS, K_CODE_SUFFIXES).items():
        before: tuple[Token | None, Token | None] = (None, None)
        for token in tokenize(text):
            if token.kind == "identifier":
                last, earlier = before
                qualified = last is not None and last.kind == "scope" and earlier is not None
                keys = [
                    (earlier.text if qualified and earlier else None, token.text),
                    (None, token.text),
                ]
                for key in dict.fromkeys(keys):
                    if key in wanted:
                        found.append(
                            f"{shown}: {token.text} exists; {wanted.pop(key)} cannot take it"
                        )
            if token.kind not in ("space", "comment"):
                before = (token, before[0])
    return found


def uncommitted_paths(root: Path) -> list[str]:
    """Returns the paths under the code roots that git status lists as changed or untracked.

    Raises:
        RenameRefusedError: git could not report the status of the repository.
    """
    command = [
        "git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--",
        *K_CODE_ROOTS,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise RenameRefusedError(f"git status could not run: {error}") from error
    if result.returncode != 0:
        raise RenameRefusedError(f"git status failed: {result.stderr.strip()}")
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def check_literals_unchanged(shown: str, original: str, rewritten: str) -> None:
    """Refuses a rewrite whose string or character literals differ from the original's.

    Raises:
        RenameError: The literals of the two texts differ.
    """
    if literals_of(original) != literals_of(rewritten):
        raise RenameError(f"{shown}: a string literal would change")


def apply_rename(
    root: Path,
    entries: list[RenameEntry],
    dry_run: bool,
    list_uncommitted: Callable[[Path], list[str]] = uncommitted_paths,
    kinds: Iterable[str] | None = None,
    files: RepositoryFiles | None = None,
) -> dict[str, Rewrite]:
    """Returns the rewrite of every file the map changes, and writes them unless dry_run.

    Args:
        root: The repository root.
        entries: Every row of the map; the rows that are not applied still protect their names.
        dry_run: Whether to leave the tree as it is.
        list_uncommitted: The reader of the freeze, replaced in tests.
        kinds: The kinds to apply; None applies every row marked rename.
        files: Where the files live; the folder at root when None.

    Raises:
        RenameRefusedError: The tree holds uncommitted work under the code roots.
        RenameError: A new name is taken, a literal changed or an old name survived.
    """
    repository = DiskRepository(root) if files is None else files
    if not dry_run:
        dirty = list_uncommitted(root)
        if dirty:
            raise RenameRefusedError(
                f"{len(dirty)} uncommitted paths under the code roots, first {dirty[0]}"
            )
    wanted = None if kinds is None else frozenset(kinds)
    applied = [entry for entry in entries if wanted is None or entry.kind in wanted]
    collisions = find_collisions(repository, applied)
    if collisions:
        raise RenameError("; ".join(collisions))
    names = RenameNames.from_entries(entries, wanted)
    changed: dict[str, Rewrite] = {}
    survivors: list[str] = []
    for shown, original in repository.texts(K_CODE_ROOTS, K_CODE_SUFFIXES).items():
        outcome = rewrite(original, names)
        check_literals_unchanged(shown, original, outcome.text)
        if rewrite(outcome.text, names).counts:
            raise RenameError(f"{shown}: a second pass would rename a token again")
        survivors.extend(f"{shown}:{left}" for left in outcome.unresolved)
        if outcome.counts:
            changed[shown] = outcome
    if survivors:
        listed = "\n  ".join(survivors)
        raise RenameError(f"{len(survivors)} old enumerator names survive:\n  {listed}")
    if not dry_run:
        for shown, outcome in changed.items():
            repository.write(shown, outcome.text)
    return changed


def report_lines(changed: Mapping[str, Rewrite], dry_run: bool) -> list[str]:
    """Returns the lines the command line prints: one for each changed file, then the total."""
    lines = []
    for shown, outcome in changed.items():
        mentions = outcome.comment_mentions
        left = f", {mentions} comment mentions left" if mentions else ""
        lines.append(f"{shown}: {sum(outcome.counts.values())}{left}")
    verb = "would rename" if dry_run else "renamed"
    total = sum(sum(outcome.counts.values()) for outcome in changed.values())
    lines.append(f"{verb} {total} tokens in {len(changed)} files")
    return lines


def main(
    argv: list[str] | None = None,
    files: RepositoryFiles | None = None,
    list_uncommitted: Callable[[Path], list[str]] = uncommitted_paths,
) -> int:
    """Runs the tool from the command line and returns its exit code.

    Args:
        argv: The command line without the program name; the process's own when None.
        files: Where the files live; the folder the command line names when None.
        list_uncommitted: The reader of the freeze, replaced in tests.
    """
    parser = argparse.ArgumentParser(description="Applies the rename map to the C++ tree.")
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--map", type=Path, help=f"default: {K_RENAME_MAP.as_posix()}")
    parser.add_argument(
        "--dry-run", action="store_true", help="print per-file counts, write nothing"
    )
    parser.add_argument(
        "--kind", choices=K_APPLIED_KINDS, help="apply the rows of one kind; default: all"
    )
    arguments = parser.parse_args(argv)
    kinds = None if arguments.kind is None else (arguments.kind,)
    repository = DiskRepository(arguments.root) if files is None else files
    map_name = Path(os.path.abspath(arguments.map)) if arguments.map else K_RENAME_MAP
    try:
        entries = load_rename_map(repository, map_name)
        changed = apply_rename(
            arguments.root,
            entries,
            arguments.dry_run,
            list_uncommitted=list_uncommitted,
            kinds=kinds,
            files=repository,
        )
    except RenameError as error:
        print(f"apply_rename: refused: {error}", file=sys.stderr)
        return K_EXIT_REFUSED if isinstance(error, RenameRefusedError) else K_EXIT_UNSAFE
    for line in report_lines(changed, arguments.dry_run):
        print(line)
    return K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

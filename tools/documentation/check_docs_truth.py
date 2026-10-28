# check_docs_truth.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks a repository's manual against its source: names, paths, examples, history, size.

Usage: python tools/documentation/check_docs_truth.py [repository root] [--report] [--rule NAME]
           [--source FOLDER ...]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import K_EXIT_CLEAN, Checker, Issue, run  # noqa: E402
from checking.ignored_paths import ignored_folders  # noqa: E402
from settings import K_MACRO_PREFIXES  # noqa: E402


class PageKind(Enum):
    """Names the kinds of manual page the rules tell apart."""

    ROOT_README = "Root `README.md`"
    MANUAL_PAGE = "Manual page"
    MODULE_README = "Module README"
    DESIGN_DOCUMENT = "Design document"
    RULE_FILE = "Rule file"


K_SOURCE_FOLDERS = (
    ".config", ".github", "Assets", "applications", "claude", "cli", "cmake", "engine",
    "include", "modules", "samples", "skills", "source", "tests", "tools",
)
K_BUILD_FOLDERS = frozenset(
    {
        ".git", "build", "out", "dist", "node_modules", "__pycache__", ".venv", "venv",
        "Library", "Logs", "obj",
    }
)
K_SKIPPED_FOLDERS = K_BUILD_FOLDERS | {"third_party", "ThirdParty"}
K_INDEXED_SUFFIXES = frozenset(
    {
        "", ".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".inl", ".inc", ".in",
        ".glsl", ".comp", ".vert", ".frag", ".geom", ".ts", ".tsx", ".js", ".mjs", ".cjs",
        ".py", ".pyi", ".cs", ".cmake", ".txt", ".toml", ".json", ".yml", ".yaml", ".sh",
        ".ps1", ".bat", ".cfg", ".ini", ".md",
    }
)
K_PAGE_NAME = "README.md"
K_LINE_CEILINGS = {
    PageKind.ROOT_README: 120,
    PageKind.MANUAL_PAGE: 600,
    PageKind.MODULE_README: 800,
}
K_SOURCE_LANGUAGES = frozenset(
    {
        "c", "h", "cc", "cpp", "c++", "cxx", "hpp", "glsl", "python", "py", "typescript", "ts",
        "tsx", "javascript", "js", "csharp", "cs",
    }
)
K_HASH_COMMENT_LANGUAGES = frozenset({"python", "py"})
K_FOREIGN_NAMESPACES = frozenset({"std", "sycl", "cl", "testing"})
K_NAMING_WORDS = frozenset({"snake_case", "kebab_case", "PascalCase", "CamelCase"})
K_FOREIGN_MACRO_PREFIXES = frozenset({"K", "CMAKE", "CPACK", "EXPECT", "ASSERT", "TEST"})
K_MACRO_PREFIX_FLOOR = 20
K_ALLOW_LIST = Path("docs/.docs_truth_allow")
K_UNCHECKED_DOCUMENTS = frozenset({"docs/reference/CHANGELOG.md", ".github/CODE_OF_CONDUCT.md"})
K_UNCHECKED_DOCS_FOLDERS = frozenset({"agent", "archive"})
K_HISTORY_EXEMPT_FOLDER = Path("docs") / "backlog"
K_PASSAGE_FLOOR = 300
K_GENERATED_MARK = ":begin -->"
K_RULE_FOLDERS = ("skills", "claude")
K_RULE_SKIPPED_FOLDERS = frozenset({"research"})
K_CITATION_REACH = 3
K_NOT_COMPILED = "not compiled"
K_ELISIONS = frozenset({"...", "// ...", "# ..."})

K_FENCE = re.compile(r"^(`{3,}|~{3,})\s*([^\s`{]*)")
K_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
K_STATUS_LINE = re.compile(r"^\*\*Status:\*\* \S")
K_CITATION = re.compile(r"^Source: (?:`([^`]+)`|(" + K_NOT_COMPILED + r"))\.?$")
K_HISTORY = re.compile(
    r"\b(?:previously|no longer|formerly|was renamed|deprecated since)\b"
    r"|\b(?:as of|new in|up to) v(?=\d|ersion)"
    r"|(?<!\bis )(?<!\bare )(?<!\bbe )(?<!\bwas )(?<!\bwere )(?<!\bbeen )(?<!\bbeing )"
    r"\bused to\b",
    re.IGNORECASE,
)
K_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")
K_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
K_QUALIFIED = re.compile(r"(?<![\w:])(?:[A-Za-z_]\w*::)+~?[A-Za-z_]\w*")
K_CALL = re.compile(r"(?<![\w:-])([A-Za-z_]\w{2,})\(")
K_PASCAL = re.compile(r"(?<![\w:-])(?=[A-Z0-9]*[a-z])(?:[A-Z]+(?![a-z])|[A-Z][a-z0-9]+){2,}(?!\w)")
K_SNAKE = re.compile(r"(?<![\w:-])_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+_?(?!\w)")
K_UPPER = re.compile(r"(?<![\w:-])[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?!\w)")
K_PLACEHOLDER = re.compile(r"(?<![A-Za-z0-9])<[A-Za-z_][\w -]*>|\*|\.\.\.|\$|\{|%")
K_PLACEHOLDER_TAIL = re.compile(r"^[\w -]+>/")
K_FILE_NAME = re.compile(r"^[\w.+-]*\.[A-Za-z][A-Za-z0-9]{0,5}$")
K_PATH_TRIM = re.compile(r"^[\"'(<\[@]+|[\"')>\],;:.]+$|(?::\d+(?:-\d+)?|#\S*)$")
K_OUTSIDE_PATH = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|[/~\\]|\.\.?/\.\.)|^-")
K_INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")
K_LINE_COMMENT = re.compile(r"//.*$")
K_HASH_COMMENT = re.compile(r"#.*$")
K_BLOCK_COMMENT = re.compile(r"/\*.*?\*/")
K_STRING = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
K_DECLARED_TYPE = re.compile(
    r"\b(?:struct|class|enum|union|using|namespace|concept|def|define)\s+([A-Za-z_]\w*)"
)
K_DECLARED_VALUE = re.compile(
    r"(?:^|[\s(,])(?!(?:return|co_return|co_await|co_yield|throw|new|delete|else|case|await"
    r"|yield|not|and|or|in|typename|import|from|ifdef|ifndef|if|elif|defined)\b)"
    r"[A-Za-z_][\w:<>,]*[&*]*\s+[&*]*([A-Za-z_]\w*)\s*(?=[=;({\[,)]|$)",
    re.MULTILINE,
)
K_ARGUMENTS = re.compile(r"\([^()]*\)")
K_ASSIGNED = re.compile(r"(?<![\w.:>])([A-Za-z_]\w*)\s*=(?!=)")


@dataclass(frozen=True, slots=True)
class Fence:
    """Holds one fenced block of a page: its language, its lines and where they start."""

    language: str
    first: int
    lines: tuple[str, ...]

    @property
    def is_source(self) -> bool:
        """Returns whether the block is written in a programming language."""
        return self.language in K_SOURCE_LANGUAGES

    @property
    def after(self) -> int:
        """Returns the number of the first line after the closing fence."""
        return self.first + len(self.lines) + 1

    def code(self) -> str:
        """Returns the block without comments, string literals and include lines."""
        kept: list[str] = []
        for line in self.lines:
            if K_INCLUDE.match(line):
                continue
            line = K_BLOCK_COMMENT.sub(" ", K_LINE_COMMENT.sub("", K_STRING.sub('""', line)))
            if self.language in K_HASH_COMMENT_LANGUAGES:
                line = K_HASH_COMMENT.sub("", line)
            kept.append(line)
        return "\n".join(kept)

    def declared(self) -> frozenset[str]:
        """Returns the names the block itself declares, which no source file need hold."""
        code = self.code()
        statements = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        return frozenset(
            K_DECLARED_TYPE.findall(code)
            + K_ASSIGNED.findall(statements)
            + K_DECLARED_VALUE.findall(statements)
        )


@dataclass(frozen=True, slots=True)
class Page:
    """Holds one manual page split into prose lines and fenced blocks."""

    path: Path
    kind: PageKind
    lines: tuple[str, ...]
    prose: tuple[str, ...]
    fences: tuple[Fence, ...]

    @property
    def describes_code(self) -> bool:
        """Returns whether the page states what the code is, not what it is meant to become."""
        return self.kind is not PageKind.DESIGN_DOCUMENT

    def spans(self) -> Iterator[tuple[int, str]]:
        """Yields the line number and the text of each code span outside a fenced block."""
        for number, line in enumerate(self.prose, start=1):
            for match in K_CODE_SPAN.finditer(line):
                yield number, match.group(2)

    def plain(self) -> Iterator[tuple[int, str]]:
        """Yields each prose line with its code spans removed."""
        for number, line in enumerate(self.prose, start=1):
            yield number, K_CODE_SPAN.sub("", line)

    def paragraphs(self) -> Iterator[tuple[int, str]]:
        """Yields the first line number and the whitespace-normalised text of each paragraph."""
        start, words = 0, []
        for number, line in enumerate((*self.prose, ""), start=1):
            if line.strip():
                start = start or number
                words += line.split()
            elif words:
                yield start, " ".join(words)
                start, words = 0, []

    def citation(self, fence: Fence) -> tuple[int, str] | None:
        """Returns the line number and the target of the citation that follows a block."""
        for number in range(fence.after, min(fence.after + K_CITATION_REACH, len(self.lines) + 1)):
            match = K_CITATION.match(self.lines[number - 1].strip())
            if match:
                return number, match.group(1) or match.group(2)
        return None


def read_page(path: Path, kind: PageKind) -> Page:
    """Returns a page with its fenced blocks lifted out of the prose, line numbers kept."""
    lines = tuple(path.read_text(encoding="utf-8-sig").splitlines())
    prose: list[str] = []
    fences: list[Fence] = []
    marker, language, first, body = "", "", 0, []
    for number, line in enumerate(lines, start=1):
        opener = K_FENCE.match(line.lstrip())
        if marker:
            if opener and opener.group(1).startswith(marker) and not opener.group(2):
                fences.append(Fence(language, first, tuple(body)))
                marker = ""
            else:
                body.append(line)
            prose.append("")
        elif opener and opener.group(1)[0] not in line.lstrip()[len(opener.group(1)):]:
            marker, language, first, body = opener.group(1), opener.group(2).lower(), number + 1, []
            prose.append("")
        else:
            prose.append(line)
    if marker:
        fences.append(Fence(language, first, tuple(body)))
    return Page(path, kind, lines, tuple(prose), tuple(fences))


@dataclass(frozen=True, slots=True)
class AllowEntry:
    """Holds one line of the allow-list: its pattern and the reason written after it."""

    number: int
    text: str
    reason: str

    @property
    def pattern(self) -> re.Pattern[str] | None:
        """Returns the entry as a regular expression, or None when it does not compile."""
        try:
            return re.compile(self.text)
        except re.error:
            return None

    def allows(self, name: str) -> bool:
        """Returns whether the entry matches the whole of a name or a path."""
        pattern = self.pattern
        return bool(pattern and self.text and pattern.fullmatch(name))


def read_allow_list(path: Path) -> tuple[AllowEntry, ...]:
    """Returns the entries of an allow-list file, skipping blank lines and whole-line comments."""
    if not path.is_file():
        return ()
    entries: list[AllowEntry] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("#"):
            text, _, reason = line.partition(" #")
            entries.append(AllowEntry(number, text.strip(), reason.strip()))
    return tuple(entries)


def _walk(
    folder: Path, ignored: frozenset[Path] = frozenset()
) -> Iterator[tuple[Path, list[str], list[str]]]:
    """Yields each folder below a path with its entries, without entering skipped or ignored folders."""
    for current, folders, files in os.walk(folder):
        folders[:] = sorted(
            name for name in folders
            if name not in K_SKIPPED_FOLDERS and (Path(current) / name).resolve() not in ignored
        )
        yield Path(current), folders, sorted(files)


@final
class SourceTree:
    """Indexes the words of a repository's source folders and the paths of the repository."""

    def __init__(self, root: Path, folders: tuple[str, ...]) -> None:
        """Stores the root and the source folders; nothing is read until a rule asks."""
        self.root = root
        self.folders = tuple(root / name for name in folders if (root / name).is_dir())

    def files(self) -> Iterator[Path]:
        """Yields every indexed file: the root's own files and those under the source folders."""
        for entry in sorted(self.root.iterdir()):
            if entry.is_file():
                yield entry
        for folder in self.folders:
            for current, _, names in _walk(folder, self.ignored):
                yield from (current / name for name in names)

    @cached_property
    def ignored(self) -> frozenset[Path]:
        """Returns the folders git ignores, which the index never enters."""
        return ignored_folders(self.root)

    @cached_property
    def words(self) -> frozenset[str]:
        """Returns every identifier-shaped word of the indexed files and of their paths."""
        words: set[str] = set()
        for path in self.files():
            words.update(K_WORD.findall(path.as_posix()))
            if path.suffix.lower() in K_INDEXED_SUFFIXES and path.name != K_PAGE_NAME:
                words.update(K_WORD.findall(path.read_text(encoding="utf-8", errors="ignore")))
        return frozenset(words)

    @cached_property
    def macro_prefixes(self) -> frozenset[str]:
        """Returns the leading segments that mark an upper-case name as the repository's own."""
        if K_MACRO_PREFIXES:
            return frozenset(K_MACRO_PREFIXES)
        counts: dict[str, int] = {}
        for word in self.words:
            if K_UPPER.fullmatch(word):
                prefix = word.split("_", 1)[0]
                counts[prefix] = counts.get(prefix, 0) + 1
        return frozenset(
            prefix for prefix, count in counts.items()
            if count >= K_MACRO_PREFIX_FLOOR and prefix not in K_FOREIGN_MACRO_PREFIXES
        )

    @cached_property
    def paths(self) -> frozenset[str]:
        """Returns every file and folder of the repository and each trailing part of its path."""
        tails: set[str] = set()
        for current, folders, files in os.walk(self.root):
            folders[:] = [
                name for name in folders
                if name not in K_BUILD_FOLDERS and (Path(current) / name).resolve() not in self.ignored
            ]
            for name in (*folders, *files):
                parts = (Path(current) / name).relative_to(self.root).parts
                tails.update("/".join(parts[index:]) for index in range(len(parts)))
        return frozenset(tails)

    @cached_property
    def top_level(self) -> frozenset[str]:
        """Returns the names of the entries directly under the root that git does not ignore."""
        return frozenset(entry.name for entry in self.root.iterdir() if entry.resolve() not in self.ignored)

    def lines_of(self, relative: str) -> frozenset[str] | None:
        """Returns the trimmed lines of a file named from the root, or None when it is absent."""
        path = self.root / relative
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        return frozenset(line.strip() for line in text.splitlines())


@final
class Manual:
    """Holds one repository's manual pages, its allow-list and its source index."""

    def __init__(self, root: Path, source_folders: tuple[str, ...], only: frozenset[Path] = frozenset()) -> None:
        """Stores the root and builds the source tree the rules look names up in.

        When only is given, the manual holds those pages alone.
        """
        self.root = root
        self.only = only
        self.source = SourceTree(root, source_folders)
        self.allow_path = root / K_ALLOW_LIST

    @cached_property
    def pages(self) -> tuple[Page, ...]:
        """Returns the root README, the documents under docs/ and the source folders' pages."""
        found: list[tuple[Path, PageKind]] = []
        if (self.root / K_PAGE_NAME).is_file():
            found.append((self.root / K_PAGE_NAME, PageKind.ROOT_README))
        docs = self.root / "docs"
        unchecked = {docs / name for name in K_UNCHECKED_DOCS_FOLDERS}
        for path in sorted(docs.rglob("*.md")) if docs.is_dir() else []:
            if unchecked & set(path.parents):
                continue
            if path.relative_to(self.root).as_posix() in K_UNCHECKED_DOCUMENTS:
                continue
            intent = bool({docs / "design", docs / "backlog"} & set(path.parents))
            found.append((path, PageKind.DESIGN_DOCUMENT if intent else PageKind.MANUAL_PAGE))
        for folder in self.source.folders:
            if folder.name in K_RULE_FOLDERS:
                continue
            for current, _, names in _walk(folder, self.source.ignored):
                found.extend((current / name, PageKind.MODULE_README) for name in names if name.endswith(".md")
                             and (current / name).relative_to(self.root).as_posix() not in K_UNCHECKED_DOCUMENTS)
        if self.only:
            kept = [(path, kind) for path, kind in found if path.resolve() in self.only]
            listed = {path.resolve() for path, _ in kept}
            found = kept + [(path, PageKind.MODULE_README) for path in sorted(self.only - listed)]
        return tuple(read_page(path, kind) for path, kind in found)

    @cached_property
    def rule_files(self) -> tuple[Page, ...]:
        """Returns the skill and instruction files, read only for repeated passages."""
        pages = {page.path for page in self.pages}
        found: list[Path] = []
        for name in K_RULE_FOLDERS:
            folder = self.root / name
            for current, folders, names in _walk(folder, self.source.ignored) if folder.is_dir() else ():
                folders[:] = [entry for entry in folders if entry not in K_RULE_SKIPPED_FOLDERS]
                found.extend(
                    current / entry for entry in names
                    if entry.endswith(".md") and current / entry not in pages
                )
        return tuple(read_page(path, PageKind.RULE_FILE) for path in found)

    @cached_property
    def allow_list(self) -> tuple[AllowEntry, ...]:
        """Returns the entries of docs/.docs_truth_allow."""
        return read_allow_list(self.allow_path)

    def is_allowed(self, name: str) -> bool:
        """Returns whether the allow-list excuses a name or a path."""
        return any(entry.allows(name) for entry in self.allow_list)

    def unverified_examples(self) -> int:
        """Returns how many source blocks carry the marker that no file backs them."""
        return sum(
            1
            for page in self.pages
            for fence in page.fences
            if fence.is_source and (page.citation(fence) or (0, ""))[1] == K_NOT_COMPILED
        )


def _without_paths(span: str) -> str:
    """Returns a code span without the tokens that are paths, file names, flags or patterns."""
    kept = [
        token for token in span.split()
        if "/" not in token and "\\" not in token and not token.startswith("-")
        and not K_PLACEHOLDER.search(token) and not K_FILE_NAME.match(K_PATH_TRIM.sub("", token))
    ]
    return " ".join(kept)


def _without_arguments(code: str) -> str:
    """Returns code with every argument list emptied, innermost first."""
    emptied = K_ARGUMENTS.sub(" ", code)
    return emptied if emptied == code else _without_arguments(emptied)


def _names(manual: Manual, code: str, bare_snake: bool) -> Iterator[str]:
    """Yields each name in a piece of code that reads as a name of the repository's own."""
    for chain in K_QUALIFIED.findall(code):
        if chain.split("::", 1)[0] not in K_FOREIGN_NAMESPACES:
            yield chain
    rest = K_QUALIFIED.sub(" ", code)
    yield from K_CALL.findall(rest)
    yield from K_PASCAL.findall(rest)
    if bare_snake:
        yield from K_SNAKE.findall(_without_arguments(rest))
    for name in K_UPPER.findall(rest):
        if name.startswith("K_") or name.split("_", 1)[0] in manual.source.macro_prefixes:
            yield name


def _named(manual: Manual, page: Page) -> Iterator[tuple[int, str]]:
    """Yields the line number and the text of every repository name a page writes in code."""
    for number, span in page.spans():
        for name in _names(manual, _without_paths(span), bare_snake=True):
            yield number, name
    for fence in page.fences:
        if not fence.is_source:
            continue
        declared = fence.declared()
        for offset, line in enumerate(fence.code().splitlines()):
            for name in _names(manual, line, bare_snake=False):
                if name not in declared:
                    yield fence.first + offset, name


def _absent_parts(manual: Manual, name: str) -> list[str]:
    """Returns the segments of a name that no source file holds and no allow entry excuses."""
    if manual.is_allowed(name):
        return []
    parts = [part.lstrip("~") for part in name.split("::")]
    held = manual.source.words | K_NAMING_WORDS
    return [part for part in parts if part not in held and not manual.is_allowed(part)]


def _cited(page: Page) -> Iterator[tuple[int, str]]:
    """Yields the line number and the text of each path a page cites in code."""
    for number, span in page.spans():
        for token in span.split():
            yield number, token
    for fence in page.fences:
        for offset, line in enumerate(fence.lines):
            include = K_INCLUDE.match(line)
            if include:
                yield fence.first + offset, include.group(1)
            elif not fence.is_source:
                for token in line.split():
                    yield fence.first + offset, token


def _repository_path(manual: Manual, token: str) -> str | None:
    """Returns a token as a repository path when it reads as one, or None."""
    if K_PLACEHOLDER_TAIL.search(token):
        return None
    path = K_PATH_TRIM.sub("", K_PATH_TRIM.sub("", token)).replace("\\", "/")
    if "/" not in path or K_PLACEHOLDER.search(path) or K_OUTSIDE_PATH.match(path):
        return None
    path = path.removeprefix("./").rstrip("/")
    parts = path.split("/")
    if not all(parts) or parts[0] in K_BUILD_FOLDERS:
        return None
    known_folder = parts[0] in manual.source.top_level
    return path if known_folder or K_FILE_NAME.match(parts[-1]) else None


def rule_named_symbols(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each name a page writes in code that the source does not hold."""
    for page in manual.pages:
        if not page.describes_code:
            continue
        seen: set[tuple[int, str]] = set()
        for number, name in _named(manual, page):
            absent = _absent_parts(manual, name)
            if absent and (number, name) not in seen:
                seen.add((number, name))
                message = f"`{name}`: no source file holds {', '.join(absent)}"
                yield Issue(page.path, number, message)


def rule_cited_paths(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each cited repository path that does not exist."""
    for page in manual.pages:
        if not page.describes_code:
            continue
        beside = page.path.parent
        for number, token in _cited(page):
            path = _repository_path(manual, token)
            if path is None or path == K_ALLOW_LIST.as_posix() or manual.is_allowed(path):
                continue
            if path not in manual.source.paths and not (beside / path).exists():
                yield Issue(page.path, number, f"cited path does not exist: {path}")


def rule_example_source(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each source block that no compiled file is cited for or backs."""
    for page in manual.pages:
        if not page.describes_code or page.kind is PageKind.MODULE_README:
            continue
        for fence in page.fences:
            if not fence.is_source:
                continue
            citation = page.citation(fence)
            if citation is None:
                yield Issue(
                    page.path, fence.first - 1,
                    f"example needs 'Source: `<file>`' or 'Source: {K_NOT_COMPILED}' "
                    f"within {K_CITATION_REACH} lines after it",
                )
                continue
            number, target = citation
            if target == K_NOT_COMPILED:
                continue
            source_lines = manual.source.lines_of(target)
            if source_lines is None:
                yield Issue(page.path, number, f"example source does not exist: {target}")
                continue
            for offset, line in enumerate(fence.lines):
                text = line.strip()
                if text and text not in K_ELISIONS and text not in source_lines:
                    yield Issue(page.path, fence.first + offset, f"line is not in {target}")


def rule_history_words(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each prose line that narrates what the code was before."""
    for page in manual.pages:
        if (manual.root / K_HISTORY_EXEMPT_FOLDER) in page.path.parents:
            continue
        for number, line in page.plain():
            match = K_HISTORY.search(K_QUOTED.sub("", line))
            if match:
                yield Issue(
                    page.path, number,
                    f"'{match.group(0).lower()}' narrates history; state what is, delete what was",
                )


def rule_page_length(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each page longer than the ceiling of its kind."""
    for page in manual.pages:
        ceiling = K_LINE_CEILINGS.get(page.kind)
        if ceiling is not None and len(page.lines) > ceiling:
            yield Issue(page.path, 1, f"{len(page.lines)} lines, ceiling {ceiling}; split the page")


def rule_duplicated_passages(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each long paragraph that an earlier page or rule file already holds.

    A block the design views generate repeats its topic's About line by design, so it is left out.
    """
    homes: dict[str, Page] = {}
    for page in (*manual.pages, *manual.rule_files):
        for number, text in page.paragraphs():
            if len(text) < K_PASSAGE_FLOOR or K_GENERATED_MARK in text:
                continue
            home = homes.setdefault(text, page)
            if home is not page:
                shown = home.path.relative_to(manual.root).as_posix()
                yield Issue(page.path, number, f"paragraph repeats {shown}; link to it instead")


def rule_em_dash(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each prose line that holds an em dash."""
    for page in manual.pages:
        for number, line in page.plain():
            if "\u2014" in line and not K_STATUS_LINE.match(line):
                yield Issue(page.path, number, "em dash in prose; use a comma or a full stop")


def rule_allow_list(manual: Manual) -> Iterator[Issue]:
    """Yields an issue for each allow-list entry without a reason, a pattern or a use."""
    if not manual.allow_list:
        return
    named = {name for page in manual.pages for _, name in _named(manual, page)}
    named |= {part for name in set(named) for part in name.split("::")}
    for page in manual.pages:
        named |= {
            path for path in (_repository_path(manual, token) for _, token in _cited(page))
            if path is not None
        }
    for entry in manual.allow_list:
        if entry.pattern is None or not entry.text:
            yield Issue(manual.allow_path, entry.number, "entry is not a regular expression")
        elif not entry.reason:
            yield Issue(manual.allow_path, entry.number, "entry needs ' # reason' after it")
        elif not any(entry.allows(name) for name in named):
            yield Issue(manual.allow_path, entry.number, "entry excuses nothing; delete it")


def collect(
    paths: list[Path], source_folders: tuple[str, ...] = K_SOURCE_FOLDERS
) -> list[Manual]:
    """Returns the manual of each given folder, and of the repository holding each given page.

    A page given by path is checked whatever its name, against its repository's source.
    """
    manuals = [Manual(path, source_folders) for path in paths if path.is_dir()]
    pages: dict[Path, set[Path]] = {}
    for path in paths:
        if path.is_file():
            root = next((parent for parent in path.resolve().parents if (parent / ".git").exists()),
                        path.resolve().parent)
            pages.setdefault(root, set()).add(path.resolve())
    return manuals + [Manual(root, source_folders, frozenset(found)) for root, found in pages.items()]


K_CHECKER = Checker(
    description=(
        "Checks the manual against the source. "
        "--source FOLDER (repeatable) replaces the default source folders."
    ),
    collect=collect,
    rules={
        "rule_named_symbols": rule_named_symbols,
        "rule_cited_paths": rule_cited_paths,
        "rule_example_source": rule_example_source,
        "rule_history_words": rule_history_words,
        "rule_page_length": rule_page_length,
        "rule_duplicated_passages": rule_duplicated_passages,
        "rule_em_dash": rule_em_dash,
        "rule_allow_list": rule_allow_list,
    },
)


def main(argv: list[str]) -> int:
    """Runs the checker, reading --source itself and leaving the rest to the shared runner."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", action="append", default=[])
    own, rest = parser.parse_known_args(argv)
    manuals: list[Manual] = []

    def collect_once(paths: list[Path]) -> list[Manual]:
        """Returns the manuals of the given folders and keeps them for the closing count."""
        manuals.extend(collect(paths, tuple(own.source) or K_SOURCE_FOLDERS))
        return manuals

    code = run(replace(K_CHECKER, collect=collect_once), rest)
    if "--report" in rest and code == K_EXIT_CLEAN:
        print(f"unverified examples: {sum(manual.unverified_examples() for manual in manuals)}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

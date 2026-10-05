# write_license_block.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Writes the license block of the source-comments skill into every tracked source file.

Usage: python tools/licensing/write_license_block.py ROOT --project LINE [--closed]
       [--skip GLOB]... [--upstream PATH=LINE;LINE]... [--report]
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.check_source_comments import K_C_SUFFIXES, K_LICENSE_LINES  # noqa: E402

K_CLOSED_LINES = ("All rights reserved. No licence is granted.",)
K_HOLDER = "Sushi Systems"
K_HASH_SUFFIXES = frozenset({".py", ".cmake", ".sh"})
K_HASH_NAMES = frozenset({"CMakeLists.txt"})
K_FRAGMENT_SUFFIXES = frozenset({".inc"})
K_TEMPLATE_SUFFIX = ".in"
K_SKIPPED_FOLDERS = frozenset({"third_party", "node_modules", "build"})
K_BOX_FIELD = 60
K_BYTE_ORDER_MARK = "﻿"
K_BOXED_ROW = re.compile(r"^\s*/\*.*\*/\s*$")
K_BOX_EDGE = re.compile(r"^\s*/\*{5,}/\s*$")
K_HASH_PREAMBLE = re.compile(r"^#(!|.*coding[:=])")
K_HASH_MARKER = re.compile(r"Copyright|Licensed under|All rights reserved")
K_DOCSTRING = re.compile(r"^[rRuUbB]{0,2}(\"\"\"|''')")
K_EXIT_CLEAN = 0
K_EXIT_PENDING = 1
K_EXIT_USAGE = 2


class LicenseWriterError(Exception):
    """Signals a repository or an argument the writer cannot work with."""


@dataclass(frozen=True, slots=True)
class Header:
    """Holds what a repository's license blocks say apart from file name and year."""

    project: str
    license_lines: tuple[str, ...]
    upstream: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def rows(self, path: str, year: int, kept: tuple[str, ...] = ()) -> list[str]:
        """Returns the block's rows for a file, with the upstream rows named for it or kept."""
        name = PurePosixPath(path).name
        own = [name, self.project, f"Copyright (c) {year} {K_HOLDER}", *self.license_lines]
        return own + list(self.upstream.get(path, kept))

    def kept_rows(self, old: list[str]) -> tuple[str, ...]:
        """Returns the rows an old block carries below this repository's own licence lines."""
        if not all(line in old for line in self.license_lines):
            return ()
        last = max(old.index(line) for line in self.license_lines)
        return tuple(old[last + 1:])


def _generated_name(path: str) -> PurePosixPath:
    """Returns a path's file name with the configure-template suffix removed."""
    pure = PurePosixPath(path)
    return PurePosixPath(pure.stem) if pure.suffix == K_TEMPLATE_SUFFIX else PurePosixPath(pure.name)


def is_hash_family(path: str) -> bool:
    """Returns whether a file takes the `#` form of the block."""
    name = _generated_name(path)
    return name.suffix in K_HASH_SUFFIXES or name.name in K_HASH_NAMES


def is_source(path: str) -> bool:
    """Returns whether the writer gives a file a license block."""
    suffix = _generated_name(path).suffix
    return is_hash_family(path) or suffix in K_C_SUFFIXES or suffix in K_FRAGMENT_SUFFIXES


def _render_boxed(rows: list[str]) -> list[str]:
    """Returns the rows inside a box whose edges are as wide as its rows."""
    width = max(K_BOX_FIELD, *(len(row) for row in rows))
    edge = "/" + "*" * (width + 4) + "/"
    return [edge, *(f"/* {row.ljust(width)} */" for row in rows), edge]


def _render_hashed(rows: list[str]) -> list[str]:
    """Returns the rows as `#` comment lines."""
    return [f"# {row}" for row in rows]


def _texts(block: list[str], marker: str) -> list[str]:
    """Returns the rows of an old block without comment markers or box edges."""
    texts = [line.strip().removeprefix(marker).removesuffix("*/").strip() for line in block]
    return [text for text in texts if text and set(text) != {"*"}]


def _split_boxed(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Returns an empty preamble, the rows of an old C-family block and the lines below it."""
    for pattern, marker in ((K_BOXED_ROW, "/*"), (re.compile(r"^//"), "//")):
        end = 0
        while end < len(lines) and pattern.match(lines[end]):
            end += 1
        run = lines[:end]
        is_box = marker == "/*" and bool(run) and bool(K_BOX_EDGE.match(run[0]))
        if run and (is_box or any("Copyright" in line for line in run)):
            return [], _texts(run, marker), lines[end:]
    return [], [], lines


def _split_hashed(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Returns the shebang and encoding lines, the rows of an old block and the lines below it."""
    start = 0
    while start < len(lines) and K_HASH_PREAMBLE.match(lines[start]):
        start += 1
    end = start
    while end < len(lines) and lines[end].startswith("#"):
        end += 1
    if any(K_HASH_MARKER.search(line) for line in lines[start:end]):
        return lines[:start], _texts(lines[start:end], "#"), lines[end:]
    return lines[:start], [], lines[start:]


def rewrite(path: str, text: str, header: Header, year: int) -> str:
    """Returns a file's text with its license block replaced or inserted."""
    mark = K_BYTE_ORDER_MARK if text.startswith(K_BYTE_ORDER_MARK) else ""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text[len(mark):].replace("\r\n", "\n").split("\n")
    hashed = is_hash_family(path)
    preamble, old, rest = _split_hashed(lines) if hashed else _split_boxed(lines)
    while rest and not rest[0].strip():
        rest = rest[1:]
    rows = header.rows(path, year, header.kept_rows(old))
    block = _render_hashed(rows) if hashed else _render_boxed(rows)
    joined = hashed and bool(rest) and bool(K_DOCSTRING.match(rest[0]))
    gap = [] if joined or not rest else [""]
    body = rest if rest else [""]
    return mark + newline.join([*preamble, *block, *gap, *body])


def _git(root: Path, *arguments: str) -> str:
    """Returns the output of a git command run in a repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "-c", "core.quotepath=off", *arguments],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LicenseWriterError(f"git {arguments[0]} failed in {root}") from error
    return completed.stdout


def first_years(root: Path) -> dict[str, int]:
    """Returns the year each committed file was first added, following renames."""
    current = {path: path for path in _git(root, "ls-files").splitlines()}
    years: dict[str, int] = {}
    year = 0
    log = _git(root, "log", "-M", "--diff-filter=AR", "--name-status", "--format=@%ad", "--date=format:%Y")
    for line in log.splitlines():
        if line.startswith("@"):
            year = int(line[1:])
            continue
        parts = line.split("\t")
        if parts[0] == "A" and parts[1] in current:
            years[current.pop(parts[1])] = year
        elif parts[0].startswith("R") and parts[2] in current:
            current[parts[1]] = current.pop(parts[2])
    return years


def _is_skipped(path: str, globs: list[str]) -> bool:
    """Returns whether a tracked path lies in a skipped folder or matches a skip glob."""
    parts = PurePosixPath(path).parts
    return bool(K_SKIPPED_FOLDERS.intersection(parts)) or any(fnmatch.fnmatch(path, glob) for glob in globs)


def pending(root: Path, header: Header, globs: list[str]) -> Iterator[tuple[Path, str]]:
    """Yields each source file whose text differs from what the writer produces, with that text."""
    years = first_years(root)
    this_year = date.today().year
    for path in sorted(_git(root, "ls-files").splitlines()):
        if not is_source(path) or _is_skipped(path, globs):
            continue
        target = root / path
        if not target.is_file():
            continue
        text = target.read_bytes().decode("utf-8", errors="surrogateescape")
        wanted = rewrite(path, text, header, years.get(path, this_year))
        if wanted != text:
            yield target, wanted


def _parse_upstream(entries: list[str]) -> dict[str, tuple[str, ...]]:
    """Returns the upstream notice lines per path from `PATH=LINE;LINE` arguments."""
    upstream: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        path, separator, lines = entry.partition("=")
        if not separator or not lines:
            raise LicenseWriterError(f"--upstream takes PATH=LINE;LINE, got: {entry}")
        upstream[path] = tuple(line.strip() for line in lines.split(";"))
    return upstream


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description="Writes the license block into tracked source files.")
    parser.add_argument("root", type=Path, help="repository root")
    parser.add_argument("--project", required=True, help="the block's second line: name and URL")
    parser.add_argument("--closed", action="store_true", help="write the reserved-rights line")
    parser.add_argument("--skip", action="append", default=[], help="glob of tracked paths to leave alone")
    parser.add_argument("--upstream", action="append", default=[], help="PATH=LINE;LINE of a ported file")
    parser.add_argument("--report", action="store_true", help="write nothing; list pending files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the writer from the command line and returns its exit code."""
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    lines = K_CLOSED_LINES if arguments.closed else K_LICENSE_LINES
    try:
        header = Header(arguments.project, lines, _parse_upstream(arguments.upstream))
        changes = list(pending(arguments.root, header, arguments.skip))
    except LicenseWriterError as error:
        print(error, file=sys.stderr)
        return K_EXIT_USAGE
    for target, wanted in changes:
        if arguments.report:
            print(target.relative_to(arguments.root).as_posix())
        else:
            target.write_bytes(wanted.encode("utf-8", errors="surrogateescape"))
    print(f"{len(changes)} files {'pending' if arguments.report else 'written'}", file=sys.stderr)
    return K_EXIT_PENDING if arguments.report and changes else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

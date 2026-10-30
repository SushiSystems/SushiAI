# copy_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Copies the shared tools and community files of SushiSkills into a repository, keeping its settings and its own tools.

Usage: python tools/checking/copy_shared_tools.py ROOT --from UPSTREAM [--report] [--force] [--keep PATH]

A shared file that holds text no version of upstream ever held is a local edit. A copy refuses to
overwrite it and writes nothing; --force overwrites it anyway, and --keep leaves a named file alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

K_TOOLS = "tools"
K_COMMUNITY = (".github/CODE_OF_CONDUCT.md", ".github/SECURITY.md")
K_KEPT = frozenset({"tools/settings.py", "tools/README.md"})
K_HEADER_TEMPLATE = "tools/check_all.py"
K_HEADER_SUFFIX = ".py"
K_EXIT_CLEAN = 0
K_EXIT_PENDING = 1
K_EXIT_USAGE = 2


class LocalEditsError(Exception):
    """Signals shared files that hold text no upstream version ever held."""

    def __init__(self, files: list[str]) -> None:
        super().__init__(", ".join(files))
        self.files = files


def _split_header(text: str) -> tuple[str, str]:
    """Returns the leading run of comment lines of a Python file and the rest."""
    lines = text.splitlines(keepends=True)
    count = 0
    while count < len(lines) and lines[count].startswith("#"):
        count += 1
    return "".join(lines[:count]), "".join(lines[count:])


def merged(relative: str, upstream: str, current: str | None, template: str | None = None) -> str:
    """Returns a shared file as the repository should hold it: upstream's body under its own header.

    A file the repository does not hold yet takes the header of *template*, one of the
    repository's own shared files, with its first line naming the new file.
    """
    if not relative.endswith(K_HEADER_SUFFIX):
        return upstream
    if current is not None:
        return _split_header(current)[0] + _split_header(upstream)[1]
    if template is None:
        return upstream
    header = _split_header(template)[0].splitlines(keepends=True)
    if not header:
        return upstream
    header[0] = f"# {Path(relative).name}\n"
    return "".join(header) + _split_header(upstream)[1]


def _read(path: Path) -> str | None:
    """Returns a file's text, or None when it does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _wanted(upstream: Path, root: Path, relative: str) -> str | None:
    """Returns the text a shared file should hold in the repository, or None when it should stay."""
    current = _read(root / relative)
    if relative in K_KEPT and current is not None:
        return None
    text = merged(relative, _read(upstream / relative) or "", current, _read(root / K_HEADER_TEMPLATE))
    return None if text == current else text


def _body(relative: str, text: str) -> str:
    """Returns what a shared file says apart from its license header and its newline style."""
    body = _split_header(text)[1] if relative.endswith(K_HEADER_SUFFIX) else text
    return body.replace("\r\n", "\n")


def known_bodies(upstream: Path, relative: str) -> set[str]:
    """Returns the body of a shared file in upstream now and in every commit that changed it."""
    bodies = {_body(relative, _read(upstream / relative) or "")}
    try:
        history = subprocess.run(["git", "-C", str(upstream), "log", "--format=%H", "--", relative],
                                 capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return bodies
    for revision in history.stdout.split() if history.returncode == 0 else []:
        shown = subprocess.run(["git", "-C", str(upstream), "show", f"{revision}:{relative}"],
                               capture_output=True, text=True, encoding="utf-8")
        if shown.returncode == 0:
            bodies.add(_body(relative, shown.stdout))
    return bodies


def local_edits(upstream: Path, root: Path, shared: Iterable[str]) -> list[str]:
    """Returns the pending shared files whose current text no upstream version ever held."""
    edited = []
    for relative in pending(upstream, root, shared):
        current = _read(root / relative)
        if current is not None and _body(relative, current) not in known_bodies(upstream, relative):
            edited.append(relative)
    return edited


def pending(upstream: Path, root: Path, shared: Iterable[str]) -> list[str]:
    """Returns the shared files a copy would write, in order."""
    return [relative for relative in sorted(shared) if _wanted(upstream, root, relative) is not None]


def copy(upstream: Path, root: Path, shared: Iterable[str], force: bool = False) -> list[str]:
    """Writes every pending shared file and returns them; no other file is touched.

    Raises LocalEditsError, with nothing written, when a pending file is a local edit and force is off.
    """
    edited = [] if force else local_edits(upstream, root, shared)
    if edited:
        raise LocalEditsError(edited)
    written = []
    for relative in sorted(shared):
        text = _wanted(upstream, root, relative)
        if text is None:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written.append(relative)
    return written


def shared_files(upstream: Path) -> list[str]:
    """Returns the files git tracks under the upstream tools folder, and the shared community files."""
    listed = subprocess.run(["git", "-C", str(upstream), "ls-files", K_TOOLS, *K_COMMUNITY],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return [line for line in listed.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    """Copies the shared tools, or lists the files a copy would write with --report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--from", dest="upstream", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite a shared file that holds a local edit")
    parser.add_argument("--keep", action="append", default=[], metavar="PATH",
                        help="leave this shared file as the repository holds it; repeatable")
    arguments = parser.parse_args(argv)
    if not (arguments.upstream / K_TOOLS).is_dir() or not arguments.root.is_dir():
        print("usage: ROOT and UPSTREAM must be repositories with a tools folder", file=sys.stderr)
        return K_EXIT_USAGE
    shared = [relative for relative in shared_files(arguments.upstream) if relative not in arguments.keep]
    if arguments.report:
        files = pending(arguments.upstream, arguments.root, shared)
        edited = set(local_edits(arguments.upstream, arguments.root, shared))
        print("\n".join(f"{relative}: differs from upstream" + (" (local edit)" if relative in edited else "")
                        for relative in files) or "clean")
        return K_EXIT_PENDING if files else K_EXIT_CLEAN
    try:
        written = copy(arguments.upstream, arguments.root, shared, arguments.force)
    except LocalEditsError as error:
        for relative in error.files:
            print(f"{relative}: a local edit; move it upstream or pass --force", file=sys.stderr)
        return K_EXIT_PENDING
    for relative in written:
        print(f"{relative}: written")
    return K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

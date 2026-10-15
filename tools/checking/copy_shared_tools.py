# copy_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Copies the shared tools of SushiSkills into a repository, keeping its settings and its own tools.

Usage: python tools/checking/copy_shared_tools.py ROOT --from UPSTREAM [--report]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

K_TOOLS = "tools"
K_SETTINGS = "tools/settings.py"
K_HEADER_SUFFIX = ".py"
K_EXIT_CLEAN = 0
K_EXIT_PENDING = 1
K_EXIT_USAGE = 2


def _split_header(text: str) -> tuple[str, str]:
    """Returns the leading run of comment lines of a Python file and the rest."""
    lines = text.splitlines(keepends=True)
    count = 0
    while count < len(lines) and lines[count].startswith("#"):
        count += 1
    return "".join(lines[:count]), "".join(lines[count:])


def merged(relative: str, upstream: str, current: str | None) -> str:
    """Returns a shared file as the repository should hold it: upstream's body under its own header."""
    if current is None or not relative.endswith(K_HEADER_SUFFIX):
        return upstream
    return _split_header(current)[0] + _split_header(upstream)[1]


def _read(path: Path) -> str | None:
    """Returns a file's text, or None when it does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _wanted(upstream: Path, root: Path, relative: str) -> str | None:
    """Returns the text a shared file should hold in the repository, or None when it should stay."""
    current = _read(root / relative)
    if relative == K_SETTINGS and current is not None:
        return None
    text = merged(relative, _read(upstream / relative) or "", current)
    return None if text == current else text


def pending(upstream: Path, root: Path, shared: Iterable[str]) -> list[str]:
    """Returns the shared files a copy would write, in order."""
    return [relative for relative in sorted(shared) if _wanted(upstream, root, relative) is not None]


def copy(upstream: Path, root: Path, shared: Iterable[str]) -> list[str]:
    """Writes every pending shared file and returns them; no other file is touched."""
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
    """Returns the files git tracks under the upstream tools folder."""
    listed = subprocess.run(["git", "-C", str(upstream), "ls-files", K_TOOLS],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return [line for line in listed.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    """Copies the shared tools, or lists the files a copy would write with --report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--from", dest="upstream", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    arguments = parser.parse_args(argv)
    if not (arguments.upstream / K_TOOLS).is_dir() or not arguments.root.is_dir():
        print("usage: ROOT and UPSTREAM must be repositories with a tools folder", file=sys.stderr)
        return K_EXIT_USAGE
    shared = shared_files(arguments.upstream)
    if arguments.report:
        files = pending(arguments.upstream, arguments.root, shared)
        print("\n".join(f"{relative}: differs from upstream" for relative in files) or "clean")
        return K_EXIT_PENDING if files else K_EXIT_CLEAN
    for relative in copy(arguments.upstream, arguments.root, shared):
        print(f"{relative}: written")
    return K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

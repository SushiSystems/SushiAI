# copy_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Copies the shared tools and community files of SushiSkills into a repository, keeping its settings and its own tools.

Usage: python tools/checking/copy_shared_tools.py ROOT --from UPSTREAM [--report] [--force] [--keep PATH]

Upstream is read at HEAD. A copy refuses a local edit (text no upstream version held) unless --force; --keep
leaves a file alone; a file upstream dropped is named, never removed.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.licence import Licence, from_module  # noqa: E402
from licensing.write_license_block import Header, is_source, rewrite  # noqa: E402

K_TOOLS = "tools"
K_COMMUNITY = (".github/CODE_OF_CONDUCT.md", ".github/SECURITY.md")
K_KEPT = frozenset({"tools/settings.py", "tools/README.md"})
K_HEADER_TEMPLATE = "tools/check_all.py"
K_HEADER_SUFFIX = ".py"
K_REVISION = "HEAD"
K_SETTINGS = "tools/settings.py"
K_PROJECT_ROW = 1
K_YEAR = re.compile(r"Copyright \(c\) (\d{4})-present")
K_NEUTRAL = Header("", Licence("", "", "", (), (), (), (), False))
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


def _is_other_source(relative: str) -> bool:
    """Returns whether a file takes a license block of the writer but is not Python."""
    return is_source(relative) and not relative.endswith(K_HEADER_SUFFIX)


def first_year(this_year: int, *texts: str | None) -> int:
    """Returns the year of the first copyright row found in the texts, or this_year."""
    for text in texts:
        found = K_YEAR.search(text or "")
        if found:
            return int(found.group(1))
    return this_year


def merged(
    relative: str,
    upstream: str,
    current: str | None,
    template: str | None = None,
    header: Header | None = None,
    year: int = 0,
) -> str:
    """Returns a shared file as the repository should hold it: upstream's body under its own header.

    A Python file the repository does not hold yet takes the header of *template*, one of the
    repository's own shared files, with its first line naming the new file. Any other source file
    takes the block *header* builds for the repository, dated *year*.
    """
    if _is_other_source(relative):
        return upstream if header is None else rewrite(relative, upstream, header, year)
    if not relative.endswith(K_HEADER_SUFFIX):
        return upstream
    if current is not None:
        return _split_header(current)[0] + _split_header(upstream)[1]
    if template is None:
        return upstream
    rows = _split_header(template)[0].splitlines(keepends=True)
    if not rows:
        return upstream
    rows[0] = f"# {Path(relative).name}\n"
    return "".join(rows) + _split_header(upstream)[1]


def _read(path: Path) -> str | None:
    """Returns a file's text, or None when it does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def committed(upstream: Path, relative: str) -> str | None:
    """Returns a file as upstream committed it at HEAD, or None when HEAD does not hold it."""
    try:
        shown = subprocess.run(
            ["git", "-C", str(upstream), "show", f"{K_REVISION}:{relative}"],
            capture_output=True,
        )
    except OSError:
        return None
    if shown.returncode != 0:
        return None
    return shown.stdout.decode("utf-8").replace("\r\n", "\n")


def uncommitted(upstream: Path) -> list[str]:
    """Returns the paths under upstream's tools folder whose working tree differs from HEAD."""
    status = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain", "--", K_TOOLS],
        capture_output=True, text=True, encoding="utf-8",
    )
    return [line[3:] for line in status.stdout.splitlines()] if status.returncode == 0 else []


def uncommitted_notice(paths: list[str]) -> str | None:
    """Returns the line that names uncommitted upstream files, or None when there are none."""
    if not paths:
        return None
    return f"upstream has uncommitted changes, copying HEAD: {', '.join(paths)}"


def target_header(root: Path) -> Header | None:
    """Returns the block the repository's own settings and template header describe, or None."""
    settings_path = root / K_SETTINGS
    template = _read(root / K_HEADER_TEMPLATE)
    if template is None or not settings_path.is_file():
        return None
    rows = _split_header(template)[0].splitlines()
    if len(rows) <= K_PROJECT_ROW:
        return None
    spec = importlib.util.spec_from_file_location("target_settings", settings_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    licence = from_module(module)
    if licence is None:
        return None
    return Header(rows[K_PROJECT_ROW].removeprefix("#").strip(), licence, closed=licence.closed)


def _wanted(upstream: Path, root: Path, relative: str) -> str | None:
    """Returns the text a shared file should hold in the repository, or None when it should stay."""
    current = _read(root / relative)
    if relative in K_KEPT and current is not None:
        return None
    text = committed(upstream, relative) or ""
    year = first_year(date.today().year, current, text)
    template = _read(root / K_HEADER_TEMPLATE)
    wanted = merged(relative, text, current, template, target_header(root), year)
    return None if wanted == current else wanted


def _body(relative: str, text: str) -> str:
    """Returns what a shared file says apart from its license block and its newline style."""
    if relative.endswith(K_HEADER_SUFFIX):
        body = _split_header(text)[1]
    elif is_source(relative):
        body = rewrite(relative, text, K_NEUTRAL, 0)
    else:
        body = text
    return body.replace("\r\n", "\n")


def known_bodies(upstream: Path, relative: str) -> set[str]:
    """Returns the body of a shared file in upstream now and in every commit that changed it."""
    bodies = {_body(relative, committed(upstream, relative) or "")}
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
    """Returns the files upstream committed under its tools folder, and the shared community files."""
    listed = subprocess.run(
        ["git", "-C", str(upstream), "ls-tree", "-r", "--name-only", K_REVISION, "--", K_TOOLS, *K_COMMUNITY],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return [line for line in listed.stdout.splitlines() if line]


def ever_shared(upstream: Path) -> list[str]:
    """Returns every path any upstream commit held under its tools folder or among the community files."""
    listed = subprocess.run(
        ["git", "-C", str(upstream), "log", "--format=", "--name-only", K_REVISION, "--", K_TOOLS, *K_COMMUNITY],
        capture_output=True, text=True, encoding="utf-8",
    )
    return sorted({line for line in listed.stdout.splitlines() if line}) if listed.returncode == 0 else []


def held_files(root: Path) -> list[str]:
    """Returns the files the repository tracks under its tools folder and among the community files."""
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", K_TOOLS, *K_COMMUNITY],
        capture_output=True, text=True, encoding="utf-8",
    )
    return [line for line in listed.stdout.splitlines() if line] if listed.returncode == 0 else []


def dropped(held: Iterable[str], shared: Iterable[str], once_shared: Iterable[str]) -> list[str]:
    """Returns the held files upstream once shared and no longer does, in order."""
    return sorted((set(held) & set(once_shared)) - set(shared))


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
    notice = uncommitted_notice(uncommitted(arguments.upstream))
    if notice:
        print(notice)
    shared = [relative for relative in shared_files(arguments.upstream) if relative not in arguments.keep]
    for relative in dropped(held_files(arguments.root), shared_files(arguments.upstream),
                            ever_shared(arguments.upstream)):
        print(f"{relative}: dropped upstream; remove it")
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

# check_module_install.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks a module against the module-install skill; a repository that is not a module is skipped.

Usage: python tools/documentation/check_module_install.py [repository root] [--report] [--rule NAME]

A repository is a module when it holds the manifest K_MODULE_MANIFEST names in tools/settings.py,
or a CLI package; one whose settings name no manifest is not checked.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402

K_DEFAULT_INSTALL_PAGE = "docs/getting_started/INSTALL.md"
K_CONFIG_SUFFIXES = frozenset({".json", ".toml", ".cmake", ".txt", ".py", ".yml", ".yaml", ".in"})
K_CODE_SUFFIXES = frozenset({".cmake", ".txt", ".py", ".yml", ".yaml", ".in"})
K_SKIPPED_FOLDERS = frozenset({"docs", "third_party", "build", ".superpowers", "tests", "tools"})
K_MACHINE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\][A-Za-z]")
K_COMMENT = re.compile(r"^\s*(?:#|//)")


@dataclass(frozen=True, slots=True)
class ModuleSettings:
    """Holds the module-install settings a module is checked against, as the caller gives them."""

    manifest: str
    fragment: str
    install_page: str
    headings: tuple[str, ...]
    workspace_read: str


@dataclass(frozen=True, slots=True)
class Module:
    """Holds one module as values: its root, its settings, what exists and the texts the rules read.

    ``present`` holds the element paths, below the root, that are files on disk. ``install_page``
    is the install page's text when it is a file; ``sources`` holds the committed config files
    outside the skipped folders, with their text.
    """

    root: Path
    config: ModuleSettings
    present: frozenset[str]
    install_page: SourceFile | None
    sources: tuple[SourceFile, ...]


def _below_skipped(root: Path, path: Path) -> bool:
    """Returns whether a file sits in a folder the machine-path and workspace rules do not read."""
    parts = path.relative_to(root).parts
    return parts[0] in K_SKIPPED_FOLDERS or "tests" in parts


def _docstring_lines(text: str) -> set[int]:
    """Returns the line numbers a Python source holds inside docstrings; none when it does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def rule_elements(module: Module) -> Iterator[Issue]:
    """Yields an issue for each element of a module that is missing."""
    elements = (module.config.manifest, module.config.fragment, module.config.install_page)
    for element in filter(None, elements):
        if element not in module.present:
            yield Issue(module.root / element, 1, "the module-install elements list this file; it is missing")


def rule_install_headings(module: Module) -> Iterator[Issue]:
    """Yields an issue for each heading the install page lacks."""
    page = module.install_page
    if page is None:
        return
    headings = {line.strip() for line in page.text.splitlines()}
    for heading in module.config.headings:
        if heading not in headings:
            yield Issue(page.path, 1, f"the install page needs the heading '{heading}'")


def rule_machine_path(module: Module) -> Iterator[Issue]:
    """Yields an issue for each committed config or code line that holds the path of one machine."""
    for source in module.sources:
        if source.path.suffix not in K_CONFIG_SUFFIXES or _below_skipped(module.root, source.path):
            continue
        for number, line in enumerate(source.text.splitlines(), 1):
            if K_MACHINE_PATH.search(line) and not K_COMMENT.match(line):
                yield Issue(source.path, number, "a machine path in a committed file; take it from the CLI or an environment variable")


def rule_no_workspace_in_code(module: Module) -> Iterator[Issue]:
    """Yields an issue for each build, CLI or workflow line that reads the workspace tool's state."""
    if not module.config.workspace_read:
        return
    workspace_read = re.compile(module.config.workspace_read)
    for source in module.sources:
        if source.path.suffix not in K_CODE_SUFFIXES or _below_skipped(module.root, source.path):
            continue
        documented = _docstring_lines(source.text) if source.path.suffix == ".py" else set()
        for number, line in enumerate(source.text.splitlines(), 1):
            if workspace_read.search(line) and not K_COMMENT.match(line) and number not in documented:
                yield Issue(source.path, number, "code and build files do not read the workspace tool; the CLI resolves the workspace")


K_SETTINGS = ModuleSettings(
    manifest=getattr(settings, "K_MODULE_MANIFEST", ""),
    fragment=getattr(settings, "K_MODULE_DEPENDENCY_FILE", ""),
    install_page=getattr(settings, "K_MODULE_INSTALL_PAGE", K_DEFAULT_INSTALL_PAGE),
    headings=tuple(getattr(settings, "K_MODULE_HEADINGS", ("## Standalone",))),
    workspace_read=getattr(settings, "K_MODULE_WORKSPACE_READ", ""),
)


def _tracked_paths(root: Path) -> list[Path]:
    """Returns the files git tracks below root, or every file when root is not a repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, encoding="utf-8")
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0 and completed.stdout:
        return [root / name for name in completed.stdout.split("\0") if name and (root / name).is_file()]
    return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def _read_module(root: Path, config: ModuleSettings) -> Module:
    """Reads one module's elements, install page and config files into values."""
    elements = [name for name in (config.manifest, config.fragment, config.install_page) if name]
    present = frozenset(name for name in elements if (root / name).is_file())
    pages = read_files([root / config.install_page], lambda _: True) if config.install_page in present else ()
    sources = read_files(
        (path for path in _tracked_paths(root)
         if path.suffix in K_CONFIG_SUFFIXES and not _below_skipped(root, path)),
        lambda _: True,
    )
    return Module(root, config, present, pages[0] if pages else None, sources)


def read_modules(paths: list[Path], config: ModuleSettings) -> list[Module]:
    """Returns each given root that is a module under the settings; none when the settings name no manifest."""
    modules = []
    for path in paths if config.manifest else []:
        root = path if path.is_dir() else path.parent
        if (root / config.manifest).is_file() or (root / "cli/pyproject.toml").is_file():
            modules.append(_read_module(root, config))
    return modules


def collect(paths: list[Path]) -> list[Module]:
    """Returns each given root that is a module; none when the settings name no manifest."""
    return read_modules(paths, K_SETTINGS)


K_CHECKER = Checker(
    description="Checks a module against the module-install skill.",
    collect=collect,
    rules={
        "rule_elements": rule_elements,
        "rule_install_headings": rule_install_headings,
        "rule_machine_path": rule_machine_path,
        "rule_no_workspace_in_code": rule_no_workspace_in_code,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

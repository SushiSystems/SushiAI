# check_module_install.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks a Sushi module against the module-install skill; any other repository is skipped.

Usage: python tools/documentation/check_module_install.py [repository root] [--report] [--rule NAME]
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

from checking.checker import Checker, Issue, run  # noqa: E402

K_MANIFEST = Path("sushi-module.toml")
K_FRAGMENT = Path("cli/sushistack.deps.toml")
K_INSTALL_PAGE = Path("docs/getting_started/INSTALL.md")
K_HEADINGS = ("## Standalone", "## With SushiHub")
K_CONFIG_SUFFIXES = frozenset({".json", ".toml", ".cmake", ".txt", ".py", ".yml", ".yaml", ".in"})
K_CODE_SUFFIXES = frozenset({".cmake", ".txt", ".py", ".yml", ".yaml", ".in"})
K_SKIPPED_FOLDERS = frozenset({"docs", "third_party", "build", ".superpowers", "tests", "tools"})
K_MACHINE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\](?:Projects\b|Users[/\\][A-Za-z])")
K_HUB_READ = re.compile(r"\bhub (?:install|add|link|doctor|status|pin)\b|SUSHISTACK_HOME|\.sushistack\b|workspace\.toml")
K_COMMENT = re.compile(r"^\s*(?:#|//)")


@dataclass(frozen=True, slots=True)
class Module:
    """Holds the root of one module and the files it commits."""

    root: Path
    files: tuple[Path, ...]


def _tracked(root: Path) -> list[Path]:
    """Returns the files git tracks below root, or every file when root is not a repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, encoding="utf-8")
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0 and completed.stdout:
        return [root / name for name in completed.stdout.split("\0") if name and (root / name).is_file()]
    return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def collect(paths: list[Path]) -> list[Module]:
    """Returns each given root that is a Sushi module."""
    modules = []
    for path in paths:
        root = path if path.is_dir() else path.parent
        if (root / K_MANIFEST).is_file() or (root / "cli/pyproject.toml").is_file():
            modules.append(Module(root, tuple(_tracked(root))))
    return modules


def _below_skipped(module: Module, path: Path) -> bool:
    """Returns whether a file sits in a folder the machine-path and hub rules do not read."""
    parts = path.relative_to(module.root).parts
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
    for element in (K_MANIFEST, K_FRAGMENT, K_INSTALL_PAGE):
        if not (module.root / element).is_file():
            yield Issue(module.root / element, 1, "the module-install elements list this file; it is missing")


def rule_install_headings(module: Module) -> Iterator[Issue]:
    """Yields an issue for each heading the install page lacks."""
    page = module.root / K_INSTALL_PAGE
    if not page.is_file():
        return
    headings = {line.strip() for line in page.read_text(encoding="utf-8").splitlines()}
    for heading in K_HEADINGS:
        if heading not in headings:
            yield Issue(page, 1, f"the install page needs the heading '{heading}'")


def rule_machine_path(module: Module) -> Iterator[Issue]:
    """Yields an issue for each committed config or code line that holds the path of one machine."""
    for path in module.files:
        if path.suffix not in K_CONFIG_SUFFIXES or _below_skipped(module, path):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if K_MACHINE_PATH.search(line) and not K_COMMENT.match(line):
                yield Issue(path, number, "a machine path in a committed file; take it from the CLI or an environment variable")


def rule_no_hub_in_code(module: Module) -> Iterator[Issue]:
    """Yields an issue for each build, CLI or workflow line that reads the SushiHub workspace."""
    for path in module.files:
        if path.suffix not in K_CODE_SUFFIXES or _below_skipped(module, path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        documented = _docstring_lines(text) if path.suffix == ".py" else set()
        for number, line in enumerate(text.splitlines(), 1):
            if K_HUB_READ.search(line) and not K_COMMENT.match(line) and number not in documented:
                yield Issue(path, number, "code and build files do not read SushiHub; the CLI resolves the workspace")


K_CHECKER = Checker(
    description="Checks a Sushi module against the module-install skill.",
    collect=collect,
    rules={
        "rule_elements": rule_elements,
        "rule_install_headings": rule_install_headings,
        "rule_machine_path": rule_machine_path,
        "rule_no_hub_in_code": rule_no_hub_in_code,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

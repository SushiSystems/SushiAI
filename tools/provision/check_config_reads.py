# check_config_reads.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks that every `cfg.<name>` the provision package reads is declared by `ProvisionConfig`.

Usage: python tools/provision/check_config_reads.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_PROVISION = Path("source/sushicore/provision")
K_CONFIG_FILE = K_PROVISION / "config.py"
K_PROTOCOL = "ProvisionConfig"
K_CFG_ATTRIBUTE = re.compile(r"\bcfg\.([a-z_]+)")


@dataclass(frozen=True, slots=True)
class Scope:
    """Holds the names the config protocol declares and the provision files with their text."""

    declared: frozenset[str]
    files: tuple[tuple[Path, str], ...]


def declared_names(config_text: str) -> frozenset[str]:
    """Returns every field, property and method name the `ProvisionConfig` class declares."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(config_text)):
        if isinstance(node, ast.ClassDef) and node.name == K_PROTOCOL:
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    names.add(member.target.id)
                elif isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(member.name)
    return frozenset(name for name in names if not name.startswith("_"))


def _collect(paths: list[Path]) -> Iterator[Scope]:
    """Yields the scope of each repository root given."""
    for root in paths:
        folder = root if root.is_dir() else root.parent
        config = folder / K_CONFIG_FILE
        if not config.is_file():
            continue
        files = tuple((path, path.read_text(encoding="utf-8"))
                      for path in sorted((folder / K_PROVISION).rglob("*.py")))
        yield Scope(declared_names(config.read_text(encoding="utf-8")), files)


def rule_undeclared_read(scope: Scope) -> Iterator[Issue]:
    """Yields an issue for each `cfg.<name>` read of a name `ProvisionConfig` does not declare."""
    for path, text in scope.files:
        for number, line in enumerate(text.splitlines(), 1):
            for name in K_CFG_ATTRIBUTE.findall(line):
                if name not in scope.declared:
                    yield Issue(path, number, f"cfg.{name} is read but ProvisionConfig declares no {name}")


K_CHECKER = Checker(
    description="Checks that provision reads only what ProvisionConfig declares.",
    collect=_collect,
    rules={"undeclared-read": rule_undeclared_read},
)


if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

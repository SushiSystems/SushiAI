# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Runs a table of rules over a set of subjects and reports their findings.

Every checker under tools/ builds its subjects, names its rules and hands both to run().
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1
K_EXIT_USAGE = 2

Subject = TypeVar("Subject")


@dataclass(frozen=True, slots=True)
class Issue:
    """Holds one problem a rule found, before the runner names the rule."""

    path: Path
    line: int
    message: str


@dataclass(frozen=True, slots=True)
class Finding:
    """Holds one reported problem with the rule that found it."""

    path: Path
    line: int
    rule: str
    message: str

    def format(self, root: Path) -> str:
        """Returns the finding as `path:line: rule: message`, the path relative to root."""
        try:
            shown = self.path.resolve().relative_to(root.resolve())
        except ValueError:
            shown = self.path
        return f"{shown.as_posix()}:{self.line}: {self.rule}: {self.message}"


Rule = Callable[[Subject], Iterator[Issue]]


@dataclass(frozen=True, slots=True)
class Checker(Generic[Subject]):
    """Describes one checker: its purpose, how it finds subjects and its rule table."""

    description: str
    collect: Callable[[list[Path]], Iterable[Subject]]
    rules: Mapping[str, Rule]


def _parse_arguments(checker: Checker[Subject], argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line for a checker."""
    parser = argparse.ArgumentParser(description=checker.description)
    parser.add_argument("paths", nargs="*", type=Path, help="files or folders; default: .")
    parser.add_argument("--report", action="store_true", help="print counts per rule, exit 0")
    parser.add_argument("--rule", choices=sorted(checker.rules), help="run one rule only")
    return parser.parse_args(argv)


def _collect_findings(
    checker: Checker[Subject], paths: list[Path], rule_names: list[str]
) -> list[Finding]:
    """Returns every finding of the selected rules over every subject."""
    findings: list[Finding] = []
    for subject in checker.collect(paths):
        for name in rule_names:
            for issue in checker.rules[name](subject):
                findings.append(Finding(issue.path, issue.line, name, issue.message))
    return sorted(findings, key=lambda finding: (str(finding.path), finding.line, finding.rule))


def run(checker: Checker[Subject], argv: list[str] | None = None) -> int:
    """Runs a checker from the command line and returns its exit code."""
    arguments = _parse_arguments(checker, sys.argv[1:] if argv is None else argv)
    paths = arguments.paths or [Path.cwd()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        print(f"no such path: {missing[0]}", file=sys.stderr)
        return K_EXIT_USAGE

    rule_names = [arguments.rule] if arguments.rule else list(checker.rules)
    findings = _collect_findings(checker, paths, rule_names)
    root = paths[0] if paths[0].is_dir() else paths[0].parent

    if arguments.report:
        counts = Counter(finding.rule for finding in findings)
        for name in rule_names:
            print(f"{name}: {counts.get(name, 0)}")
        print(f"total: {len(findings)}")
        return K_EXIT_CLEAN

    for finding in findings:
        print(finding.format(root))
    return K_EXIT_FINDINGS if findings else K_EXIT_CLEAN

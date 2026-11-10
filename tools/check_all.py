# check_all.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Runs every checker and the checker tests over a repository; prints only what failed.

The shared checkers run first, then every other `tools/<area>/check_<what>.py` the repository
holds as its own.

Usage: python tools/check_all.py [root]
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings  # noqa: E402
from checking.known_findings import (  # noqa: E402
    K_BASELINE_FILE,
    K_ID_PATTERN,
    Entry,
    KnownFindingsError,
    apply_baseline,
    orphan_lines,
    read_baseline,
    summary,
)
from commits.check_backlog_trailer import open_ids  # noqa: E402

K_TOOLS = Path(__file__).resolve().parent
#: How many checks run at once.
K_WORKERS = min(8, os.cpu_count() or 1)
K_EXTRA_TEST_FOLDERS: tuple[str, ...] = getattr(settings, "K_EXTRA_TEST_FOLDERS", ())
#: The variables that point git at a repository; a test suite runs without them.
K_REPOSITORY_VARIABLES: tuple[str, ...] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_NAMESPACE",
)
K_CHECKERS = (
    "documentation/check_source_comments.py",
    "documentation/check_docs_layout.py",
    "documentation/check_changelog.py",
    "documentation/check_backlog.py",
    "documentation/check_design_format.py",
    "documentation/check_design_topics.py",
    "documentation/check_module_install.py",
    "documentation/check_docs_truth.py",
    "documentation/check_docs_follow.py",
    "documentation/check_docs_facts.py",
    "layering/check_layering.py",
    "layering/check_agnostic.py",
    "commits/check_backlog_trailer.py",
    "testing/check_test_mirror.py",
    "testing/check_test_sizes.py",
    "testing/check_test_value.py",
    "testing/check_test_odr.py",
    "testing/check_probes.py",
    "logs/check_logging.py",
    "ci/check_workflows.py",
)


@dataclass(frozen=True, slots=True)
class Check:
    """Holds one check: its name, the command that runs it and whether it is a test suite."""

    name: str
    command: tuple[str, ...]
    is_suite: bool


def own_checker_names(found: Sequence[str]) -> tuple[str, ...]:
    """Returns the names among the found paths that are checkers the shared list does not name."""
    return tuple(
        name for name in found
        if PurePosixPath(name).name.startswith("check_")
        and name not in K_CHECKERS
        and not name.startswith("tests/")
    )


def plan_checks(root: Path, tools: Path, interpreter: str, own: Sequence[str],
                suite_folders: Sequence[Path]) -> list[Check]:
    """Returns each check with its command: shared checkers, own ones, then test suites."""
    checks = [
        Check(name, (interpreter, str(tools / name), str(root)), False)
        for name in (*K_CHECKERS, *own)
    ]
    suites = [
        Check(_suite_name(root, tools, folder),
              (interpreter, "-m", "unittest", "discover", "-q", "-s", str(folder)), True)
        for folder in suite_folders
    ]
    return [*checks, *suites]


def report(checks: Sequence[Check], results: Sequence[tuple[int, str]], known: str = "",
           orphans: Sequence[str] = ()) -> tuple[list[str], int]:
    """Returns the lines to print for the failed checks, and the exit code.

    *known* ends the clean line, such as `; 3 known findings in 1 entry`; *orphans* are the lines
    for baseline entries that name no checker, which fail under the baseline file's own name.
    """
    failed = [check.name for check, (code, _) in zip(checks, results) if code != 0]
    lines = [f"== {check.name}\n{output}".rstrip()
             for check, (code, output) in zip(checks, results) if code != 0]
    if orphans:
        failed.append(K_BASELINE_FILE)
        lines.append("\n".join([f"== {K_BASELINE_FILE}", *orphans]))
    checkers = sum(not check.is_suite for check in checks)
    lines.append(f"failed: {', '.join(failed)}" if failed
                 else f"clean: {checkers} checkers, tests{known}")
    return lines, 1 if failed else 0


def main(argv: list[str] | None = None,
         run: Callable[[Sequence[Check]], list[tuple[int, str]]] | None = None) -> int:
    """Runs every check, all at once, and returns 1 when any of them failed.

    The checks read the repository and write nothing, so they run side by side; their output
    is printed in the order they are listed. *run* runs the planned checks; it defaults to the
    thread pool.
    """
    arguments = sys.argv[1:] if argv is None else argv
    root = Path(arguments[0]) if arguments else Path.cwd()
    try:
        entries = read_baseline(root)
    except KnownFindingsError as error:
        print(error)
        return 1
    checks, results = run_plan(root, run)
    names = [check.name for check in checks if not check.is_suite]
    lines, code = report(checks, _judged(root, checks, results, entries), summary(entries),
                         orphan_lines(entries, names))
    for line in lines:
        print(line)
    return code


def run_plan(root: Path,
             run: Callable[[Sequence[Check]], list[tuple[int, str]]] | None = None,
             ) -> tuple[list[Check], list[tuple[int, str]]]:
    """Plans every check over a root, runs them and returns the checks with their raw results."""
    checks = plan_checks(root, K_TOOLS, sys.executable, own_checker_names(_read_found(K_TOOLS)),
                         _read_suite_folders(root))
    return checks, (run or _run_all)(checks)


def _judged(root: Path, checks: Sequence[Check], results: Sequence[tuple[int, str]],
            entries: Sequence[Entry]) -> list[tuple[int, str]]:
    """Returns the results with each checker's known findings taken out; test suites keep theirs."""
    if not entries:
        return list(results)
    open_backlog = open_ids(root, None) or None
    return [(code, output) if check.is_suite
            else apply_baseline(check.name, code, output, entries, K_ID_PATTERN, open_backlog)
            for check, (code, output) in zip(checks, results)]


def _suite_name(root: Path, tools: Path, folder: Path) -> str:
    """Returns the name of a test suite: `tests`, or `tests: <folder>` for another folder."""
    return "tests" if folder == tools / "tests" else f"tests: {folder.relative_to(root).as_posix()}"


def _read_found(tools: Path) -> list[str]:
    """Returns each check_*.py one folder below tools, relative to tools and in path order."""
    return [path.relative_to(tools).as_posix() for path in sorted(tools.glob("*/check_*.py"))]


def _read_suite_folders(root: Path) -> list[Path]:
    """Returns the test folders that exist: the tools' own, then the repository's extra ones."""
    candidates = [K_TOOLS / "tests", *(root / folder for folder in K_EXTRA_TEST_FOLDERS)]
    return [folder for folder in candidates if folder.is_dir()]


def suite_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Returns a copy of an environment without the variables that point git at a repository."""
    return {name: value for name, value in environment.items()
            if name not in K_REPOSITORY_VARIABLES}


def _run_check(command: Sequence[str],
               environment: Mapping[str, str] | None = None) -> tuple[int, str]:
    """Runs one check and returns its exit code and everything it printed, read as UTF-8."""
    base = os.environ if environment is None else environment
    completed = subprocess.run(list(command), capture_output=True, text=True, encoding="utf-8",
                               env={**base, "PYTHONIOENCODING": "utf-8"})
    return completed.returncode, f"{completed.stdout}{completed.stderr}"


def _run_checked(check: Check) -> tuple[int, str]:
    """Runs one check; a test suite runs without the variables that point git at a repository."""
    environment = suite_environment(os.environ) if check.is_suite else None
    return _run_check(check.command, environment)


def _run_all(checks: Sequence[Check]) -> list[tuple[int, str]]:
    """Runs every check on a thread pool and returns the results in the order of the checks."""
    with ThreadPoolExecutor(max_workers=K_WORKERS) as pool:
        return list(pool.map(_run_checked, checks))


if __name__ == "__main__":
    sys.exit(main())

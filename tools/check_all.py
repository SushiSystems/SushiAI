# check_all.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs every checker and the checker tests over a repository; prints only what failed.

The shared checkers run first, then every other `tools/<area>/check_<what>.py` the repository
holds as its own.

Usage: python tools/check_all.py [root]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

K_TOOLS = Path(__file__).resolve().parent
K_CHECKERS = (
    "documentation/check_source_comments.py",
    "documentation/check_docs_layout.py",
    "documentation/check_changelog.py",
    "documentation/check_backlog.py",
    "documentation/check_design_format.py",
    "documentation/check_design_topics.py",
    "documentation/check_module_install.py",
    "documentation/check_docs_truth.py",
    "layering/check_layering.py",
    "testing/check_test_mirror.py",
    "testing/check_test_sizes.py",
    "testing/check_probes.py",
    "logs/check_logging.py",
    "ci/check_workflows.py",
)


def own_checkers(tools: Path) -> tuple[str, ...]:
    """Returns the checkers under a tools folder that are not shared, as paths below it."""
    found = (path.relative_to(tools).as_posix() for path in sorted(tools.glob("*/check_*.py")))
    return tuple(name for name in found if name not in K_CHECKERS and not name.startswith("tests/"))


def _commands(root: Path) -> list[tuple[str, list[str]]]:
    """Returns each check by name with the command that runs it."""
    names = (*K_CHECKERS, *own_checkers(K_TOOLS))
    checks = [(name, [sys.executable, str(K_TOOLS / name), str(root)]) for name in names]
    tests = [sys.executable, "-m", "unittest", "discover", "-q", "-s", str(K_TOOLS / "tests")]
    return [*checks, ("tests", tests)]


def run_check(command: list[str]) -> tuple[int, str]:
    """Runs one check and returns its exit code and everything it printed, read as UTF-8."""
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
    return completed.returncode, f"{completed.stdout}{completed.stderr}"


def main(argv: list[str] | None = None) -> int:
    """Runs every check and returns 1 when any of them failed."""
    arguments = sys.argv[1:] if argv is None else argv
    root = Path(arguments[0]) if arguments else Path.cwd()
    failed = []
    for name, command in _commands(root):
        code, output = run_check(command)
        if code != 0:
            failed.append(name)
            print(f"== {name}\n{output}".rstrip())
    count = len(K_CHECKERS) + len(own_checkers(K_TOOLS))
    print(f"failed: {', '.join(failed)}" if failed else f"clean: {count} checkers, tests")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

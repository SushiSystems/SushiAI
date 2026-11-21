# run_suite.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Runs the unittest files of a folder side by side, one process per file name.

Usage: python tools/checking/run_suite.py <folder>

One process takes the sum of the files; this takes about the slowest one. It prints the output of
each failed file and exits 1 when any failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

K_WORKERS = min(8, os.cpu_count() or 1)
K_PATTERN = "test_*.py"
K_NO_TESTS_RAN = 5
K_SEPARATOR = "\n"


def test_file_names(folder: Path) -> list[str]:
    """Returns the distinct names of the test files below *folder*, sorted."""
    return sorted({path.name for path in folder.rglob(K_PATTERN) if path.is_file()})


def run_file(interpreter: str, folder: Path, name: str) -> tuple[int, str]:
    """Runs the tests of one file name below *folder*; returns the exit code and its output.

    A file with no test is no failure; unittest reports it as exit code 5.
    """
    done = subprocess.run(
        [interpreter, "-m", "unittest", "discover", "-q", "-s", str(folder), "-p", name],
        capture_output=True, text=True, encoding="utf-8", check=False)
    code = 0 if done.returncode == K_NO_TESTS_RAN else done.returncode
    return code, f"{done.stdout}{done.stderr}"


def run_folder(interpreter: str, folder: Path) -> tuple[int, str]:
    """Runs every test file of *folder* side by side; returns 1 and the failed output, or 0."""
    names = test_file_names(folder)
    with ThreadPoolExecutor(max_workers=K_WORKERS) as pool:
        results = list(pool.map(lambda name: run_file(interpreter, folder, name), names))
    failed = [f"== {name}\n{output}".rstrip()
              for name, (code, output) in zip(names, results) if code != 0]
    return (1, K_SEPARATOR.join(failed)) if failed else (0, "")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the folder named by the argument and prints what failed."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) != 1:
        print("usage: run_suite.py <folder>")
        return 2
    code, output = run_folder(sys.executable, Path(arguments[0]))
    if output:
        print(output)
    return code


if __name__ == "__main__":
    sys.exit(main())

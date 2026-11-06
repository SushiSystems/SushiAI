# next_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Prints the jobs of a plan that the orchestrator may start now.

Usage: python tools/routing/next_jobs.py PLAN.md [--free N]

Exit 0 when at least one job is printed, 1 when none is ready and jobs remain, 2 when every
job is done and 3 when the table is malformed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.job_table import read_jobs  # noqa: E402
from routing.ready_jobs import K_DONE, graph_problems, ready_jobs  # noqa: E402

K_EXIT_READY = 0
K_EXIT_WAITING = 1
K_EXIT_FINISHED = 2
K_EXIT_MALFORMED = 3


def plan_report(label: str, text: str, free: int) -> tuple[int, list[str]]:
    """Returns the exit code of a plan and its lines: the faults, or the ids that may start."""
    jobs, problems = read_jobs(text)
    if not problems:
        problems = graph_problems(jobs)
    if problems:
        faults = [f"{label}:{p.line}: {p.rule}: {p.message}" for p in problems]
        return K_EXIT_MALFORMED, faults
    if all(job.status == K_DONE for job in jobs):
        return K_EXIT_FINISHED, []
    ready = ready_jobs(jobs, free)
    return (K_EXIT_READY if ready else K_EXIT_WAITING), ready


def main(argv: list[str] | None = None) -> int:
    """Prints the ids of the ready jobs of a plan, one per line, and returns the exit code."""
    parser = argparse.ArgumentParser(description="Lists the jobs of a plan that may start now.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--free", type=int, default=1, help="how many subagents are free")
    arguments = parser.parse_args(argv)
    code, lines = plan_report(str(arguments.plan), _read_plan(arguments.plan), arguments.free)
    for line in lines:
        print(line)
    return code


def _read_plan(path: Path) -> str:
    """Returns the text of a plan file, read as UTF-8."""
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

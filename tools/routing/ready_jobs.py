# ready_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Chooses the jobs of a plan that may start now, by the rules of the parallel mode."""

from __future__ import annotations

from collections.abc import Sequence

from routing.job_table import Job, Problem

K_TODO = "todo"
K_RUNNING = "running"
K_DONE = "done"


def graph_problems(jobs: Sequence[Job]) -> list[Problem]:
    """Returns a fault for each wait on an unknown job and one for each job in a cycle."""
    by_id = {job.id: job for job in jobs}
    problems = [
        Problem(job.line, "unknown_job", f"job {job.id} waits on unknown job {name}")
        for job in jobs for name in job.waits_on if name not in by_id
    ]
    for job in jobs:
        if _reaches(job.id, job, by_id, set()):
            problems.append(Problem(job.line, "cycle", f"job {job.id} waits on itself"))
    return problems


def ready_jobs(jobs: Sequence[Job], free: int) -> list[str]:
    """Returns the ids of up to `free` jobs that may start, in list order.

    A job is ready when it is todo, every job it waits on is done, and none of its files
    equals, lies inside or lies above a file of a running job. A job picked here counts as
    running for the jobs after it.
    """
    by_id = {job.id: job for job in jobs}
    held = [path for job in jobs if job.status == K_RUNNING for path in job.files]
    picked: list[str] = []
    for job in jobs:
        if len(picked) >= free:
            break
        if job.status != K_TODO or not all(by_id[name].status == K_DONE for name in job.waits_on):
            continue
        if any(_overlap(path, other) for path in job.files for other in held):
            continue
        picked.append(job.id)
        held.extend(job.files)
    return picked


def _overlap(first: str, second: str) -> bool:
    """Returns True when two paths are equal or one lies inside the other."""
    one, two = first.rstrip("/"), second.rstrip("/")
    return one == two or one.startswith(two + "/") or two.startswith(one + "/")


def _reaches(target: str, job: Job, by_id: dict[str, Job], seen: set[str]) -> bool:
    """Returns True when following the waits of job leads back to target."""
    for name in job.waits_on:
        if name == target:
            return True
        if name in by_id and name not in seen:
            seen.add(name)
            if _reaches(target, by_id[name], by_id, seen):
                return True
    return False

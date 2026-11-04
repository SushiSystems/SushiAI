# job_table.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads the Jobs table of a plan into job records and reports a malformed row."""

from __future__ import annotations

import re
from dataclasses import dataclass

K_HEADING = re.compile(r"^##\s+Jobs\s*$")
K_COLUMNS = ("Job", "Files", "Waits on", "Status")
K_STATUSES = ("todo", "running", "done", "failed")
K_NONE = "-"
K_ID = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True, slots=True)
class Job:
    """Holds one row of the Jobs table."""

    id: str
    files: tuple[str, ...]
    waits_on: tuple[str, ...]
    status: str
    line: int


@dataclass(frozen=True, slots=True)
class Problem:
    """Holds one fault of the table, with the line it was found on."""

    line: int
    rule: str
    message: str


def read_jobs(text: str) -> tuple[list[Job], list[Problem]]:
    """Returns the jobs of the table under the `Jobs` heading and the faults of its rows."""
    rows = _table_rows(text.splitlines())
    if not rows:
        return [], [Problem(1, "missing_table", "no table under a `## Jobs` heading")]
    header_line, header = rows[0]
    if tuple(header) != K_COLUMNS:
        return [], [Problem(header_line, "columns", f"expected columns {', '.join(K_COLUMNS)}")]
    jobs: list[Job] = []
    problems: list[Problem] = []
    for line, cells in rows[1:]:
        job = _read_row(line, cells, {job.id for job in jobs}, problems)
        if job is not None:
            jobs.append(job)
    return jobs, problems


def _table_rows(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Returns the cells of each row of the first table after the heading, without the rule row."""
    rows: list[tuple[int, list[str]]] = []
    seen_heading = False
    for number, text in enumerate(lines, start=1):
        if not seen_heading:
            seen_heading = bool(K_HEADING.match(text))
            continue
        stripped = text.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append((number, cells))
    return rows


def _read_row(line: int, cells: list[str], known: set[str], problems: list[Problem]) -> Job | None:
    """Returns the job of a row, or None after adding the row's fault to problems."""
    if len(cells) != len(K_COLUMNS):
        problems.append(Problem(line, "columns", f"a row holds {len(cells)} cells, not {len(K_COLUMNS)}"))
        return None
    job_id, files, waits, status = cells
    if not K_ID.match(job_id):
        problems.append(Problem(line, "job_id", f"`{job_id}` is not letters and digits"))
        return None
    if job_id in known:
        problems.append(Problem(line, "repeated_id", f"job {job_id} appears twice"))
        return None
    if status not in K_STATUSES:
        problems.append(Problem(line, "status", f"`{status}` is not one of {', '.join(K_STATUSES)}"))
        return None
    return Job(job_id, _split(files), _split(waits), status, line)


def _split(cell: str) -> tuple[str, ...]:
    """Returns the comma separated items of a cell without code marks; a dash means none."""
    if cell == K_NONE:
        return ()
    return tuple(item.strip().strip("`") for item in cell.split(",") if item.strip())

# test_ready_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests which jobs of a plan may start and which graph faults are reported."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.job_table import Job  # noqa: E402
from routing.ready_jobs import graph_problems, ready_jobs  # noqa: E402


K_TEST_SIZE = "small"


def job(job_id: str, files: str = "", waits: str = "", status: str = "todo") -> Job:
    """Returns a job whose files and waits are comma separated."""
    split = lambda text: tuple(item for item in text.split(",") if item)  # noqa: E731
    return Job(job_id, split(files), split(waits), status, 1)


class ReadyJobsTest(unittest.TestCase):
    """Covers dependency order, skipping a blocked job, file clashes and cycles."""

    def test_blocked_job_is_skipped_for_a_later_ready_one(self) -> None:
        jobs = [job("A", "a", status="running"), job("B", "b", "A"), job("C", "c")]
        self.assertEqual(ready_jobs(jobs, 2), ["C"])

    def test_job_after_a_done_dependency_is_ready(self) -> None:
        self.assertEqual(ready_jobs([job("A", "a", status="done"), job("B", "b", "A")], 1), ["B"])

    def test_failed_dependency_blocks_the_jobs_behind_it(self) -> None:
        self.assertEqual(ready_jobs([job("A", "a", status="failed"), job("B", "b", "A")], 1), [])

    def test_file_inside_a_running_jobs_folder_is_held(self) -> None:
        jobs = [job("A", "tools/", status="running"), job("B", "tools/x.py"), job("C", "docs/y.md")]
        self.assertEqual(ready_jobs(jobs, 3), ["C"])

    def test_file_above_a_running_jobs_file_is_held(self) -> None:
        jobs = [job("A", "tools/x.py", status="running"), job("B", "tools")]
        self.assertEqual(ready_jobs(jobs, 1), [])

    def test_two_ready_jobs_sharing_a_file_are_not_both_picked(self) -> None:
        self.assertEqual(ready_jobs([job("A", "f"), job("B", "f"), job("C", "g")], 3), ["A", "C"])

    def test_count_limits_the_picks_in_list_order(self) -> None:
        self.assertEqual(ready_jobs([job("A", "a"), job("B", "b"), job("C", "c")], 2), ["A", "B"])

    def test_no_free_subagent_picks_nothing(self) -> None:
        self.assertEqual(ready_jobs([job("A", "a")], 0), [])

    def test_cycle_is_reported(self) -> None:
        rules = [p.rule for p in graph_problems([job("A", waits="B"), job("B", waits="A")])]
        self.assertEqual(rules, ["cycle", "cycle"])

    def test_unknown_dependency_is_reported(self) -> None:
        self.assertEqual(graph_problems([job("A", waits="Z")])[0].rule, "unknown_job")

    def test_acyclic_graph_has_no_faults(self) -> None:
        self.assertEqual(graph_problems([job("A"), job("B", waits="A"), job("C", waits="A,B")]), [])


if __name__ == "__main__":
    unittest.main()

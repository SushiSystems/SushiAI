# test_budget.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Applies the time limits of `testing` to the times of one test run and prints a line and the findings.

Usage: python tools/testing/test_budget.py JUNIT [--cases FOLDER] [--programs MAP] [--exemptions TOML]
           [--rerun JUNIT_OR_FOLDER ...] [--scope default|large|partial] [--wall SECONDS] [--root ROOT --base REVISION]
           [--not-rebuilt] [--gate] [--factor N] [--list-over FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from commits.check_backlog_trailer import K_ID_PATTERN, open_ids  # noqa: E402
from commits.history import commits_since  # noqa: E402
from testing.added_tests import added_cases  # noqa: E402

K_BUDGET: dict[str, float] = getattr(settings, "K_TEST_TIMES", {})
K_NOISE_FACTOR: float = getattr(settings, "K_TEST_NOISE_FACTOR", 1.25)
K_SIZES = ("small", "medium", "large")
K_TEST_LIMITS = {"small": "small_test_s", "medium": "medium_test_s", "large": "large_test_s"}
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1
K_EXIT_USAGE = 2


@dataclass(frozen=True, slots=True)
class Timing:
    """Holds the time one test took in one run, and its size when the JUnit file gives one as a property."""

    name: str
    seconds: float
    size: str | None = None


@dataclass(frozen=True, slots=True)
class Report:
    """Holds the budget line, the findings and the names to rerun alone."""

    line: str
    findings: tuple[str, ...]
    rerun: tuple[str, ...]


def _parse_root(data: bytes, name: str) -> ElementTree.Element:
    """Returns the root of an XML test file; a file with a DOCTYPE or an ENTITY is refused unread.

    CTest and GoogleTest write neither, and the standard library parser, the only one a checker may
    use, expands them.
    """
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise SystemExit(f"{name}: a test result file has no DOCTYPE or ENTITY; refusing to parse it")
    return ElementTree.fromstring(data)


def _was_skipped(case: ElementTree.Element) -> bool:
    """Returns whether a test case did not run: CTest and GoogleTest mark that by a child or a status."""
    return (case.find("skipped") is not None or case.get("status") == "notrun"
            or case.get("result") in ("skipped", "suppressed"))


def parse_junit(data: bytes, name: str) -> list[Timing]:
    """Returns the time of each test that ran in a JUnit file; skipped and not-run tests are left out."""
    timings: list[Timing] = []
    for case in _parse_root(data, name).iter("testcase"):
        if not _was_skipped(case):
            timings.append(Timing(case.get("name", ""), float(case.get("time", "0") or 0), _property_size(case)))
    return timings


def parse_gtest_xml(data: bytes, name: str) -> list[Timing]:
    """Returns the time of each case that ran in a GoogleTest XML file, named `Suite.Case`.

    GoogleTest writes the suite as `classname` and the case as `name`; a parametrised case keeps its
    `/index` and its instantiation prefix, so `Prefix/Suite.Case/0` reads as `suite_and_size` expects.
    """
    timings: list[Timing] = []
    for case in _parse_root(data, name).iter("testcase"):
        if not _was_skipped(case):
            timings.append(Timing(f"{case.get('classname', '')}.{case.get('name', '')}",
                                  float(case.get("time", "0") or 0)))
    return timings


def read_junit(path: Path) -> list[Timing]:
    """Returns the times of the tests in a JUnit file on disk."""
    return parse_junit(path.read_bytes(), str(path))


def read_gtest_folder(folder: Path) -> list[Timing]:
    """Returns the per-case times of every GoogleTest XML file in a folder, the files of the shards of a run."""
    timings: list[Timing] = []
    for path in sorted(folder.glob("*.xml")):
        timings.extend(parse_gtest_xml(path.read_bytes(), str(path)))
    return timings


def read_times(path: Path) -> list[Timing]:
    """Returns the times in a JUnit file, or in the GoogleTest files of a folder."""
    return read_gtest_folder(path) if path.is_dir() else read_junit(path)


def _property_size(case: ElementTree.Element) -> str | None:
    """Returns the `size` property of a JUnit test case, such as pytest writes it, when it names a size."""
    for prop in case.iter("property"):
        if prop.get("name") == "size" and prop.get("value") in K_SIZES:
            return prop.get("value")
    return None


def suite_and_size(name: str) -> tuple[str | None, str | None]:
    """Returns the suite and the size of a `Suite.Case` name; a parametrised name `Prefix/Suite.Case/0` works too."""
    for part in name.split(".")[0].split("/"):
        for size in K_SIZES:
            if part.removeprefix("Characterization_").startswith(f"{size.capitalize()}_"):
                return part, size
    return None, None


def parse_exemptions(text: str) -> dict[str, str]:
    """Returns the test name and backlog ID of each entry under `[exempt]` in TOML text."""
    table = tomllib.loads(text).get("exempt", {})
    return {str(name): str(entry) for name, entry in table.items()}


def parse_programs(text: str) -> dict[str, str]:
    """Returns the test program of each suite, from a JSON object of suite name to program."""
    return json.loads(text)


def missing_limits(budget: dict[str, float]) -> list[str]:
    """Returns the keys of the limits the reader needs that *budget* does not hold, in a fixed order."""
    return [key for key in (*K_TEST_LIMITS.values(), "pre_merge_s") if key not in budget]


def _limit(budget: dict[str, float], key: str) -> float:
    """Returns one limit of the budget."""
    return float(budget[key])


#: The share of its limit an exempt test falls to before its entry must go.
K_EXEMPT_RETURN_SHARE = 0.5


def evaluate(timings: list[Timing], *, programs: dict[str, str], exemptions: dict[str, str],
             rerun: list[Timing] | None, scope: str, wall: float | None, factor: float,
             added: int | None, base: str, rebuilt: bool, open_backlog: frozenset[str] | None,
             budget: dict[str, float] | None = None) -> Report:
    """Applies the time limits to a run and returns its line, its findings and the tests to rerun.

    Each size has a per-test limit; only `small` and `medium` count toward the pre-merge total.
    *budget* holds the limits and defaults to K_TEST_TIMES of the settings.
    """
    budget = K_BUDGET if budget is None else budget
    alone = {timing.name: timing.seconds for timing in rerun or []}
    findings: list[str] = []
    over: list[str] = []
    totals: dict[tuple[str, str], float] = defaultdict(float)
    sized: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    exempt_seconds = 0.0
    for timing in timings:
        suite, size = suite_and_size(timing.name)
        if (size or timing.size) in K_SIZES:
            counts[size or timing.size] += 1
        if size is None or suite is None:
            continue
        key = K_TEST_LIMITS.get(size)
        if timing.name in exemptions:
            exempt_seconds += timing.seconds
            if key is not None and timing.seconds <= K_EXEMPT_RETURN_SHARE * _limit(budget, key):
                findings.append(f"{timing.name}: exempt as {exemptions[timing.name]} but well within its limit; "
                                f"remove the entry")
            continue
        sized[size] += timing.seconds
        totals[(size, programs.get(suite, ""))] += timing.seconds
        if key is None:
            continue
        limit = _limit(budget, key)
        if timing.seconds > limit:
            over.append(timing.name)
            if rerun is not None and alone.get(timing.name, 0.0) > limit:
                findings.append(f"{timing.name}: {alone[timing.name]:.1f} s alone, over the {size} limit of {limit:g} s")
            elif rerun is None:
                findings.append(f"{timing.name}: {timing.seconds:.1f} s, over the {size} limit of {limit:g} s (not rerun)")
    findings += _whole_run_findings(budget, scope, sized, wall, factor)
    findings += _exemption_findings(exemptions, open_backlog)
    return Report(_line(budget, totals, sized, counts, scope, wall, added, base, rebuilt, exempt_seconds,
                       len(exemptions)),
                  tuple(findings), tuple(sorted(over)))


def _whole_run_findings(budget: dict[str, float], scope: str, sized: dict[str, float], wall: float | None,
                        factor: float) -> list[str]:
    """Returns a finding when a whole pre-merge run of small and medium tests is over its limit."""
    if scope != "default":
        return []
    seconds = wall if wall is not None else sized["small"] + sized["medium"]
    limit = _limit(budget, "pre_merge_s")
    what = "wall time" if wall is not None else "sum of times"
    return [f"pre-merge run: {seconds:.0f} s {what}, over {limit:g} s x {factor:g}"] if seconds > limit * factor else []


def _shape(counts: dict[str, int]) -> str:
    """Returns the share of each size among the cases that ran, such as `shape 80% small, 18% medium, 2% large`."""
    total = sum(counts.values())
    if total == 0:
        return "shape: no sized cases"
    return "shape " + ", ".join(f"{round(100 * counts[size] / total)}% {size}" for size in K_SIZES)


def _exemption_findings(exemptions: dict[str, str], open_backlog: frozenset[str] | None) -> list[str]:
    """Returns a finding for each exemption whose backlog entry is not an open ID."""
    found = []
    for name, entry in sorted(exemptions.items()):
        if not re.fullmatch(K_ID_PATTERN, entry) or (open_backlog is not None and entry not in open_backlog):
            found.append(f"exemption {name}: {entry!r} is not an open backlog ID")
    return found


def _line(budget: dict[str, float], totals: dict[tuple[str, str], float], sized: dict[str, float],
          counts: dict[str, int], scope: str, wall: float | None, added: int | None, base: str,
          rebuilt: bool, exempt_seconds: float, exempt_count: int) -> str:
    """Returns the one-line summary of a run."""
    parts = []
    small = {program: seconds for (size, program), seconds in totals.items() if size == "small"}
    if small:
        worst = max(small, key=lambda program: small[program])
        parts.append(f"small {small[worst]:.1f} s ({worst or 'no program map'})")
    if scope == "default":
        wall_text = f", {wall:.0f} s wall" if wall is not None else ""
        merged_seconds = sized["small"] + sized["medium"]
        parts.append(f"pre-merge {merged_seconds:.0f} s sum{wall_text} of {_limit(budget, 'pre_merge_s'):g} s")
    elif scope == "large":
        wall_text = f", {wall:.0f} s wall" if wall is not None else ""
        parts.append(f"large tier {sized['large']:.0f} s sum{wall_text}")
    else:
        parts.append("whole run not measured")
    parts.append(_shape(counts))
    if exempt_count:
        parts.append(f"{exempt_count} exempt, {exempt_seconds:.0f} s left out")
    if added is not None:
        parts.append(f"{added} cases added since {base}")
    return "budget: " + ", ".join(parts) + ("" if rebuilt else " (not rebuilt)")


def render(report: Report, *, rerun_given: bool) -> list[str]:
    """Returns the lines a run prints: the budget line, the rerun line when tests need one, the findings."""
    lines = [report.line]
    if report.rerun and not rerun_given:
        lines.append("rerun: ^(" + "|".join(re.escape(name) for name in report.rerun) + ")$")
    lines.extend(report.findings)
    return lines


def exit_code(report: Report, *, gate: bool, rebuilt: bool) -> int:
    """Returns the exit code of a run: a finding fails it only when the run gates and was rebuilt."""
    return K_EXIT_FINDINGS if gate and rebuilt and report.findings else K_EXIT_CLEAN


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description="Apply the time limits of testing to one test run.")
    parser.add_argument("junit", type=Path)
    parser.add_argument("--cases", type=Path,
                        help="folder of the GoogleTest XML files the shards wrote; their per-case times join JUNIT")
    parser.add_argument("--programs", type=Path, help="JSON object from suite name to test program")
    parser.add_argument("--exemptions", type=Path)
    parser.add_argument("--rerun", type=Path, action="append",
                        help="JUnit file, or folder of GoogleTest XML files, of tests run alone; repeatable")
    parser.add_argument("--scope", choices=("default", "large", "partial"), default="partial")
    parser.add_argument("--wall", type=float, help="wall time of the whole run, in seconds")
    parser.add_argument("--root", type=Path, help="the repository root, for open backlog IDs and added cases")
    parser.add_argument("--base", default="", help="count the cases added after this revision")
    parser.add_argument("--not-rebuilt", action="store_true", help="the run did not rebuild; it never gates")
    parser.add_argument("--gate", action="store_true", help="exit 1 when there is a finding")
    parser.add_argument("--factor", type=float, default=K_NOISE_FACTOR)
    parser.add_argument("--list-over", type=Path,
                        help="write the tests over a limit to this file, one per line, and print nothing else")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, budget: dict[str, float] | None = None) -> int:
    """Prints the budget line, the tests to rerun and the findings of a run; returns the exit code.

    The JUnit file is what `ctest --output-junit` writes. A sharded C++ program is one CTest test per
    shard, so its cases come from --cases, the folder of the GoogleTest XML files, as `Suite.Case`
    names; a CTest entry named after no size counts nowhere. MAP is a JSON object
    from suite name to test program. The exemption TOML holds `"Suite.Case" = "RW-12"` under
    `[exempt]`, and an exempt test is left out of every limit and total. The first run names the tests
    over a limit in a `rerun:` line; the caller reruns them alone and passes that file as --rerun, and
    then only a test over its limit alone is a finding. --rerun may be given more than once, and
    --list-over writes the names of that `rerun:` line to a file for a caller that reads them, printing
    nothing. Without --gate the exit code is 0. *budget* holds
    the limits and defaults to K_TEST_TIMES of the settings.
    """
    budget = K_BUDGET if budget is None else budget
    arguments = _arguments(sys.argv[1:] if argv is None else argv)
    missing = missing_limits(budget)
    if missing:
        print(f"K_TEST_TIMES in tools/settings.py lacks {', '.join(missing)}", file=sys.stderr)
        return K_EXIT_USAGE
    exemptions = parse_exemptions(arguments.exemptions.read_text(encoding="utf-8")) if arguments.exemptions else {}
    programs = parse_programs(arguments.programs.read_text(encoding="utf-8")) if arguments.programs else {}
    added = None
    open_backlog = None
    if arguments.root:
        open_backlog = open_ids(arguments.root, None)
        if arguments.base:
            added = sum(len(added_cases(commit)) for commit in commits_since(arguments.root, arguments.base))
    timings = read_junit(arguments.junit)
    if arguments.cases:
        timings += read_gtest_folder(arguments.cases)
    report = evaluate(
        timings, programs=programs, exemptions=exemptions,
        rerun=[timing for path in arguments.rerun for timing in read_times(path)] if arguments.rerun else None,
        scope=arguments.scope,
        wall=arguments.wall, factor=arguments.factor, added=added, base=arguments.base,
        rebuilt=not arguments.not_rebuilt, open_backlog=open_backlog, budget=budget)
    if arguments.list_over:
        arguments.list_over.write_text("".join(f"{name}\n" for name in report.rerun), encoding="utf-8")
        return K_EXIT_CLEAN
    for line in render(report, rerun_given=arguments.rerun is not None):
        print(line)
    return exit_code(report, gate=arguments.gate, rebuilt=not arguments.not_rebuilt)


if __name__ == "__main__":
    sys.exit(main())

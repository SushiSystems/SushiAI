# test_budget.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Applies the budget of `testing` to the times of one CTest run and prints a line and the findings.

Usage: python tools/testing/test_budget.py JUNIT [--programs MAP] [--exemptions TOML] [--rerun JUNIT]
           [--scope default|large|partial] [--wall SECONDS] [--root ROOT --base REVISION]
           [--not-rebuilt] [--gate] [--factor N]
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

K_BUDGET: dict[str, int] = getattr(settings, "K_TEST_BUDGET", {})
K_NOISE_FACTOR: float = getattr(settings, "K_TEST_NOISE_FACTOR", 1.25)
K_SIZES = ("small", "medium", "large")
K_TEST_LIMITS = {"small": "small_test_s", "medium": "medium_test_s", "large": "large_test_s"}
K_EXIT_CLEAN = 0
K_EXIT_FINDINGS = 1
K_EXIT_USAGE = 2


@dataclass(frozen=True, slots=True)
class Timing:
    """Holds the time one test took in one run."""

    name: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Report:
    """Holds the budget line, the findings and the names to rerun alone."""

    line: str
    findings: tuple[str, ...]
    rerun: tuple[str, ...]


def read_junit(path: Path) -> list[Timing]:
    """Returns the time of each test that ran in a JUnit file; skipped and not-run tests are left out.

    A file with a DOCTYPE or an ENTITY is refused unread: CTest writes neither, and the standard
    library parser, the only one a checker may use, expands them.
    """
    data = path.read_bytes()
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise SystemExit(f"{path}: a JUnit file from CTest has no DOCTYPE or ENTITY; refusing to parse it")
    timings: list[Timing] = []
    for case in ElementTree.fromstring(data).iter("testcase"):
        if case.find("skipped") is not None or case.get("status") == "notrun":
            continue
        timings.append(Timing(case.get("name", ""), float(case.get("time", "0") or 0)))
    return timings


def suite_and_size(name: str) -> tuple[str | None, str | None]:
    """Returns the suite and the size of a `Suite.Case` name; a parametrised name `Prefix/Suite.Case/0` works too."""
    for part in name.split(".")[0].split("/"):
        for size in K_SIZES:
            if part.startswith(f"{size.capitalize()}_"):
                return part, size
    return None, None


def read_exemptions(path: Path) -> dict[str, str]:
    """Returns the test name and backlog ID of each entry under `[exempt]` in a TOML file."""
    with path.open("rb") as handle:
        table = tomllib.load(handle).get("exempt", {})
    return {str(name): str(entry) for name, entry in table.items()}


def _limit(key: str) -> float:
    """Returns one limit of the budget."""
    return float(K_BUDGET[key])


def evaluate(timings: list[Timing], *, programs: dict[str, str], exemptions: dict[str, str],
             rerun: list[Timing] | None, scope: str, wall: float | None, factor: float,
             added: int | None, base: str, rebuilt: bool, open_backlog: frozenset[str] | None) -> Report:
    """Applies the budget to a run and returns its line, its findings and the tests to rerun."""
    alone = {timing.name: timing.seconds for timing in rerun or []}
    findings: list[str] = []
    over: list[str] = []
    totals: dict[tuple[str, str], float] = defaultdict(float)
    sized: dict[str, float] = defaultdict(float)
    exempt_seconds = 0.0
    for timing in timings:
        suite, size = suite_and_size(timing.name)
        if size is None or suite is None:
            continue
        if timing.name in exemptions:
            exempt_seconds += timing.seconds
            if timing.seconds <= _limit(K_TEST_LIMITS[size]):
                findings.append(f"{timing.name}: exempt as {exemptions[timing.name]} but within its limit; "
                                f"remove the entry")
            continue
        sized[size] += timing.seconds
        totals[(size, programs.get(suite, ""))] += timing.seconds
        limit = _limit(K_TEST_LIMITS[size])
        if timing.seconds > limit:
            over.append(timing.name)
            if rerun is not None and alone.get(timing.name, 0.0) > limit:
                findings.append(f"{timing.name}: {alone[timing.name]:.1f} s alone, over the {size} limit of {limit:g} s")
            elif rerun is None:
                findings.append(f"{timing.name}: {timing.seconds:.1f} s, over the {size} limit of {limit:g} s (not rerun)")
    findings += _program_findings(totals, programs, factor)
    findings += _whole_run_findings(scope, sized, wall, factor)
    findings += _exemption_findings(exemptions, open_backlog)
    return Report(_line(totals, sized, scope, wall, added, base, rebuilt, exempt_seconds, len(exemptions)),
                  tuple(findings), tuple(sorted(over)))


def _program_findings(totals: dict[tuple[str, str], float], programs: dict[str, str], factor: float) -> list[str]:
    """Returns a finding for each test program whose small or large tests take longer than its total allows."""
    if not programs:
        return []
    found = []
    for (size, program), seconds in sorted(totals.items()):
        key = {"small": "small_program_s", "large": "large_program_s"}.get(size)
        if key and program and seconds > _limit(key) * factor:
            found.append(f"{program}: its {size} tests take {seconds:.1f} s, over {_limit(key):g} s x {factor:g}")
    return found


def _whole_run_findings(scope: str, sized: dict[str, float], wall: float | None, factor: float) -> list[str]:
    """Returns a finding when a whole default run or a whole large tier is over its limit."""
    if scope == "default":
        seconds = wall if wall is not None else sized["small"] + sized["medium"]
        limit = _limit("default_run_s")
        what = "wall time" if wall is not None else "sum of times"
        return [f"default run: {seconds:.0f} s {what}, over {limit:g} s x {factor:g}"] if seconds > limit * factor else []
    if scope == "large":
        seconds = wall if wall is not None else sized["large"]
        limit = _limit("large_tier_s")
        return [f"large tier: {seconds:.0f} s, over {limit:g} s x {factor:g}"] if seconds > limit * factor else []
    return []


def _exemption_findings(exemptions: dict[str, str], open_backlog: frozenset[str] | None) -> list[str]:
    """Returns a finding for each exemption whose backlog entry is not an open ID."""
    found = []
    for name, entry in sorted(exemptions.items()):
        if not re.fullmatch(K_ID_PATTERN, entry) or (open_backlog is not None and entry not in open_backlog):
            found.append(f"exemption {name}: {entry!r} is not an open backlog ID")
    return found


def _line(totals: dict[tuple[str, str], float], sized: dict[str, float], scope: str, wall: float | None,
          added: int | None, base: str, rebuilt: bool, exempt_seconds: float, exempt_count: int) -> str:
    """Returns the one-line summary of a run."""
    parts = []
    small = {program: seconds for (size, program), seconds in totals.items() if size == "small"}
    if small:
        worst = max(small, key=lambda program: small[program])
        parts.append(f"small {small[worst]:.1f} s of {_limit('small_program_s'):g} s ({worst or 'no program map'})")
    if scope == "default":
        wall_text = f", {wall:.0f} s wall" if wall is not None else ""
        parts.append(f"default {sized['small'] + sized['medium']:.0f} s sum{wall_text} of {_limit('default_run_s'):g} s")
    elif scope == "large":
        parts.append(f"large tier {sized['large']:.0f} s of {_limit('large_tier_s'):g} s")
    else:
        parts.append("whole run not measured")
    if exempt_count:
        parts.append(f"{exempt_count} exempt, {exempt_seconds:.0f} s left out")
    if added is not None:
        parts.append(f"{added} cases added since {base}")
    return "budget: " + ", ".join(parts) + ("" if rebuilt else " (not rebuilt)")


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description="Apply the test budget to one test run.")
    parser.add_argument("junit", type=Path)
    parser.add_argument("--programs", type=Path, help="JSON object from suite name to test program")
    parser.add_argument("--exemptions", type=Path)
    parser.add_argument("--rerun", type=Path, help="JUnit file of the tests named by `rerun:`, run alone")
    parser.add_argument("--scope", choices=("default", "large", "partial"), default="partial")
    parser.add_argument("--wall", type=float, help="wall time of the whole run, in seconds")
    parser.add_argument("--root", type=Path, help="the repository root, for open backlog IDs and added cases")
    parser.add_argument("--base", default="", help="count the cases added after this revision")
    parser.add_argument("--not-rebuilt", action="store_true", help="the run did not rebuild; it never gates")
    parser.add_argument("--gate", action="store_true", help="exit 1 when there is a finding")
    parser.add_argument("--factor", type=float, default=K_NOISE_FACTOR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Prints the budget line, the tests to rerun and the findings of a run; returns the exit code.

    The JUnit file is what `ctest --output-junit` writes, with `Suite.Case` names. MAP is a JSON object
    from suite name to test program. The exemption TOML holds `"Suite.Case" = "RW-12"` under
    `[exempt]`, and an exempt test is left out of every limit and total. The first run names the tests
    over a limit in a `rerun:` line; the caller reruns them alone and passes that file as --rerun, and
    then only a test over its limit alone is a finding. Without --gate the exit code is 0.
    """
    arguments = _arguments(sys.argv[1:] if argv is None else argv)
    if not K_BUDGET:
        print("no K_TEST_BUDGET in tools/settings.py", file=sys.stderr)
        return K_EXIT_USAGE
    exemptions = read_exemptions(arguments.exemptions) if arguments.exemptions else {}
    programs = json.loads(arguments.programs.read_text(encoding="utf-8")) if arguments.programs else {}
    added = None
    open_backlog = None
    if arguments.root:
        open_backlog = open_ids(arguments.root, None)
        if arguments.base:
            added = sum(len(added_cases(commit)) for commit in commits_since(arguments.root, arguments.base))
    report = evaluate(
        read_junit(arguments.junit), programs=programs, exemptions=exemptions,
        rerun=read_junit(arguments.rerun) if arguments.rerun else None, scope=arguments.scope,
        wall=arguments.wall, factor=arguments.factor, added=added, base=arguments.base,
        rebuilt=not arguments.not_rebuilt, open_backlog=open_backlog)
    print(report.line)
    if report.rerun and not arguments.rerun:
        print("rerun: ^(" + "|".join(re.escape(name) for name in report.rerun) + ")$")
    for finding in report.findings:
        print(finding)
    gates = arguments.gate and not arguments.not_rebuilt and report.findings
    return K_EXIT_FINDINGS if gates else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

# known_findings.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Judges a checker's findings against `tools/known_findings.toml`, a baseline that only shrinks.

Pure core on top, file and process shell below. Usage: known_findings.py ROOT --write --backlog ID [--replace]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402

K_ID_PATTERN: str = getattr(settings, "K_BACKLOG_ID_PATTERN", r"[A-Z]{2}-[0-9]+")
K_BASELINE_FILE = "tools/known_findings.toml"
K_FINDING_LINE = re.compile(r"^(?P<path>[^\s:][^:]*):(?P<line>\d+): (?P<rule>\w+): (?P<message>.*)$")
K_COVERABLE_CODES = (0, 1)
K_EXIT_CLEAN = 0
K_EXIT_FAILED = 1
K_EXIT_USAGE = 2


class KnownFindingsError(Exception):
    """Signals a baseline file that cannot be read or written."""


@dataclass(frozen=True, slots=True)
class Finding:
    """Holds one finding line of a checker with the rule and the path that group it."""

    text: str
    rule: str
    path: str


@dataclass(frozen=True, slots=True)
class Parsed:
    """Holds what a checker printed: its finding lines and every other non-blank line."""

    findings: tuple[Finding, ...]
    others: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Entry:
    """Holds one known group: a checker's rule over a path, how often it may be found, and its owner."""

    checker: str
    rule: str
    path: str
    count: int
    backlog: str

    def label(self) -> str:
        """Returns the entry as `<checker> <rule> <path>`, the way messages name it."""
        return f"{self.checker} {self.rule} {self.path}"


def parse_output(text: str) -> Parsed:
    """Splits a checker's output into finding lines (`path:line: rule: message`) and the rest."""
    findings: list[Finding] = []
    others: list[str] = []
    for line in text.splitlines():
        match = K_FINDING_LINE.match(line)
        if match:
            findings.append(Finding(line, match["rule"], match["path"]))
        elif line.strip():
            others.append(line)
    return Parsed(tuple(findings), tuple(others))


def count_groups(findings: Sequence[Finding]) -> dict[tuple[str, str], int]:
    """Returns how often each (rule, path) group occurs."""
    return dict(Counter((finding.rule, finding.path) for finding in findings))


def entry_problems(entry: Entry, id_pattern: str, open_backlog: frozenset[str] | None) -> list[str]:
    """Returns what is wrong with an entry's backlog ID; empty when it matches and, if known, is open."""
    if not re.fullmatch(id_pattern, entry.backlog):
        return [f"known finding backlog invalid: {entry.label()}: {entry.backlog!r} does not match {id_pattern}"]
    if open_backlog is not None and entry.backlog not in open_backlog:
        return [f"known finding backlog closed: {entry.label()}: {entry.backlog} is not an open backlog ID"]
    return []


def apply_baseline(checker: str, code: int, output: str, entries: Sequence[Entry], id_pattern: str,
                   open_backlog: frozenset[str] | None) -> tuple[int, str]:
    """Returns the exit code and output of a checker once its known findings are taken out.

    A checker with no entry is returned as it came. One that exited other than 0 or 1, or printed a
    line that is no finding, is never covered. Otherwise a group found as often as its entry allows
    is dropped, a group found more often keeps every line, and a group found less often, an entry
    found never and a bad backlog ID each give one line; the checker passes when nothing is left.
    """
    mine = [entry for entry in entries if entry.checker == checker]
    if not mine:
        return code, output
    problems = [line for entry in mine for line in entry_problems(entry, id_pattern, open_backlog)]
    parsed = parse_output(output)
    if code not in K_COVERABLE_CODES or parsed.others:
        return (code or int(bool(problems))), _joined([output.rstrip("\n"), *problems])
    allowed = {(entry.rule, entry.path): entry for entry in mine}
    found = count_groups(parsed.findings)
    lines = [finding.text for finding in parsed.findings
             if (finding.rule, finding.path) not in allowed
             or found[(finding.rule, finding.path)] > allowed[(finding.rule, finding.path)].count]
    for key, entry in allowed.items():
        if found.get(key, 0) == 0:
            problems.append(f"known finding gone: {entry.label()}: allowed {entry.count}, found 0; remove the entry")
        elif found[key] < entry.count:
            problems.append(f"known finding shrank: {entry.label()}: found {found[key]}, "
                            f"allowed {entry.count}; lower the count")
    lines += problems
    return (K_EXIT_FAILED if lines else K_EXIT_CLEAN), _joined(lines)


def orphan_lines(entries: Sequence[Entry], names: Sequence[str]) -> list[str]:
    """Returns a line for each entry whose checker is not among *names*."""
    return [f"known finding names no checker: {entry.label()}" for entry in entries
            if entry.checker not in names]


def summary(entries: Sequence[Entry]) -> str:
    """Returns the tail of the clean line, `; N known findings in M entries`, or "" for no entries."""
    if not entries:
        return ""
    total = sum(entry.count for entry in entries)
    return (f"; {total} known finding{'' if total == 1 else 's'} "
            f"in {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")


def parse_entries(data: Mapping[str, object]) -> list[Entry]:
    """Returns the entries of a parsed baseline; raises KnownFindingsError on a bad or repeated one."""
    tables = data.get("finding", [])
    if not isinstance(tables, list):
        raise KnownFindingsError("`finding` must be an array of tables")
    entries = [_entry_of(table) for table in tables]
    repeated = [key for key, times in Counter((e.checker, e.rule, e.path) for e in entries).items() if times > 1]
    if repeated:
        raise KnownFindingsError(f"entry repeated: {' '.join(repeated[0])}")
    return entries


def render_baseline(found: Mapping[str, Sequence[Finding]], backlog: str) -> str:
    """Returns the baseline text for the findings of each checker: one sorted entry per group, LF only."""
    blocks = []
    for checker in sorted(found):
        for (rule, path), count in sorted(count_groups(found[checker]).items()):
            blocks.append("[[finding]]\n" + "".join(
                f"{name} = {json.dumps(value, ensure_ascii=False)}\n"
                for name, value in (("checker", checker), ("rule", rule), ("path", path), ("count", count),
                                    ("backlog", backlog))))
    return "\n".join(blocks)


def _entry_of(table: object) -> Entry:
    """Returns one entry from its parsed table; raises KnownFindingsError when a field is missing or wrong."""
    fields = [table.get(name) for name in ("checker", "rule", "path", "backlog", "count")] \
        if isinstance(table, dict) else []
    if (len(fields) != 5 or not all(isinstance(field, str) and field for field in fields[:4])
            or not isinstance(fields[4], int) or isinstance(fields[4], bool) or fields[4] < 1):
        raise KnownFindingsError(f"a finding needs checker, rule, path, backlog and a count of 1 or more: {table}")
    return Entry(fields[0], fields[1], fields[2], fields[4], fields[3])


def _joined(lines: Sequence[str]) -> str:
    """Returns the non-empty lines as text with a closing newline, or "" for none."""
    kept = [line for line in lines if line]
    return "\n".join(kept) + "\n" if kept else ""


def read_baseline(root: Path) -> list[Entry]:
    """Returns the entries of a root's baseline file; an absent file is an empty baseline."""
    path = root / K_BASELINE_FILE
    if not path.is_file():
        return []
    try:
        return parse_entries(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise KnownFindingsError(f"{K_BASELINE_FILE}: {error}") from error


def write_baseline(root: Path, text: str, replace: bool) -> Path:
    """Writes the baseline text with LF endings; refuses an existing file unless *replace*."""
    path = root / K_BASELINE_FILE
    if path.exists() and not replace:
        raise KnownFindingsError(f"{K_BASELINE_FILE} exists; give --replace to overwrite it")
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Returns the parsed command line."""
    parser = argparse.ArgumentParser(description="Write the known-findings baseline from the findings now.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true", required=True)
    parser.add_argument("--backlog", required=True, help="the backlog ID that owns every entry")
    parser.add_argument("--replace", action="store_true", help="overwrite an existing baseline")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    run_plan: Callable[[Path], tuple[list, list[tuple[int, str]]]] | None = None,
) -> int:
    """Runs the checkers of `check_all.py` over a root and writes the baseline of what they print.

    *run_plan* takes the root and returns the checks and their results; it defaults to `check_all.run_plan`.
    """
    import check_all  # here, because check_all imports this module at its top

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    if not re.fullmatch(K_ID_PATTERN, arguments.backlog):
        print(f"--backlog {arguments.backlog!r} does not match {K_ID_PATTERN}", file=sys.stderr)
        return K_EXIT_USAGE
    checks, results = (check_all.run_plan if run_plan is None else run_plan)(arguments.root)
    found: dict[str, Sequence[Finding]] = {}
    uncovered = []
    for check, (code, output) in zip(checks, results):
        if check.is_suite:
            continue
        parsed = parse_output(output)
        if parsed.findings:
            found[check.name] = parsed.findings
        if parsed.others or (code and not parsed.findings):
            uncovered.append(check.name)
    try:
        path = write_baseline(arguments.root, render_baseline(found, arguments.backlog), arguments.replace)
    except KnownFindingsError as error:
        print(error, file=sys.stderr)
        return K_EXIT_USAGE
    print(f"wrote {path.as_posix()}: {sum(len(items) for items in found.values())} findings")
    for name in uncovered:
        print(f"not covered, prints more than findings: {name}")
    return K_EXIT_FAILED if uncovered else K_EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())

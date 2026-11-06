# added_tests.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Lists the test cases and test programs each commit of a range adds.

Usage: python tools/testing/added_tests.py [--root ROOT] [--base REVISION]

A case that moves between files, or is renamed with its file, adds nothing; a benchmark is not a
case. The list is what a report on work that added tests names, as `testing` says.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.tests_tree import K_TEST_FOLDERS  # noqa: E402
from commits.history import Commit, commits_since, git  # noqa: E402
from testing.case_reader import K_BENCHMARK, Case, cases_of, is_test_file  # noqa: E402

K_SINCE: str = getattr(settings, "K_TEST_ADDITIONS_SINCE", "")
K_PROGRAM = re.compile(
    getattr(settings, "K_TEST_PROGRAM_PATTERN", r"^[ \t]*(?:\w*add_test_program|add_executable)[ \t]*\("),
    re.MULTILINE)
K_BUILD_FILE = "CMakeLists.txt"


def under_test_folders(names: Sequence[str]) -> list[Path]:
    """Returns the paths, of those names, that lie below a test folder."""
    paths = [Path(name) for name in names if name]
    return [path for path in paths if any(path.parts[:len(Path(folder).parts)] == Path(folder).parts
                                          for folder in K_TEST_FOLDERS)]


def _cases(texts: Mapping[Path, str]) -> list[Case]:
    """Returns the cases the test files of a revision hold, benchmarks left out."""
    return [case for path, text in texts.items() if is_test_file(path)
            for case in cases_of(path, text) if case.size != K_BENCHMARK]


def new_case_names(before: Mapping[Path, str], after: Mapping[Path, str]) -> list[str]:
    """Returns the sorted names of the cases that the after-texts add to the before-texts.

    A case that keeps its name is not new, whatever file it sits in and however its body changes. One
    with a new name is not new either when a case with the same body went away in the change, which is
    a rename. A copy under a new name leaves the original in place, so it counts.
    """
    prior = _cases(before)
    current = _cases(after)
    pool = Counter(case.name for case in prior)
    unmatched: list[Case] = []
    for case in current:
        if pool[case.name] > 0:
            pool[case.name] -= 1
        else:
            unmatched.append(case)
    gone: Counter[str] = Counter()
    for case in prior:
        if pool[case.name] > 0:
            pool[case.name] -= 1
            gone[case.digest] += 1
    added: list[str] = []
    for case in unmatched:
        if gone[case.digest] > 0:
            gone[case.digest] -= 1
        else:
            added.append(case.name)
    return sorted(added)


def _programs(text: str) -> int:
    """Returns how many test programs a build file declares."""
    return len(K_PROGRAM.findall(text))


def new_program_files(before: Mapping[Path, str], after: Mapping[Path, str]) -> list[Path]:
    """Returns the build files whose after-text declares a second test program or more, having gained one."""
    return [path for path, text in after.items() if path.name == K_BUILD_FILE
            and _programs(text) > max(1, _programs(before.get(path, "")))]


def changed_test_files(commit: Commit) -> list[Path]:
    """Returns the files a commit adds, edits or removes under a test folder."""
    listing = git(commit.root, "diff", "--name-only", "-z", commit.parent, commit.sha) or ""
    return under_test_folders(listing.split("\0"))


def _texts_at(commit: Commit, revision: str, paths: Sequence[Path]) -> dict[Path, str]:
    """Returns the text of each path at a revision; a path the revision lacks is left out."""
    texts: dict[Path, str] = {}
    for path in paths:
        text = git(commit.root, "show", f"{revision}:{path.as_posix()}")
        if text is not None:
            texts[path] = text
    return texts


def added_cases(commit: Commit) -> list[str]:
    """Returns the sorted names of the cases a commit adds."""
    paths = changed_test_files(commit)
    return new_case_names(_texts_at(commit, commit.parent, paths), _texts_at(commit, commit.sha, paths))


def added_programs(commit: Commit) -> list[Path]:
    """Returns the build files of a commit that end with a second test program or more, having gained one."""
    paths = changed_test_files(commit)
    return new_program_files(_texts_at(commit, commit.parent, paths), _texts_at(commit, commit.sha, paths))


def main(argv: list[str] | None = None) -> int:
    """Prints, for each commit after a base, the cases and the build files it adds; returns 0."""
    parser = argparse.ArgumentParser(description="List the test cases each commit of a range adds.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="the repository root; default: .")
    parser.add_argument("--base", default=K_SINCE, help="the revision before the first commit to list")
    arguments = parser.parse_args(argv)
    if not arguments.base:
        print("no base: pass --base or set K_TEST_ADDITIONS_SINCE in tools/settings.py", file=sys.stderr)
        return 2
    for commit in commits_since(arguments.root, arguments.base):
        names = added_cases(commit)
        programs = added_programs(commit)
        if not names and not programs:
            continue
        print(f"{commit.sha[:9]} {commit.message.splitlines()[0]}")
        for name in names:
            print(f"    case {name}")
        for path in programs:
            print(f"    test program in {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

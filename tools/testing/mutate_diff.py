# mutate_diff.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Mutates each changed line of a Python diff once and reports which mutants the tests kill.

The core takes strings and returns values; the shell runs git and the tests on a temporary copy.
It reports and does not gate. Usage: python tools/testing/mutate_diff.py [root] [--base COMMIT]
[--tests-dir DIR] [--test-command COMMAND] [--no-narrow] [--timeout SECONDS]
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

K_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+?)\t?$")
K_DIFF_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
K_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "critical", "exception", "log"})
K_CAPACITY_METHODS = frozenset({"reserve"})
K_TESTS_FOLDER = "tests"
K_DEFAULT_TESTS_DIR = "tools/tests"
K_DEFAULT_TIMEOUT_IN_SECONDS = 120
K_COMPARE_FLIPS: dict[type[ast.cmpop], tuple[bytes, str, str]] = {
    ast.Eq: (rb"==", "==", "!="),
    ast.NotEq: (rb"!=", "!=", "=="),
    ast.Lt: (rb"<(?!=)", "<", ">="),
    ast.LtE: (rb"<=", "<=", ">"),
    ast.Gt: (rb">(?!=)", ">", "<="),
    ast.GtE: (rb">=", ">=", "<"),
    ast.Is: (rb"\bis\b(?!\s+not\b)", "is", "is not"),
    ast.IsNot: (rb"\bis\s+not\b", "is not", "is"),
    ast.In: (rb"\bin\b", "in", "not in"),
    ast.NotIn: (rb"\bnot\s+in\b", "not in", "in"),
}


class MutateError(Exception):
    """Raised when the mutation run cannot start or cannot trust its baseline."""


@dataclass(frozen=True, slots=True)
class Mutation:
    """Holds one edit of one source line: a byte span of the line, its replacement and its name."""

    line: int
    start: int
    end: int
    replacement: str
    label: str


@dataclass(frozen=True, slots=True)
class Result:
    """Holds the outcome of one mutant."""

    path: str
    mutation: Mutation
    killed: bool


def parse_changed_lines(diff_text: str) -> dict[str, frozenset[int]]:
    """Returns, for each file a `git diff -U0` text adds or edits, the lines it adds or edits."""
    lines: dict[str, set[int]] = {}
    current: str | None = None
    for text in diff_text.splitlines():
        if text.startswith("+++ "):
            file = K_DIFF_FILE.match(text)
            current = file.group(1) if file else None
            continue
        hunk = K_DIFF_HUNK.match(text)
        if hunk and current is not None:
            first = int(hunk.group(1))
            count = 1 if hunk.group(2) is None else int(hunk.group(2))
            lines.setdefault(current, set()).update(range(first, first + count))
    return {path: frozenset(numbers) for path, numbers in lines.items() if numbers}


def is_mutable_path(path: str) -> bool:
    """Returns whether a diff path is a Python source outside every tests folder."""
    parts = PurePosixPath(path).parts
    is_test = K_TESTS_FOLDER in parts[:-1] or parts[-1].startswith("test_")
    return parts[-1].endswith(".py") and not is_test


def mutate_line(source: str, line: int) -> tuple[Mutation, str] | None:
    """Returns the one mutation of a line and the mutated source, or None when nothing applies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if _is_skipped(tree, line):
        return None
    row = source.split("\n")[line - 1].encode("utf-8")
    nodes = sorted(
        (node for node in ast.walk(tree) if getattr(node, "lineno", None) == line
         and getattr(node, "end_lineno", None) == line),
        key=lambda node: node.col_offset,
    )
    for find in K_FINDERS:
        for mutation in find(nodes, row, line):
            mutated = apply_mutation(source, mutation)
            if mutated != source and _parses(mutated):
                return mutation, mutated
    return None


def apply_mutation(source: str, mutation: Mutation) -> str:
    """Returns the source with the mutation's span of its line replaced."""
    lines = source.split("\n")
    row = lines[mutation.line - 1].encode("utf-8")
    edited = row[: mutation.start] + mutation.replacement.encode("utf-8") + row[mutation.end:]
    lines[mutation.line - 1] = edited.decode("utf-8")
    return "\n".join(lines)


def module_names(path: str) -> tuple[str, ...]:
    """Returns the dotted names a test may import a source file by, longest first."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return tuple(".".join(parts[start:]) for start in range(len(parts)))


def imports_any(test_source: str, names: Sequence[str]) -> bool:
    """Returns whether a test source imports a module by one of the dotted names."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return False
    wanted = set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name in wanted for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module in wanted:
                return True
            if any(f"{node.module}.{alias.name}" in wanted for alias in node.names):
                return True
    return False


def format_result(result: Result) -> str:
    """Returns the output line of one mutant."""
    verdict = "killed" if result.killed else "survived"
    return f"{result.path}:{result.mutation.line} {result.mutation.label} {verdict}"


def format_summary(results: Sequence[Result]) -> str:
    """Returns the closing line with the mutant, killed and survived counts."""
    killed = sum(1 for result in results if result.killed)
    return f"mutants: {len(results)}, killed: {killed}, survived: {len(results) - killed}"


def _parses(source: str) -> bool:
    """Returns whether a source text is valid Python."""
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _innermost_statement(tree: ast.Module, line: int) -> ast.stmt | None:
    """Returns the smallest statement that covers a line."""
    covering = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    return min(covering, key=lambda node: (node.end_lineno or node.lineno) - node.lineno, default=None)


def _root_name(node: ast.expr) -> str:
    """Returns the leftmost name of an attribute or call chain, or an empty string."""
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


def _is_skipped(tree: ast.Module, line: int) -> bool:
    """Returns whether a line holds no statement, or only an import, a docstring, a log call or a hint."""
    statement = _innermost_statement(tree, line)
    if statement is None or isinstance(statement, (ast.Import, ast.ImportFrom)):
        return True
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        is_log = value.func.attr in K_LOG_METHODS and "log" in _root_name(value.func).lower()
        return is_log or value.func.attr in K_CAPACITY_METHODS
    return False


def _find_compare(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the flips of comparison operators."""
    for node in nodes:
        if not isinstance(node, ast.Compare):
            continue
        before = node.left
        for operator, after in zip(node.ops, node.comparators):
            if before.end_lineno == line and after.lineno == line:
                pattern, name, flipped = K_COMPARE_FLIPS[type(operator)]
                found = re.search(pattern, row[before.end_col_offset: after.col_offset])
                if found:
                    start = before.end_col_offset + found.start()
                    end = before.end_col_offset + found.end()
                    yield Mutation(line, start, end, flipped, f"{name} to {flipped}")
            before = after


def _find_bool_operator(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the swaps of `and` and `or`."""
    for node in nodes:
        if not isinstance(node, ast.BoolOp):
            continue
        first, second = node.values[0], node.values[1]
        if first.end_lineno == line and second.lineno == line:
            found = re.search(rb"\b(and|or)\b", row[first.end_col_offset: second.col_offset])
            if found:
                name = found.group(1).decode()
                swapped = "or" if name == "and" else "and"
                start = first.end_col_offset + found.start()
                yield Mutation(line, start, first.end_col_offset + found.end(), swapped,
                               f"{name} to {swapped}")


def _find_not(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the removals of `not`."""
    for node in nodes:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            found = re.match(rb"not\s*", row[node.col_offset:])
            if found:
                yield Mutation(line, node.col_offset, node.col_offset + found.end(), "", "not dropped")


def _find_bool_constant(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the flips of `True` and `False`."""
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            flipped = str(not node.value)
            yield Mutation(line, node.col_offset, node.end_col_offset, flipped,
                           f"{node.value} to {flipped}")


def _find_number(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the increments of integer and float constants."""
    for node in nodes:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            raised = repr(node.value + 1)
            yield Mutation(line, node.col_offset, node.end_col_offset, raised,
                           f"{node.value!r} to {raised}")


def _find_return(nodes: Sequence[ast.AST], row: bytes, line: int) -> Iterator[Mutation]:
    """Yields the replacements of `return x` by `return None`."""
    for node in nodes:
        if isinstance(node, ast.Return) and node.value is not None:
            is_none = isinstance(node.value, ast.Constant) and node.value.value is None
            if not is_none:
                yield Mutation(line, node.col_offset, node.end_col_offset, "return None",
                               "return value to return None")


K_FINDERS: tuple[Callable[[Sequence[ast.AST], bytes, int], Iterator[Mutation]], ...] = (
    _find_compare,
    _find_bool_operator,
    _find_not,
    _find_bool_constant,
    _find_number,
    _find_return,
)


def _git(root: Path, *arguments: str) -> str:
    """Returns the output of a git command, or raises MutateError with its reason."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, encoding="utf-8")
    except OSError as error:
        raise MutateError(f"cannot start git: {error}") from error
    if completed.returncode != 0:
        raise MutateError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def _snapshot(root: Path, destination: Path) -> None:
    """Copies the working tree's tracked and unignored files into a folder."""
    listing = _git(root, "ls-files", "-co", "--exclude-standard", "-z")
    for name in filter(None, listing.split("\0")):
        source = root / name
        if source.is_file():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / name)


def _passes(command: Sequence[str], cwd: Path, pythonpath: str, timeout: int) -> bool:
    """Runs the test command and returns whether it exited 0 before the timeout."""
    environment = {**os.environ, "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, env=environment, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except OSError as error:
        raise MutateError(f"cannot start the test command: {error}") from error
    return completed.returncode == 0


def _test_modules(tests_root: Path, names: Sequence[str]) -> list[str]:
    """Returns the dotted test modules under a tests folder that import a module by one of the names."""
    found = []
    for path in sorted(tests_root.rglob("test_*.py")):
        if imports_any(path.read_text(encoding="utf-8", errors="replace"), names):
            found.append(".".join(path.relative_to(tests_root).with_suffix("").parts))
    return found


def _command_for(
    copy: Path, tests_dir: str, relative: str, test_command: Sequence[str] | None, narrow: bool
) -> list[str]:
    """Returns the test command for one mutated file: the given one, the narrowed one or the default."""
    if test_command:
        return list(test_command)
    modules = _test_modules(copy / tests_dir, module_names(relative)) if narrow else []
    if modules:
        return [sys.executable, "-m", "unittest", *modules]
    return [sys.executable, "-m", "unittest", "discover", "-s", tests_dir, "-t", tests_dir]


def _run_mutant(target: Path, original: bytes, mutated: str, run: Callable[[], bool]) -> bool:
    """Writes the mutant, runs the tests and restores the file; returns whether the mutant is killed."""
    try:
        target.write_bytes(mutated.encode("utf-8"))
        return not run()
    finally:
        target.write_bytes(original)


def mutate_diff(
    root: Path,
    base: str,
    tests_dir: str,
    test_command: Sequence[str] | None,
    narrow: bool,
    timeout: int,
    report: Callable[[Result], None],
) -> list[Result]:
    """Mutates each changed line of the diff after *base* and returns the outcome of each mutant."""
    diff = _git(root, "-c", "core.quotepath=off", "diff", "-U0", "--no-color", "--no-ext-diff", base, "--")
    changed = {path: lines for path, lines in parse_changed_lines(diff).items() if is_mutable_path(path)}
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="mutate_diff_") as scratch:
        copy = Path(scratch)
        _snapshot(root, copy)
        pythonpath = str(copy / tests_dir)
        if test_command is None and not (copy / tests_dir).is_dir():
            raise MutateError(f"the tests folder {tests_dir} does not exist")
        baseline = _command_for(copy, tests_dir, "", test_command, narrow=False)
        if not _passes(baseline, copy, pythonpath, timeout):
            raise MutateError("the test command fails or times out on the unmutated tree")
        for path in sorted(changed):
            target = copy / path
            if not target.is_file():
                continue
            original = target.read_bytes()
            source = original.decode("utf-8", errors="replace")
            command = _command_for(copy, tests_dir, path, test_command, narrow)
            for line in sorted(changed[path]):
                outcome = mutate_line(source, line)
                if outcome is None:
                    continue
                mutation, mutated = outcome
                killed = _run_mutant(
                    target, original, mutated, lambda: _passes(command, copy, pythonpath, timeout))
                result = Result(path, mutation, killed)
                results.append(result)
                report(result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the mutation of the diff and prints one line per mutant, then the totals."""
    parser = argparse.ArgumentParser(description="Mutates the changed lines of a diff once each.")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--base", default="HEAD~1", help="commit the diff starts from")
    parser.add_argument("--tests-dir", default=K_DEFAULT_TESTS_DIR)
    parser.add_argument("--test-command", help="command run per mutant, instead of unittest")
    parser.add_argument("--no-narrow", action="store_true", help="run every test for every mutant")
    parser.add_argument("--timeout", type=int, default=K_DEFAULT_TIMEOUT_IN_SECONDS)
    arguments = parser.parse_args(argv)
    command = shlex.split(arguments.test_command) if arguments.test_command else None
    try:
        root = Path(_git(arguments.root, "rev-parse", "--show-toplevel").strip())
        results = mutate_diff(
            root, arguments.base, arguments.tests_dir, command, not arguments.no_narrow,
            arguments.timeout, lambda result: print(format_result(result), flush=True))
    except MutateError as error:
        print(f"mutate_diff: {error}", file=sys.stderr)
        return 2
    print(format_summary(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

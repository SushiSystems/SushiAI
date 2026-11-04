# check_test_sizes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every test writes its size in its source, from `testing`.

The rules read SourceFile values only; collect() reads the disk through checking.tests_tree.

Usage: python tools/testing/check_test_sizes.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.source_file import SourceFile, line_of, read_files  # noqa: E402
from checking.tests_tree import discover  # noqa: E402
from testing.case_reader import class_size, is_test_class  # noqa: E402

K_SIZES = ("small", "medium", "large")
K_BENCHMARK_PREFIX = "Benchmark_"
K_CPP_SUFFIXES = frozenset({".cpp", ".cc", ".cxx"})
K_CPP_SUITE = re.compile(
    r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*(\w+)\s*,",
    re.MULTILINE,
)
K_CPP_PREFIXES = tuple(f"{size.capitalize()}_" for size in K_SIZES)
K_TS_DESCRIBE = re.compile(r"^describe\s*\(\s*([\"'`])(.*?)\1", re.MULTILINE)
K_TS_PREFIXES = tuple(f"{size}:" for size in K_SIZES)
K_CS_CLASS = re.compile(
    r"^[ \t]*((?:\[[^\]\n]*\][ \t]*\n?[ \t]*)*)"
    r"((?:(?:public|internal|private|protected|sealed|static|abstract|partial)\s+)*)"
    r"class\s+(\w+)",
    re.MULTILINE,
)
K_CS_TEST_ATTRIBUTE = re.compile(r"\[\s*(?:Test|UnityTest|TestCase|TestCaseSource|Theory)\b")
K_CS_CATEGORY = re.compile(r"Category\s*\(\s*\"(\w+)\"\s*\)")


def _is_cpp_test(path: Path) -> bool:
    """Returns whether a path names a C++ test source."""
    return path.suffix in K_CPP_SUFFIXES and path.name.startswith("test_")


def _is_python_test(path: Path) -> bool:
    """Returns whether a path names a pytest file."""
    return path.suffix == ".py" and path.name.startswith("test_")


def _is_typescript_test(path: Path) -> bool:
    """Returns whether a path names a TypeScript test file."""
    return path.name.endswith((".test.ts", ".test.tsx"))


def _is_csharp_source(path: Path) -> bool:
    """Returns whether a path names a C# source."""
    return path.suffix == ".cs"


K_FILE_KINDS = (_is_cpp_test, _is_python_test, _is_typescript_test, _is_csharp_source)


def is_sized_file(path: Path) -> bool:
    """Returns whether any rule reads the file a path names."""
    return any(is_kind(path) for is_kind in K_FILE_KINDS)


def _files_matching(
    files: Sequence[SourceFile], is_kind: Callable[[Path], bool]
) -> Iterator[SourceFile]:
    """Yields the files whose path the predicate accepts."""
    return (file for file in files if is_kind(file.path))


def rule_unsized_cpp_suite(files: Sequence[SourceFile]) -> Iterator[Issue]:
    """Yields an issue for each C++ test suite whose name opens with no size."""
    for file in _files_matching(files, _is_cpp_test):
        for match in K_CPP_SUITE.finditer(file.text):
            if not match.group(1).startswith((*K_CPP_PREFIXES, K_BENCHMARK_PREFIX)):
                yield Issue(
                    file.path,
                    line_of(file.text, match.start()),
                    f"suite {match.group(1)} opens with none of {', '.join(K_CPP_PREFIXES)}",
                )


def _marks(node: ast.expr) -> set[str]:
    """Returns the pytest marks an expression names, one mark or a list of them."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return {ast.unparse(item).rsplit(".", 1)[-1] for item in items}


def _sizes_of(decorators: list[ast.expr]) -> set[str]:
    """Returns the sizes a list of decorators names."""
    return {name for decorator in decorators for name in _marks(decorator)} & set(K_SIZES)


def _file_sizes(module: ast.Module) -> set[str]:
    """Returns the sizes a module sets for all its tests, by `pytestmark` or `K_TEST_SIZE`."""
    sizes: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "pytestmark":
                sizes |= _marks(node.value)
            elif target.id == "K_TEST_SIZE" and isinstance(node.value, ast.Constant):
                sizes.add(str(node.value.value))
    return sizes & set(K_SIZES)


def _is_python_test_node(node: ast.stmt) -> bool:
    """Returns whether a top-level statement is a test function or a test class."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test_")
    return isinstance(node, ast.ClassDef) and is_test_class(node)


def rule_unsized_python_test(files: Sequence[SourceFile]) -> Iterator[Issue]:
    """Yields an issue for each pytest test whose file, class or function carries no size."""
    for file in _files_matching(files, _is_python_test):
        module = ast.parse(file.text)
        if _file_sizes(module):
            continue
        for node in module.body:
            if not _is_python_test_node(node):
                continue
            is_benchmark = node.name.removeprefix("Test").startswith(K_BENCHMARK_PREFIX)
            sized = class_size(node) if isinstance(node, ast.ClassDef) else _sizes_of(node.decorator_list)
            if not is_benchmark and not sized:
                yield Issue(
                    file.path,
                    node.lineno,
                    f"{node.name} carries no size, and the file sets none",
                )


def rule_unsized_typescript_suite(files: Sequence[SourceFile]) -> Iterator[Issue]:
    """Yields an issue for each top-level describe block whose name opens with no size."""
    for file in _files_matching(files, _is_typescript_test):
        for match in K_TS_DESCRIBE.finditer(file.text):
            if not match.group(2).startswith((*K_TS_PREFIXES, K_BENCHMARK_PREFIX)):
                yield Issue(
                    file.path,
                    line_of(file.text, match.start()),
                    f"describe '{match.group(2)}' opens with none of {', '.join(K_TS_PREFIXES)}",
                )


def rule_unsized_csharp_fixture(files: Sequence[SourceFile]) -> Iterator[Issue]:
    """Yields an issue for each NUnit fixture class with no size category."""
    for file in _files_matching(files, _is_csharp_source):
        text = file.text
        declarations = list(K_CS_CLASS.finditer(text))
        for index, match in enumerate(declarations):
            end = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
            name = match.group(3)
            is_fixture = "TestFixture" in match.group(1) \
                or K_CS_TEST_ATTRIBUTE.search(text, match.end(), end)
            is_abstract = "abstract" in match.group(2).split()
            if not is_fixture or is_abstract or name.startswith(K_BENCHMARK_PREFIX):
                continue
            if not set(K_CS_CATEGORY.findall(match.group(1))) & set(K_SIZES):
                opening = match.start(1) if match.group(1) else match.start(3)
                yield Issue(
                    file.path,
                    line_of(text, opening),
                    f"fixture {name} carries no [Category(\"small\")], medium or large",
                )


def collect(paths: list[Path]) -> Iterable[tuple[SourceFile, ...]]:
    """Returns the test files the rules read, from each given folder."""
    return [read_files(discover(path).files, is_sized_file) for path in paths if path.is_dir()]


K_CHECKER = Checker(
    description="Checks that every test writes its size in its source.",
    collect=collect,
    rules={
        "rule_unsized_cpp_suite": rule_unsized_cpp_suite,
        "rule_unsized_python_test": rule_unsized_python_test,
        "rule_unsized_typescript_suite": rule_unsized_typescript_suite,
        "rule_unsized_csharp_fixture": rule_unsized_csharp_fixture,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

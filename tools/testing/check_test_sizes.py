# check_test_sizes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every test writes its size in its source, from `testing`.

Usage: python tools/testing/check_test_sizes.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402

K_SIZES = ("small", "medium", "large")
K_TEST_FOLDERS = ("tests", "cli/tests")
K_CPP_SUFFIXES = frozenset({".cpp", ".cc", ".cxx"})
K_CPP_SUITE = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*(\w+)\s*,", re.MULTILINE)
K_CPP_PREFIXES = tuple(f"{size.capitalize()}_" for size in K_SIZES)
K_TS_DESCRIBE = re.compile(r"^describe\s*\(\s*([\"'`])(.*?)\1", re.MULTILINE)
K_TS_PREFIXES = tuple(f"{size}:" for size in K_SIZES)


@dataclass(frozen=True, slots=True)
class TestTree:
    """Holds the test files of one repository that git tracks."""

    root: Path
    files: tuple[Path, ...]

    def of(self, test: callable) -> Iterator[Path]:
        """Yields the files whose name the predicate accepts."""
        return (path for path in self.files if test(path))


def discover(root: Path) -> TestTree:
    """Returns the test files under the test folders of a repository."""
    found: list[Path] = []
    for name in K_TEST_FOLDERS:
        folder = root / name
        if folder.is_dir():
            found.extend(path for path in sorted(folder.rglob("*")) if path.is_file())
    return TestTree(root, tuple(drop_ignored(root, found)))


def collect(paths: list[Path]) -> Iterable[TestTree]:
    """Returns the test tree of each given folder."""
    return [discover(path) for path in paths if path.is_dir()]


def _line(text: str, offset: int) -> int:
    """Returns the line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def rule_unsized_cpp_suite(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each C++ test suite whose name opens with no size."""
    for path in tree.of(lambda path: path.suffix in K_CPP_SUFFIXES and path.name.startswith("test_")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_CPP_SUITE.finditer(text):
            if not match.group(1).startswith(K_CPP_PREFIXES):
                yield Issue(path, _line(text, match.start()),
                            f"suite {match.group(1)} opens with none of {', '.join(K_CPP_PREFIXES)}")


def _marks(node: ast.expr) -> set[str]:
    """Returns the pytest marks an expression names, such as `pytest.mark.small` or a list of them."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return {ast.unparse(item).rsplit(".", 1)[-1] for item in items}


def _sizes_of(decorators: list[ast.expr]) -> set[str]:
    """Returns the sizes a list of decorators names."""
    return {name for decorator in decorators for name in _marks(decorator)} & set(K_SIZES)


def rule_unsized_python_test(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each pytest test whose file, class or function carries no size."""
    for path in tree.of(lambda path: path.suffix == ".py" and path.name.startswith("test_")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        file_sizes = {
            name
            for node in module.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
            for name in _marks(node.value)
        } & set(K_SIZES)
        if file_sizes:
            continue
        for node in module.body:
            is_test = (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")) \
                or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            if is_test and not _sizes_of(node.decorator_list):
                yield Issue(path, node.lineno, f"{node.name} carries no size, and the file sets none")


def rule_unsized_typescript_suite(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each top-level TypeScript describe block whose name opens with no size."""
    for path in tree.of(lambda path: path.name.endswith((".test.ts", ".test.tsx"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_TS_DESCRIBE.finditer(text):
            if not match.group(2).startswith(K_TS_PREFIXES):
                yield Issue(path, _line(text, match.start()),
                            f"describe '{match.group(2)}' opens with none of {', '.join(K_TS_PREFIXES)}")


K_CHECKER = Checker(
    description="Checks that every test writes its size in its source.",
    collect=collect,
    rules={
        "rule_unsized_cpp_suite": rule_unsized_cpp_suite,
        "rule_unsized_python_test": rule_unsized_python_test,
        "rule_unsized_typescript_suite": rule_unsized_typescript_suite,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

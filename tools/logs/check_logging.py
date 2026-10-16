# check_logging.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks the log calls, categories and console writes of library code, from `logging`.

Usage: python tools/logs/check_logging.py [repository root] [--report] [--rule NAME]
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
from settings import K_CONSOLE_PATHS, K_LOG_PREFIX  # noqa: E402

K_SOURCE_FOLDERS = ("modules", "source", "cli")
K_CPP_SUFFIXES = frozenset({".hpp", ".h", ".cpp", ".cc", ".cxx", ".inl"})
K_LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
K_CPP_CONSOLE = re.compile(r"std::(?:cout|cerr|clog)\b|(?<![\w.])(?:f?printf|puts)\s*\(")
K_PY_CONSOLE = re.compile(r"(?<![\w.])(?<!def )print\s*\(")
K_CPP_EMPTY_CATCH = re.compile(r"catch\s*\([^)]*\)\s*\{\s*\}")
K_CATEGORY = re.compile(r"K_(\w+)_LOG_CATEGORY")
K_PY_ROOT_CALL = re.compile(r"(?<![\w.])logging\.(?:debug|info|warning|error|critical|exception|log)\s*\(")
K_PY_NAMED_LOGGER = re.compile(r"getLogger\s*\((?!\s*__name__\s*\))")
K_PY_FSTRING_CALL = re.compile(r"\b_?log\w*\.(?:trace|debug|info|warning|error|critical|exception)\s*\(\s*f[\"']")
K_COMMENT_STARTS = ("//", "/*", "*", "#")


@dataclass(frozen=True, slots=True)
class SourceTree:
    """Holds the library source files of one repository that git tracks, tests left out."""

    root: Path
    files: tuple[Path, ...]

    def of(self, suffixes: frozenset[str]) -> Iterator[Path]:
        """Yields the files with one of the given suffixes."""
        return (path for path in self.files if path.suffix in suffixes)

    def may_print(self, path: Path) -> bool:
        """Returns whether settings name the file as one that writes to the console."""
        relative = path.relative_to(self.root).as_posix()
        return any(relative == entry or relative.startswith(entry.rstrip("/") + "/") for entry in K_CONSOLE_PATHS)


def _is_test(path: Path, root: Path) -> bool:
    """Returns whether a file is a test or lives under a tests folder."""
    parts = path.relative_to(root).parts
    return "tests" in parts or path.name.startswith("test_")


def discover(root: Path) -> SourceTree:
    """Returns the library source files under the source folders of a repository."""
    found: list[Path] = []
    for name in K_SOURCE_FOLDERS:
        folder = root / name
        if folder.is_dir():
            found.extend(path for path in sorted(folder.rglob("*"))
                         if path.is_file() and not _is_test(path, root))
    return SourceTree(root, tuple(drop_ignored(root, found)))


def collect(paths: list[Path]) -> Iterable[SourceTree]:
    """Returns the source tree of each given folder."""
    return [discover(path) for path in paths if path.is_dir()]


def _code_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yields the number and text of each line that does not open with a comment."""
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.lstrip().startswith(K_COMMENT_STARTS):
            yield number, line


def _line(text: str, offset: int) -> int:
    """Returns the line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def _log_call() -> re.Pattern[str] | None:
    """Returns the pattern of a C++ log call and its first argument, or None with no prefix set."""
    if not K_LOG_PREFIX:
        return None
    return re.compile(rf"\b{K_LOG_PREFIX}_LOG_(?:{'|'.join(K_LEVELS)})\s*\(\s*([^,)]*)")


def _calls(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    """Yields each log call outside a preprocessor or comment line, so the logger's own macros are left out."""
    for match in pattern.finditer(text):
        start = text.rfind("\n", 0, match.start()) + 1
        before = text[start:match.start()]
        if not before.lstrip().startswith(K_COMMENT_STARTS) and "//" not in before:
            yield match


def _module(path: Path, root: Path) -> tuple[Path, str] | None:
    """Returns the folder and name of the module `modules/<tier>/<module>/` a file is in."""
    parts = path.relative_to(root).parts
    if len(parts) > 3 and parts[0] == "modules":
        return root.joinpath(*parts[:3]), parts[2]
    return None


def rule_console_output(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each console write in library code that settings do not allow."""
    for path in tree.of(K_CPP_SUFFIXES | {".py"}):
        if tree.may_print(path):
            continue
        pattern = K_PY_CONSOLE if path.suffix == ".py" else K_CPP_CONSOLE
        for number, line in _code_lines(path):
            if pattern.search(line):
                yield Issue(path, number, "library code writes to the console; use the logger")


def rule_swallowed_error(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each empty catch and each except whose body is only pass."""
    for path in tree.of(K_CPP_SUFFIXES):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_CPP_EMPTY_CATCH.finditer(text):
            yield Issue(path, _line(text, match.start()), "an empty catch hides the failure")
    for path in tree.of(frozenset({".py"})):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ExceptHandler) and all(isinstance(item, ast.Pass) for item in node.body):
                yield Issue(path, node.lineno, "an except whose body is only pass hides the failure")


def rule_uncategorized_call(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each C++ log call whose first argument is not a log category."""
    pattern = _log_call()
    for path in tree.of(K_CPP_SUFFIXES) if pattern else ():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _calls(pattern, text):
            if not K_CATEGORY.fullmatch(match.group(1).strip()):
                yield Issue(path, _line(text, match.start()), "the call names no K_<MODULE>_LOG_CATEGORY")


def rule_foreign_category(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each C++ log call under another module's category."""
    pattern = _log_call()
    for path in tree.of(K_CPP_SUFFIXES) if pattern else ():
        module = _module(path, tree.root)
        if module is None:
            continue
        own = f"K_{module[1].upper()}_LOG_CATEGORY"
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _calls(pattern, text):
            named = match.group(1).strip()
            if K_CATEGORY.fullmatch(named) and named != own:
                yield Issue(path, _line(text, match.start()), f"module {module[1]} logs under {named}, not {own}")


def rule_missing_category_header(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each module that logs and has no `<module>_log_category.hpp`."""
    pattern = _log_call()
    reported: set[Path] = set()
    for path in tree.of(K_CPP_SUFFIXES) if pattern else ():
        module = _module(path, tree.root)
        if module is None or module[0] in reported:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = next(_calls(pattern, text), None)
        if match and not any(module[0].rglob(f"{module[1]}_log_category.hpp")):
            reported.add(module[0])
            yield Issue(path, _line(text, match.start()), f"module {module[1]} logs and has no {module[1]}_log_category.hpp")


def rule_python_logger(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each root-logger call, named logger or f-string record in Python."""
    for path in tree.of(frozenset({".py"})):
        if tree.may_print(path):
            continue
        for number, line in _code_lines(path):
            if K_PY_ROOT_CALL.search(line):
                yield Issue(path, number, "logs on the root logger; use the module's _log")
            if K_PY_NAMED_LOGGER.search(line):
                yield Issue(path, number, "a logger not named __name__")
            if K_PY_FSTRING_CALL.search(line):
                yield Issue(path, number, "an f-string record formats when the level is off; pass %s arguments")


K_CHECKER = Checker(
    description="Checks the log calls, categories and console writes of library code.",
    collect=collect,
    rules={
        "rule_console_output": rule_console_output,
        "rule_swallowed_error": rule_swallowed_error,
        "rule_uncategorized_call": rule_uncategorized_call,
        "rule_foreign_category": rule_foreign_category,
        "rule_missing_category_header": rule_missing_category_header,
        "rule_python_logger": rule_python_logger,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

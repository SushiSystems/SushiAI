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
import fnmatch
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored  # noqa: E402
from checking.source_file import SourceFile, line_of, read_files  # noqa: E402
import settings  # noqa: E402
from settings import K_CONSOLE_PATHS, K_LOG_PREFIX  # noqa: E402

K_SOURCE_FOLDERS = ("modules", "source", "cli")
K_MODULES_FOLDER = "modules"
K_CPP_SUFFIXES = frozenset({".hpp", ".h", ".cpp", ".cc", ".cxx", ".inl"})
K_PY_SUFFIX = ".py"
K_READ_SUFFIXES = K_CPP_SUFFIXES | frozenset({K_PY_SUFFIX})
K_LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
K_CPP_CONSOLE = re.compile(r"std::(?:cout|cerr|clog)\b|(?<![\w.])(?:f?printf|puts)\s*\(")
K_PY_CONSOLE = re.compile(r"(?<![\w.])(?<!def )print\s*\(")
K_CPP_EMPTY_CATCH = re.compile(r"catch\s*\([^)]*\)\s*\{\s*\}")
K_CATEGORY = re.compile(r"K_(\w+)_LOG_CATEGORY")
K_PY_ROOT_CALL = re.compile(r"(?<![\w.])logging\.(?:debug|info|warning|error|critical|exception|log)\s*\(")
K_PY_NAMED_LOGGER = re.compile(r"getLogger\s*\((?!\s*__name__\s*\))")
K_PY_FSTRING_CALL = re.compile(r"\b_?log\w*\.(?:trace|debug|info|warning|error|critical|exception)\s*\(\s*f[\"']")
K_COMMENT_STARTS = ("//", "/*", "*", "#")
K_LOGGER_PREFIX: str = getattr(settings, "K_LOGGER_MACRO_PREFIX", "")
K_LOGGER_NAMESPACE: str = getattr(settings, "K_LOGGER_NAMESPACE", "")
K_LOGGER_C_PREFIX: str = getattr(settings, "K_LOGGER_C_PREFIX", "")


@dataclass(frozen=True, slots=True)
class LogSettings:
    """Holds the settings the rules read: the macro prefixes, the console paths and the shared logger names."""

    log_prefix: str
    console_paths: tuple[str, ...]
    logger_prefix: str
    logger_namespace: str
    logger_c_prefix: str


@dataclass(frozen=True, slots=True)
class SourceTree:
    """Holds one repository's library sources as text, its settings and the names below its module folders.

    The shell reads the files git tracks, tests left out. Each module folder that holds a C++ source
    maps to the names of everything below it, which the category-header rule checks.
    """

    root: Path
    log_settings: LogSettings
    files: tuple[SourceFile, ...]
    module_entries: Mapping[Path, frozenset[str]]

    def of(self, suffixes: frozenset[str]) -> Iterator[SourceFile]:
        """Yields the files with one of the given suffixes."""
        return (source for source in self.files if source.path.suffix in suffixes)

    def may_print(self, path: Path) -> bool:
        """Returns whether settings name the file as one that writes to the console."""
        relative = path.relative_to(self.root).as_posix()
        return any(relative == entry or relative.startswith(entry.rstrip("/") + "/")
                   for entry in self.log_settings.console_paths)


def module_folder(path: Path, root: Path) -> Path | None:
    """Returns the folder of the module `modules/<tier>/<module>/` a file is in, or None outside one."""
    parts = path.relative_to(root).parts
    return root.joinpath(*parts[:3]) if len(parts) > 3 and parts[0] == K_MODULES_FOLDER else None


def _code_lines(source: SourceFile) -> Iterator[tuple[int, str]]:
    """Yields the number and text of each line that does not open with a comment."""
    for number, line in enumerate(source.text.splitlines(), 1):
        if not line.lstrip().startswith(K_COMMENT_STARTS):
            yield number, line


def _log_call(prefix: str) -> re.Pattern[str] | None:
    """Returns the pattern of a C++ log call and its first argument, or None with no prefix set."""
    if not prefix:
        return None
    return re.compile(rf"\b{prefix}_LOG_(?:{'|'.join(K_LEVELS)})\s*\(\s*([^,)]*)")


def _calls(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    """Yields each log call outside a preprocessor or comment line, so the logger's own macros are left out."""
    for match in pattern.finditer(text):
        start = text.rfind("\n", 0, match.start()) + 1
        before = text[start:match.start()]
        if not before.lstrip().startswith(K_COMMENT_STARTS) and "//" not in before:
            yield match


def _has_category_header(tree: SourceTree, folder: Path) -> bool:
    """Returns whether a module folder holds an entry named `<module>_log_category.hpp`."""
    header = f"{folder.name}_log_category.hpp"
    return any(fnmatch.fnmatch(entry, header) for entry in tree.module_entries.get(folder, frozenset()))


def _reexport_patterns(namespace: str, c_prefix: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Returns the patterns of a re-exported logger name and of a function that only forwards to the logger."""
    escaped = re.escape(namespace)
    names = rf"{escaped}::\w+" + (rf"|{re.escape(c_prefix)}\w+" if c_prefix else "")
    reexport = re.compile(rf"\busing\s+(?:\w+\s*=\s*)?(?:::)?{escaped}::\w+")
    shim = re.compile(rf"\binline\b[^{{}};]*\)\s*\{{\s*(?:return\s+)?(?:::)?(?:{names})\s*\([^;]*\)\s*;\s*\}}")
    return reexport, shim


def rule_console_output(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each console write in library code that settings do not allow."""
    for source in tree.of(K_READ_SUFFIXES):
        if tree.may_print(source.path):
            continue
        pattern = K_PY_CONSOLE if source.path.suffix == K_PY_SUFFIX else K_CPP_CONSOLE
        for number, line in _code_lines(source):
            if pattern.search(line):
                yield Issue(source.path, number, "library code writes to the console; use the logger")


def rule_swallowed_error(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each empty catch and each except whose body is only pass."""
    for source in tree.of(K_CPP_SUFFIXES):
        for match in K_CPP_EMPTY_CATCH.finditer(source.text):
            yield Issue(source.path, line_of(source.text, match.start()), "an empty catch hides the failure")
    for source in tree.of(frozenset({K_PY_SUFFIX})):
        for node in ast.walk(ast.parse(source.text)):
            if isinstance(node, ast.ExceptHandler) and all(isinstance(item, ast.Pass) for item in node.body):
                yield Issue(source.path, node.lineno, "an except whose body is only pass hides the failure")


def rule_uncategorized_call(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each C++ log call whose first argument is not a log category."""
    pattern = _log_call(tree.log_settings.log_prefix)
    for source in tree.of(K_CPP_SUFFIXES) if pattern else ():
        for match in _calls(pattern, source.text):
            if not K_CATEGORY.fullmatch(match.group(1).strip()):
                yield Issue(source.path, line_of(source.text, match.start()),
                            "the call names no K_<MODULE>_LOG_CATEGORY")


def rule_foreign_category(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each C++ log call under another module's category."""
    pattern = _log_call(tree.log_settings.log_prefix)
    for source in tree.of(K_CPP_SUFFIXES) if pattern else ():
        folder = module_folder(source.path, tree.root)
        if folder is None:
            continue
        own = f"K_{folder.name.upper()}_LOG_CATEGORY"
        for match in _calls(pattern, source.text):
            named = match.group(1).strip()
            if K_CATEGORY.fullmatch(named) and named != own:
                yield Issue(source.path, line_of(source.text, match.start()),
                            f"module {folder.name} logs under {named}, not {own}")


def rule_missing_category_header(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each module that logs and has no `<module>_log_category.hpp`."""
    pattern = _log_call(tree.log_settings.log_prefix)
    reported: set[Path] = set()
    for source in tree.of(K_CPP_SUFFIXES) if pattern else ():
        folder = module_folder(source.path, tree.root)
        if folder is None or folder in reported:
            continue
        match = next(_calls(pattern, source.text), None)
        if match and not _has_category_header(tree, folder):
            reported.add(folder)
            yield Issue(source.path, line_of(source.text, match.start()),
                        f"module {folder.name} logs and has no {folder.name}_log_category.hpp")


def rule_python_logger(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each root-logger call, named logger or f-string record in Python."""
    for source in tree.of(frozenset({K_PY_SUFFIX})):
        if tree.may_print(source.path):
            continue
        for number, line in _code_lines(source):
            if K_PY_ROOT_CALL.search(line):
                yield Issue(source.path, number, "logs on the root logger; use the module's _log")
            if K_PY_NAMED_LOGGER.search(line):
                yield Issue(source.path, number, "a logger not named __name__")
            if K_PY_FSTRING_CALL.search(line):
                yield Issue(source.path, number, "an f-string record formats when the level is off; pass %s arguments")


def rule_front_reexport(tree: SourceTree) -> Iterator[Issue]:
    """Yields an issue for each alias, using or forwarding function of a logger name in a front."""
    names = tree.log_settings
    if not names.logger_namespace or names.log_prefix in ("", names.logger_prefix):
        return
    reexport, shim = _reexport_patterns(names.logger_namespace, names.logger_c_prefix)
    namespace = names.logger_namespace
    call_site = f"{namespace}::..." + (f" or {names.logger_c_prefix}..." if names.logger_c_prefix else "")
    for source in tree.of(K_CPP_SUFFIXES):
        for pattern, what in ((reexport, f"re-exports a {namespace} name"),
                              (shim, f"forwards to {namespace}")):
            for match in pattern.finditer(source.text):
                yield Issue(source.path, line_of(source.text, match.start()),
                            f"{what}; write {call_site} at the call site")


def _is_test(path: Path, root: Path) -> bool:
    """Returns whether a file is a test or lives under a tests folder."""
    parts = path.relative_to(root).parts
    return "tests" in parts or path.name.startswith("test_")


def _read_paths(root: Path) -> list[Path]:
    """Returns the library files of the source folders that git does not ignore, in discovery order."""
    found: list[Path] = []
    for name in K_SOURCE_FOLDERS:
        folder = root / name
        if folder.is_dir():
            found.extend(path for path in sorted(folder.rglob("*"))
                         if path.is_file() and path.suffix in K_READ_SUFFIXES and not _is_test(path, root))
    return drop_ignored(root, found)


def _read_entries(root: Path, paths: Iterable[Path]) -> dict[Path, frozenset[str]]:
    """Lists the names below the module folder of each C++ source, every entry of it, not only headers."""
    entries: dict[Path, frozenset[str]] = {}
    for path in paths:
        folder = module_folder(path, root) if path.suffix in K_CPP_SUFFIXES else None
        if folder is not None and folder not in entries:
            entries[folder] = frozenset(entry.name for entry in folder.rglob("*") if entry.exists())
    return entries


def _read_tree(root: Path, log_settings: LogSettings) -> SourceTree:
    """Reads one repository's library sources and module entries into a tree."""
    paths = _read_paths(root)
    return SourceTree(root, log_settings, read_files(paths, lambda _: True), _read_entries(root, paths))


def collect(paths: list[Path]) -> Iterable[SourceTree]:
    """Returns the source tree of each given folder, read with the settings this module names."""
    log_settings = LogSettings(K_LOG_PREFIX, K_CONSOLE_PATHS, K_LOGGER_PREFIX, K_LOGGER_NAMESPACE, K_LOGGER_C_PREFIX)
    return [_read_tree(path, log_settings) for path in paths if path.is_dir()]


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
        "rule_front_reexport": rule_front_reexport,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

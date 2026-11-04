# test_check_logging.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/logs/check_logging.py on source trees held in memory, and its command line on a temporary tree."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from logs import check_logging  # noqa: E402
from logs.check_logging import (  # noqa: E402
    K_CHECKER,
    LogSettings,
    SourceTree,
    rule_console_output,
    rule_front_reexport,
    rule_foreign_category,
    rule_missing_category_header,
    rule_python_logger,
    rule_swallowed_error,
    rule_uncategorized_call,
)

K_ROOT = Path("repo")
K_SETTINGS = LogSettings(
    log_prefix="PX",
    console_paths=("cli/px/commands",),
    logger_prefix="AL",
    logger_namespace="AcmeLog",
    logger_c_prefix="al_",
)
K_NO_PREFIX = LogSettings(log_prefix="", console_paths=("cli/px/commands",), logger_prefix="AL",
                          logger_namespace="AcmeLog", logger_c_prefix="al_")


def _tree(files: dict[str, str], settings_: LogSettings = K_SETTINGS,
          headers: dict[str, frozenset[str]] | None = None) -> SourceTree:
    """Returns the tree the shell builds from these files, given as text, and the names below their module folders."""
    return SourceTree(
        K_ROOT,
        settings_,
        tuple(SourceFile(K_ROOT / name, text) for name, text in files.items()),
        {K_ROOT / folder: names for folder, names in (headers or {}).items()},
    )


def _found(rule, files: dict[str, str], **kwargs) -> list[tuple[str, int]]:
    """Returns the file name and line of each issue a rule yields for the files."""
    return [(issue.path.name, issue.line) for issue in rule(_tree(files, **kwargs))]


class ConsoleOutputTest(unittest.TestCase):
    """Covers the console writes library code may not make, outside the paths settings allow."""
    K_TEST_SIZE = "small"

    def test_reports_console_writes_outside_comments_and_allowed_paths(self) -> None:
        files = {
            "modules/core/ecs/source/world.cpp":
                "// printf(\"x\")\nstd::cout << 1;\nstd::snprintf(b, 4, \"x\");\nfprintf(stderr, \"x\");\n"
                "std::fprintf(stderr, \"x\");\n",
            "source/px/store.py": "print('x')\nconsole.print('y')\nclass Sink:\n    def print(self) -> None: ...\n",
            "cli/px/commands/build.py": "print('ok')\n",
        }
        self.assertEqual(_found(rule_console_output, files),
                         [("world.cpp", 2), ("world.cpp", 4), ("world.cpp", 5), ("store.py", 1)])


class SwallowedErrorTest(unittest.TestCase):
    """Covers the empty catch and the except that only passes."""
    K_TEST_SIZE = "small"

    def test_reports_an_empty_catch_and_an_except_that_only_passes(self) -> None:
        files = {
            "modules/core/ecs/source/world.cpp": "try { f(); } catch (...) {}\ntry { g(); } catch (const E& e) { log(e); }\n",
            "source/px/store.py": "try:\n    f()\nexcept OSError:\n    pass\ntry:\n    g()\nexcept OSError:\n    raise\n",
        }
        self.assertEqual(_found(rule_swallowed_error, files), [("world.cpp", 1), ("store.py", 3)])


class CategoryCallTest(unittest.TestCase):
    """Covers the C++ log calls and the category each one names."""
    K_TEST_SIZE = "small"

    def test_reports_a_call_without_a_category(self) -> None:
        files = {
            "modules/core/ecs/source/world.cpp":
                "PX_LOG_WARN(K_ECS_LOG_CATEGORY, \"a {}\", 1);\nPX_LOG_ERROR(\"b {}\", 2);\n"
                "#define PX_LOG_INFO(...) PX_LOG_AT(__VA_ARGS__)\n"
                " * @code PX_LOG_WARN(\"c\"); @endcode\nf(); // PX_LOG_WARN(\"d\")\n",
        }
        self.assertEqual(_found(rule_uncategorized_call, files), [("world.cpp", 2)])

    def test_reports_a_call_under_another_modules_category(self) -> None:
        files = {
            "modules/core/ecs/source/world.cpp":
                "PX_LOG_INFO(K_ECS_LOG_CATEGORY, \"a\");\nPX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"b\");\n",
        }
        self.assertEqual(_found(rule_foreign_category, files), [("world.cpp", 2)])

    def test_no_prefix_turns_the_cpp_call_rules_off(self) -> None:
        files = {"modules/core/ecs/source/world.cpp": "PX_LOG_ERROR(\"b\");\nPX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"b\");\n"}
        self.assertEqual(_found(rule_uncategorized_call, files, settings_=K_NO_PREFIX), [])
        self.assertEqual(_found(rule_foreign_category, files, settings_=K_NO_PREFIX), [])


class CategoryHeaderTest(unittest.TestCase):
    """Covers a module that logs and has no `<module>_log_category.hpp` below its folder."""
    K_TEST_SIZE = "small"

    def test_reports_a_module_that_logs_without_its_category_header(self) -> None:
        files = {
            "modules/core/ecs/source/world.cpp": "PX_LOG_INFO(K_ECS_LOG_CATEGORY, \"a\");\n",
            "modules/core/memory/source/pool.cpp": "PX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"a\");\n",
        }
        headers = {"modules/core/memory": frozenset({"source", "memory_log_category.hpp"})}
        self.assertEqual(_found(rule_missing_category_header, files, headers=headers), [("world.cpp", 1)])


class PythonLoggerTest(unittest.TestCase):
    """Covers the root-logger call, the named logger and the f-string record in Python."""
    K_TEST_SIZE = "small"

    def test_reports_root_named_and_fstring_python_logging(self) -> None:
        files = {
            "source/px/store.py":
                "import logging\n_log = logging.getLogger(__name__)\nlogging.info('x')\n"
                "other = logging.getLogger('px')\n_log.warning(f'{x}')\n_log.warning('%s', x)\n",
        }
        self.assertEqual(_found(rule_python_logger, files), [("store.py", 3), ("store.py", 4), ("store.py", 5)])


class FrontReexportTest(unittest.TestCase):
    """Covers a front that re-exports the shared logger's names or forwards to them."""
    K_TEST_SIZE = "small"

    def test_reports_a_front_that_re_exports_the_logger(self) -> None:
        files = {  # in the path order the shell reads them
            "modules/foundation/log/include/PX/log/log_c_sink.h":
                "static inline void px_set_log_sink(AcmeLogSinkFunction f, void* u)\n{\n"
                "    al_set_log_sink(f, u);\n}\n",
            "modules/foundation/log/include/PX/log/log_level.hpp":
                "using LogLevel = AcmeLog::LogLevel;\nusing AcmeLog::make_log_category;\n"
                "inline constexpr AcmeLog::LogProject K_PX_LOG_PROJECT{\"px\", \"PX_LOG_LEVEL\"};\n",
            "modules/foundation/log/source/capture.cpp":
                "Capture::~Capture()\n{\n    AcmeLog::set_log_sink(AcmeLog::LogSink());\n}\n",
        }
        self.assertEqual(_found(rule_front_reexport, files),
                         [("log_c_sink.h", 1), ("log_level.hpp", 1), ("log_level.hpp", 2)])


class ShellTest(unittest.TestCase):
    """Runs the command line on a tree on disk, so the shell's discovery and settings reach every rule."""
    K_TEST_SIZE = "medium"

    def test_cli_reads_the_source_folders_and_reports_each_finding(self) -> None:
        """Prints the findings of the source folders, skips the tests and the console paths, and exits non-zero."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in {
                "modules/core/ecs/source/world.cpp": "std::cout << 1;\nPX_LOG_INFO(\"a\");\n",
                "modules/core/memory/source/pool.cpp": "PX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"a\");\n",
                "modules/core/memory/source/memory_log_category.hpp": "\n",
                "source/px/store.py": "print('x')\n",
                "cli/px/commands/build.py": "print('ok')\n",
                "tests/core/test_world.cpp": "std::cout << 1;\n",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            output = io.StringIO()
            with mock.patch.object(check_logging, "K_LOG_PREFIX", "PX"), \
                    mock.patch.object(check_logging, "K_CONSOLE_PATHS", ("cli/px/commands",)), \
                    mock.patch.object(check_logging, "K_LOGGER_PREFIX", "AL"), \
                    mock.patch.object(check_logging, "K_LOGGER_NAMESPACE", "AcmeLog"), \
                    mock.patch.object(check_logging, "K_LOGGER_C_PREFIX", "al_"), \
                    contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [folder])
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue().splitlines(), [
                "modules/core/ecs/source/world.cpp:1: rule_console_output: "
                "library code writes to the console; use the logger",
                "modules/core/ecs/source/world.cpp:2: rule_missing_category_header: "
                "module ecs logs and has no ecs_log_category.hpp",
                "modules/core/ecs/source/world.cpp:2: rule_uncategorized_call: "
                "the call names no K_<MODULE>_LOG_CATEGORY",
                "source/px/store.py:1: rule_console_output: library code writes to the console; use the logger",
            ])


if __name__ == "__main__":
    unittest.main()

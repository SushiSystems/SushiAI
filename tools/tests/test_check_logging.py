# test_check_logging.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/logs/check_logging.py on source trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logs import check_logging  # noqa: E402
from logs.check_logging import (  # noqa: E402
    discover,
    rule_console_output,
    rule_foreign_category,
    rule_missing_category_header,
    rule_python_logger,
    rule_swallowed_error,
    rule_uncategorized_call,
)


class LoggingTest(unittest.TestCase):
    """Builds a repository for each test, with PX as its macro prefix."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        patches = [mock.patch.object(check_logging, "K_LOG_PREFIX", "PX"),
                   mock.patch.object(check_logging, "K_CONSOLE_PATHS", ("cli/px/commands",))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, relative: str, text: str) -> None:
        """Writes a file below the repository root."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def found(self, rule) -> list[tuple[str, int]]:
        """Returns the file name and line of each issue a rule yields."""
        return [(issue.path.name, issue.line) for issue in rule(discover(self.root))]

    def test_reports_console_writes_outside_the_allowed_paths_and_tests(self) -> None:
        self.write("modules/core/ecs/source/world.cpp",
                   "// printf(\"x\")\nstd::cout << 1;\nstd::snprintf(b, 4, \"x\");\nfprintf(stderr, \"x\");\n"
                   "std::fprintf(stderr, \"x\");\n")
        self.write("source/px/store.py", "print('x')\nconsole.print('y')\nclass Sink:\n    def print(self) -> None: ...\n")
        self.write("cli/px/commands/build.py", "print('ok')\n")
        self.write("tests/core/test_world.cpp", "std::cout << 1;\n")
        self.assertEqual(self.found(rule_console_output),
                         [("world.cpp", 2), ("world.cpp", 4), ("world.cpp", 5), ("store.py", 1)])

    def test_reports_an_empty_catch_and_an_except_that_only_passes(self) -> None:
        self.write("modules/core/ecs/source/world.cpp",
                   "try { f(); } catch (...) {}\ntry { g(); } catch (const E& e) { log(e); }\n")
        self.write("source/px/store.py",
                   "try:\n    f()\nexcept OSError:\n    pass\ntry:\n    g()\nexcept OSError:\n    raise\n")
        self.assertEqual(self.found(rule_swallowed_error), [("world.cpp", 1), ("store.py", 3)])

    def test_reports_a_call_without_a_category(self) -> None:
        self.write("modules/core/ecs/source/world.cpp",
                   "PX_LOG_WARN(K_ECS_LOG_CATEGORY, \"a {}\", 1);\nPX_LOG_ERROR(\"b {}\", 2);\n"
                   "#define PX_LOG_INFO(...) PX_LOG_AT(__VA_ARGS__)\n"
                   " * @code PX_LOG_WARN(\"c\"); @endcode\nf(); // PX_LOG_WARN(\"d\")\n")
        self.assertEqual(self.found(rule_uncategorized_call), [("world.cpp", 2)])

    def test_reports_a_call_under_another_modules_category(self) -> None:
        self.write("modules/core/ecs/source/world.cpp",
                   "PX_LOG_INFO(K_ECS_LOG_CATEGORY, \"a\");\nPX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"b\");\n")
        self.assertEqual(self.found(rule_foreign_category), [("world.cpp", 2)])

    def test_reports_a_module_that_logs_without_its_category_header(self) -> None:
        self.write("modules/core/ecs/source/world.cpp", "PX_LOG_INFO(K_ECS_LOG_CATEGORY, \"a\");\n")
        self.write("modules/core/memory/source/pool.cpp", "PX_LOG_INFO(K_MEMORY_LOG_CATEGORY, \"a\");\n")
        self.write("modules/core/memory/source/memory_log_category.hpp", "\n")
        self.assertEqual(self.found(rule_missing_category_header), [("world.cpp", 1)])

    def test_no_prefix_turns_the_cpp_call_rules_off(self) -> None:
        self.write("modules/core/ecs/source/world.cpp", "PX_LOG_ERROR(\"b\");\n")
        with mock.patch.object(check_logging, "K_LOG_PREFIX", ""):
            self.assertEqual(self.found(rule_uncategorized_call), [])

    def test_reports_root_named_and_fstring_python_logging(self) -> None:
        self.write("source/px/store.py",
                   "import logging\n_log = logging.getLogger(__name__)\nlogging.info('x')\n"
                   "other = logging.getLogger('px')\n_log.warning(f'{x}')\n_log.warning('%s', x)\n")
        self.assertEqual(self.found(rule_python_logger), [("store.py", 3), ("store.py", 4), ("store.py", 5)])


if __name__ == "__main__":
    unittest.main()

# test_check_device_sources.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rules of tools/layering/check_device_sources.py on trees held in memory, and its disk reader once."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_device_sources import (  # noqa: E402
    DiskFiles,
    MemoryFiles,
    UnitTree,
    rule_device_source_missing,
    rule_device_source_outside_module,
    rule_device_source_unneeded,
)

K_KERNEL_INCLUDE = "#include <SushiBLAS/elementwise/dispatch.hpp>"
K_ROOT = Path("repo")
K_MODULE_FILES = {
    "modules/core/optim/include/SushiAI/optim/step.hpp": K_KERNEL_INCLUDE,
    "modules/core/optim/include/SushiAI/optim/plain.hpp": "int plain();",
    "modules/core/optim/source/sgd.cpp": "#include <SushiAI/optim/step.hpp>",
    "modules/core/optim/source/schedule.cpp": "#include <SushiAI/optim/plain.hpp>",
}
K_MISSING_MESSAGE = (
    "sgd.cpp reaches a kernel through SushiAI/optim/step.hpp -> SushiBLAS/elementwise/dispatch.hpp "
    "and is not in DEVICE_SOURCES of optim"
)


def _build_file(sources: list[str], device: list[str]) -> str:
    """Returns a module build file listing the given sources and device sources."""
    lines = ["sushiai_add_module(NAME optim TIER core", "    SOURCES"]
    lines += [f"        {name}" for name in sources]
    if device:
        lines.append("    DEVICE_SOURCES")
        lines += [f"        {name}" for name in device]
    lines.append("    PUBLIC_DEPENDS dtype)")
    return "\n".join(lines)


def _tree(
    sources: list[str], device: list[str], extra: dict[str, str] | None = None
) -> UnitTree:
    """Returns the unit tree of a repository held in memory: the optim module, its build file and extra files."""
    files = {
        **K_MODULE_FILES,
        "modules/core/optim/CMakeLists.txt": _build_file(sources, device),
        **(extra or {}),
    }
    return UnitTree(K_ROOT, files=MemoryFiles(K_ROOT, files))


def _messages(rule, tree: UnitTree) -> list[str]:
    """Returns the message of each issue a rule yields."""
    return [issue.message for issue in rule(tree)]


def _test_program(device: list[str]) -> dict[str, str]:
    """Returns a test unit that records a kernel pair and the build file of the test program listing it."""
    lines = ["sushiai_add_test_program(OWNER optim", "    SOURCES", "        test_step.cpp"]
    if device:
        lines += ["    DEVICE_SOURCES"] + [f"        {name}" for name in device]
    lines.append("    DEPENDS dtype)")
    return {
        "tests/core/optim/test_step.cpp": "void f() { recorder.record_late_pair(a, b); }",
        "tests/core/optim/CMakeLists.txt": "\n".join(lines),
    }


class PassingTreeTest(unittest.TestCase):
    """Covers a tree whose device list is exactly the units that reach a kernel."""

    K_TEST_SIZE = "small"

    def test_reports_nothing_when_the_list_matches(self) -> None:
        """Gives no issue under any of the three rules."""
        tree = _tree(["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"])
        rules = (
            ("missing", rule_device_source_missing),
            ("unneeded", rule_device_source_unneeded),
            ("outside_module", rule_device_source_outside_module),
        )
        for name, rule in rules:
            with self.subTest(rule=name):
                self.assertEqual(_messages(rule, tree), [])


class RuleDeviceSourceMissingTest(unittest.TestCase):
    """Covers the rule that every unit reaching a kernel is listed."""

    K_TEST_SIZE = "small"

    def test_reports_a_unit_that_reaches_a_kernel(self) -> None:
        """Names a unit that reaches a kernel through a header and a unit that launches one itself."""
        own = {"modules/core/optim/source/own.cpp": "void f() { queue.parallel_for(range, body); }"}
        rows = (
            ("header", _tree(["source/sgd.cpp", "source/schedule.cpp"], []), K_MISSING_MESSAGE),
            (
                "launch",
                _tree(["source/own.cpp"], [], own),
                "own.cpp reaches a kernel through a kernel launch and is not in DEVICE_SOURCES of optim",
            ),
        )
        for name, tree, message in rows:
            with self.subTest(reach=name):
                self.assertEqual(_messages(rule_device_source_missing, tree), [message])

    def test_ignores_a_kernel_name_in_a_comment(self) -> None:
        """Gives no issue for a unit that names a launch only in a comment."""
        note = {"modules/core/optim/source/note.cpp": "// parallel_for( is not called here"}
        tree = _tree(["source/note.cpp"], [], note)
        self.assertEqual(_messages(rule_device_source_missing, tree), [])


class RuleDeviceSourceUnneededTest(unittest.TestCase):
    """Covers the rule that a listed unit reaches a kernel."""

    K_TEST_SIZE = "small"

    def test_reports_a_listed_unit_that_reaches_no_kernel(self) -> None:
        """Names the listed unit that holds no launch and no kernel header."""
        tree = _tree(
            ["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp", "source/schedule.cpp"]
        )
        self.assertEqual(
            _messages(rule_device_source_unneeded, tree),
            ["schedule.cpp is in DEVICE_SOURCES and reaches no kernel; list it in SOURCES only"],
        )


class RuleDeviceSourceOutsideModuleTest(unittest.TestCase):
    """Covers the rule that no test or application source reaches a kernel."""

    K_TEST_SIZE = "small"

    def test_reports_a_test_and_an_application_that_reach_a_kernel(self) -> None:
        """Names each of the two units and leaves the test that reaches none."""
        outside = {
            "tests/core/optim/test_step.cpp": "#include <SushiAI/optim/step.hpp>",
            "applications/tool/main.cpp": K_KERNEL_INCLUDE,
            "tests/core/optim/test_plain.cpp": "#include <SushiAI/optim/plain.hpp>",
        }
        tree = _tree(["source/schedule.cpp"], [], outside)
        issues = list(rule_device_source_outside_module(tree))
        self.assertEqual(sorted(issue.path.name for issue in issues), ["main.cpp", "test_step.cpp"])


class TestProgramDeviceSourcesTest(unittest.TestCase):
    """Covers test units that instantiate a kernel through a task recorder call."""

    K_TEST_SIZE = "small"

    def test_a_listed_test_unit_that_records_a_kernel_is_not_reported_outside_its_module(self) -> None:
        """Gives no issue under the outside-module rule for a unit its test program lists."""
        tree = _tree(
            ["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"], _test_program(["test_step.cpp"])
        )
        self.assertEqual(_messages(rule_device_source_outside_module, tree), [])

    def test_a_listed_test_unit_that_records_a_kernel_is_not_reported_as_unneeded(self) -> None:
        """Gives no issue under the unneeded rule for the DEVICE_SOURCES entry of that test program."""
        tree = _tree(
            ["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"], _test_program(["test_step.cpp"])
        )
        self.assertEqual(_messages(rule_device_source_unneeded, tree), [])

    def test_a_test_unit_that_reaches_a_kernel_through_a_support_header_is_not_unneeded(self) -> None:
        """Follows `support/<name>.hpp` into tests/support, where a shared test header records a kernel."""
        program = _test_program(["test_step.cpp"])
        program["tests/core/optim/test_step.cpp"] = "#include <support/late_recorder.hpp>"
        program["tests/support/late_recorder.hpp"] = "void f() { recorder.record_late_pair(a, b); }"
        tree = _tree(["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"], program)
        self.assertEqual(_messages(rule_device_source_unneeded, tree), [])

    def test_an_unlisted_test_unit_that_records_a_kernel_is_reported(self) -> None:
        """Names the test unit that records a kernel pair and that no test program lists."""
        tree = _tree(["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"], _test_program([]))
        messages = _messages(rule_device_source_outside_module, tree)
        self.assertEqual(len(messages), 1)
        self.assertIn("test_step.cpp", messages[0])


class DiskFilesTest(unittest.TestCase):
    """Checks that the files on disk answer as the files in memory do."""

    K_TEST_SIZE = "medium"

    def test_answers_a_tree_as_the_files_in_memory_do(self) -> None:
        """Gives the same messages for a module whose header reaches a kernel, read from a folder."""
        files = {**K_MODULE_FILES, "modules/core/optim/CMakeLists.txt": _build_file(
            ["source/sgd.cpp", "source/schedule.cpp"], []
        )}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", encoding="utf-8")
            tree = UnitTree(root, files=DiskFiles(root))
            self.assertEqual(_messages(rule_device_source_missing, tree), [K_MISSING_MESSAGE])


if __name__ == "__main__":
    unittest.main()

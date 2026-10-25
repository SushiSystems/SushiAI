# test_check_device_sources.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/layering/check_device_sources.py on trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_device_sources import (  # noqa: E402
    UnitTree,
    rule_device_source_missing,
    rule_device_source_outside_module,
    rule_device_source_unneeded,
)

K_KERNEL_INCLUDE = "#include <SushiBLAS/elementwise/dispatch.hpp>"


def _write(root: Path, relative: str, text: str = "") -> None:
    """Writes a file and its folders below root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _build_file(sources: list[str], device: list[str]) -> str:
    """Returns a module build file listing the given sources and device sources."""
    lines = ["sushiai_add_module(NAME optim TIER core", "    SOURCES"]
    lines += [f"        {name}" for name in sources]
    if device:
        lines.append("    DEVICE_SOURCES")
        lines += [f"        {name}" for name in device]
    lines.append("    PUBLIC_DEPENDS dtype)")
    return "\n".join(lines)


class DeviceSourcesTest(unittest.TestCase):
    """Builds a temporary repository with a module whose header reaches a kernel."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        _write(self.root, "modules/core/optim/include/SushiAI/optim/step.hpp", K_KERNEL_INCLUDE)
        _write(self.root, "modules/core/optim/include/SushiAI/optim/plain.hpp", "int plain();")
        _write(self.root, "modules/core/optim/source/sgd.cpp", "#include <SushiAI/optim/step.hpp>")
        _write(self.root, "modules/core/optim/source/schedule.cpp", "#include <SushiAI/optim/plain.hpp>")

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _declare(self, sources: list[str], device: list[str]) -> UnitTree:
        """Writes the module build file and returns the tree."""
        _write(self.root, "modules/core/optim/CMakeLists.txt", _build_file(sources, device))
        return UnitTree(self.root)

    def _messages(self, rule, tree: UnitTree) -> list[str]:
        """Returns the message of each issue a rule yields."""
        return [issue.message for issue in rule(tree)]


class PassingTreeTest(DeviceSourcesTest):
    """Covers a tree whose device list is exactly the units that reach a kernel."""

    def test_reports_nothing_when_the_list_matches(self) -> None:
        tree = self._declare(["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp"])
        for rule in (rule_device_source_missing, rule_device_source_unneeded, rule_device_source_outside_module):
            self.assertEqual(self._messages(rule, tree), [])


class RuleDeviceSourceMissingTest(DeviceSourcesTest):
    """Covers the rule that every unit reaching a kernel is listed."""

    def test_reports_a_unit_that_reaches_a_kernel_through_a_header(self) -> None:
        tree = self._declare(["source/sgd.cpp", "source/schedule.cpp"], [])
        self.assertEqual(
            self._messages(rule_device_source_missing, tree),
            ["sgd.cpp reaches a kernel through SushiAI/optim/step.hpp -> SushiBLAS/elementwise/dispatch.hpp "
             "and is not in DEVICE_SOURCES of optim"])

    def test_reports_a_unit_that_launches_a_kernel_itself(self) -> None:
        _write(self.root, "modules/core/optim/source/own.cpp", "void f() { queue.parallel_for(range, body); }")
        tree = self._declare(["source/own.cpp"], [])
        self.assertEqual(len(self._messages(rule_device_source_missing, tree)), 1)

    def test_ignores_a_kernel_name_in_a_comment(self) -> None:
        _write(self.root, "modules/core/optim/source/note.cpp", "// parallel_for( is not called here")
        tree = self._declare(["source/note.cpp"], [])
        self.assertEqual(self._messages(rule_device_source_missing, tree), [])


class RuleDeviceSourceUnneededTest(DeviceSourcesTest):
    """Covers the rule that a listed unit reaches a kernel."""

    def test_reports_a_listed_unit_that_reaches_no_kernel(self) -> None:
        tree = self._declare(["source/sgd.cpp", "source/schedule.cpp"], ["source/sgd.cpp", "source/schedule.cpp"])
        self.assertEqual(self._messages(rule_device_source_unneeded, tree),
                         ["schedule.cpp is in DEVICE_SOURCES and reaches no kernel; list it in SOURCES only"])


class RuleDeviceSourceOutsideModuleTest(DeviceSourcesTest):
    """Covers the rule that no test or application source reaches a kernel."""

    def test_reports_a_test_and_an_application_that_reach_a_kernel(self) -> None:
        _write(self.root, "tests/core/optim/test_step.cpp", "#include <SushiAI/optim/step.hpp>")
        _write(self.root, "applications/tool/main.cpp", K_KERNEL_INCLUDE)
        _write(self.root, "tests/core/optim/test_plain.cpp", "#include <SushiAI/optim/plain.hpp>")
        tree = self._declare(["source/schedule.cpp"], [])
        self.assertEqual(len(self._messages(rule_device_source_outside_module, tree)), 2)


if __name__ == "__main__":
    unittest.main()

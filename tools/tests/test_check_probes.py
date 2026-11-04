# test_check_probes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/testing/check_probes.py on repositories held in memory, and its command line once on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from testing.check_probes import (  # noqa: E402
    K_CHECKER,
    Repository,
    rule_program_name_stem,
    rule_program_in_test_runner,
    rule_program_outside_folder,
    rule_program_test_framework,
    rule_unlisted_program,
)

K_ROOT = Path("repo")


def repository_of(texts: dict[str, str]) -> Repository:
    """Returns a repository whose tracked files are the given paths below its root, with their text."""
    sources = tuple(SourceFile(K_ROOT / name, text) for name, text in texts.items())
    return Repository(K_ROOT, tuple(source.path for source in sources), sources)


def found(rule, texts: dict[str, str]) -> list[str]:
    """Returns the path, relative to the root, of each issue a rule yields."""
    return [issue.path.relative_to(K_ROOT).as_posix() for issue in rule(repository_of(texts))]


class TestProbeRules(unittest.TestCase):
    """Runs each rule on repositories held in memory."""
    K_TEST_SIZE = "small"

    def test_reports_a_probes_folder_outside_applications(self) -> None:
        texts = {
            "applications/probes/water/main.cpp": "",
            "tools/probes/clouds/main.cpp": "",
            "applications/render_probe/main.cpp": "",
            "modules/render/source/gi/probe/volume.cpp": "",
            "tests/foundation/device_probe/test_query.cpp": "",
        }
        self.assertEqual(found(rule_program_outside_folder, texts), ["applications/render_probe", "tools/probes"])

    def test_reports_a_probe_that_uses_a_test_framework(self) -> None:
        texts = {
            "applications/probes/golden/main.cpp": "#include <cstdio>\n#include <gtest/gtest.h>\n",
            "applications/probes/water/main.cpp": "#include <cstdio>\n",
        }
        self.assertEqual(found(rule_program_test_framework, texts), ["applications/probes/golden/main.cpp"])

    def test_reports_a_probe_added_to_the_test_runner(self) -> None:
        texts = {"applications/probes/water/CMakeLists.txt":
                 "add_executable(water main.cpp)\nadd_test(NAME w COMMAND water)\n"}
        self.assertEqual(found(rule_program_in_test_runner, texts), ["applications/probes/water/CMakeLists.txt"])

    def test_reports_a_probe_named_after_another(self) -> None:
        texts = {f"applications/probes/{name}/main.cpp": "" for name in ("atmosphere", "atmosphere_global", "water")}
        self.assertEqual(found(rule_program_name_stem, texts), ["applications/probes/atmosphere_global"])

    def test_reports_a_probe_the_readme_does_not_name(self) -> None:
        texts = {"applications/probes/README.md": "# probes\n\n### water\n\nMeasures `clouds` too.\n"}
        texts.update({f"applications/probes/{name}/main.cpp": "" for name in ("water", "clouds", "atmosphere")})
        self.assertEqual(found(rule_unlisted_program, texts), ["applications/probes/README.md"])

    def test_applies_the_same_rules_to_scenario_benchmarks(self) -> None:
        texts = {
            "applications/benchmarks/README.md": "# benchmarks\n\n### matter\n",
            "applications/benchmarks/matter/main.cpp": "#include <cstdio>\n",
            "applications/benchmarks/matter_extra/main.cpp": "#include <gtest/gtest.h>\n",
            "applications/matter_benchmark/main.cpp": "",
            "applications/probes/matter/main.cpp": "",
            "applications/probes/README.md": "`matter`\n",
        }
        self.assertEqual(found(rule_program_outside_folder, texts), ["applications/matter_benchmark"])
        self.assertEqual(found(rule_program_test_framework, texts),
                         ["applications/benchmarks/matter_extra/main.cpp"])
        self.assertEqual(found(rule_program_name_stem, texts), ["applications/benchmarks/matter_extra"])
        self.assertEqual(found(rule_unlisted_program, texts), ["applications/benchmarks/README.md"])


class TestProbeCommand(unittest.TestCase):
    """Runs the checker's command line once on a tree written to a temporary folder."""
    K_TEST_SIZE = "medium"

    def test_command_reports_each_rule_with_its_line_and_exits_non_zero(self) -> None:
        files = {
            "applications/probes/README.md": "### golden\n\n### water\n",
            "applications/probes/golden/main.cpp": "#include <gtest/gtest.h>\n",
            "applications/probes/water/CMakeLists.txt": "add_executable(water main.cpp)\nadd_test(NAME w COMMAND water)\n",
            "applications/render_probe/main.cpp": "",
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative, text in files.items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(text, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [str(root)])
        self.assertEqual((code, output.getvalue().splitlines()), (1, [
            "applications/probes/golden/main.cpp:1: rule_program_test_framework: a probe uses a test "
            "framework; a program that decides pass or fail is a test",
            "applications/probes/water/CMakeLists.txt:2: rule_program_in_test_runner: "
            "a probe is never added to the test runner",
            "applications/render_probe:1: rule_program_outside_folder: "
            "probes live in applications/probes/<name>/",
        ]))


if __name__ == "__main__":
    unittest.main()

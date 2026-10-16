# test_check_probes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/testing/check_probes.py on repositories built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.check_probes import (  # noqa: E402
    discover,
    rule_probe_name_stem,
    rule_probe_in_test_runner,
    rule_probe_outside_folder,
    rule_probe_test_framework,
    rule_unlisted_probe,
)


class ProbesTest(unittest.TestCase):
    """Builds a repository for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, relative: str, text: str = "") -> None:
        """Writes a file below the repository root."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def found(self, rule) -> list[str]:
        """Returns the path, relative to the root, of each issue a rule yields."""
        return [issue.path.relative_to(self.root).as_posix() for issue in rule(discover(self.root))]

    def test_reports_a_probes_folder_outside_applications(self) -> None:
        self.write("applications/probes/water/main.cpp")
        self.write("tools/probes/clouds/main.cpp")
        self.write("applications/render_probe/main.cpp")
        self.write("modules/render/source/gi/probe/volume.cpp")
        self.write("tests/foundation/device_probe/test_query.cpp")
        self.assertEqual(self.found(rule_probe_outside_folder), ["applications/render_probe", "tools/probes"])

    def test_reports_a_probe_that_uses_a_test_framework(self) -> None:
        self.write("applications/probes/golden/main.cpp", "#include <cstdio>\n#include <gtest/gtest.h>\n")
        self.write("applications/probes/water/main.cpp", "#include <cstdio>\n")
        self.assertEqual(self.found(rule_probe_test_framework), ["applications/probes/golden/main.cpp"])

    def test_reports_a_probe_added_to_the_test_runner(self) -> None:
        self.write("applications/probes/water/CMakeLists.txt", "add_executable(water main.cpp)\nadd_test(NAME w COMMAND water)\n")
        self.assertEqual(self.found(rule_probe_in_test_runner), ["applications/probes/water/CMakeLists.txt"])

    def test_reports_a_probe_named_after_another(self) -> None:
        for name in ("atmosphere", "atmosphere_global", "water"):
            self.write(f"applications/probes/{name}/main.cpp")
        self.assertEqual(self.found(rule_probe_name_stem), ["applications/probes/atmosphere_global"])

    def test_reports_a_probe_the_readme_does_not_name(self) -> None:
        self.write("applications/probes/README.md", "# probes\n\n### water\n\nMeasures `clouds` too.\n")
        for name in ("water", "clouds", "atmosphere"):
            self.write(f"applications/probes/{name}/main.cpp")
        self.assertEqual(self.found(rule_unlisted_probe), ["applications/probes/README.md"])


if __name__ == "__main__":
    unittest.main()

# test_check_test_mirror.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the test mirror checker on single-module and multi-module trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.check_test_mirror import (  # noqa: E402
    collect,
    discover,
    rule_mirror_folders,
    rule_orphan_test,
    rule_unit_file_and_folder,
    rule_untested_unit,
)


def _touch(root: Path, relative: str) -> None:
    """Creates an empty file and its folders below root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n", encoding="utf-8")


def _names(issues) -> list[str]:
    """Returns the file or folder name of each issue."""
    return [issue.path.name for issue in issues]


class MultiModuleTest(unittest.TestCase):
    """Covers modules/<tier>/<module>/ mirrored to tests/<tier>/<module>/."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        _touch(self.root, "modules/physics/rigid_body/include/Sushi/rigid_body/physics_world.hpp")
        _touch(self.root, "modules/physics/rigid_body/include/Sushi/rigid_body/contact_solver.hpp")
        _touch(self.root, "modules/physics/rigid_body/include/Sushi/rigid_body.hpp")
        _touch(self.root, "tests/physics/rigid_body/test_physics_world.cpp")
        _touch(self.root, "tests/physics/rigid_body/test_rigid_body.cpp")
        _touch(self.root, "tests/physics/rigid_body/fixtures/contact_solver/scene.json")

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_a_header_without_its_test_is_reported(self) -> None:
        self.assertEqual(_names(rule_untested_unit(discover(self.root))), ["contact_solver.hpp"])

    def test_a_test_with_no_header_is_reported(self) -> None:
        _touch(self.root, "tests/physics/rigid_body/test_old_solver.cpp")
        self.assertEqual(_names(rule_orphan_test(discover(self.root))), ["test_old_solver.cpp"])

    def test_a_kind_folder_is_reported(self) -> None:
        _touch(self.root, "tests/unit/test_physics_world.cpp")
        self.assertEqual(_names(rule_mirror_folders(discover(self.root))), ["unit"])

    def test_an_include_path_with_the_tier_mirrors_like_one_without(self) -> None:
        _touch(self.root, "modules/vision/training/include/Sushi/vision/training/trainer.hpp")
        _touch(self.root, "tests/vision/training/test_trainer.cpp")
        self.assertEqual(_names(rule_untested_unit(discover(self.root))), ["contact_solver.hpp"])
        self.assertEqual(list(rule_orphan_test(discover(self.root))), [])

    def test_a_folder_of_aspect_files_stands_for_the_unit_test(self) -> None:
        _touch(self.root, "tests/physics/rigid_body/test_contact_solver/test_friction.cpp")
        _touch(self.root, "tests/physics/rigid_body/test_contact_solver/test_restitution.cpp")
        mirror = discover(self.root)
        self.assertEqual(list(rule_untested_unit(mirror)), [])
        self.assertEqual(list(rule_orphan_test(mirror)), [])
        self.assertEqual(list(rule_mirror_folders(mirror)), [])

    def test_a_unit_with_both_a_test_file_and_a_folder_is_reported(self) -> None:
        _touch(self.root, "tests/physics/rigid_body/test_physics_world/test_gravity.cpp")
        self.assertEqual(_names(rule_unit_file_and_folder(discover(self.root))), ["test_physics_world.cpp"])

    def test_data_folders_and_the_package_test_are_allowed(self) -> None:
        _touch(self.root, "tests/package/main.cpp")
        self.assertEqual(list(rule_mirror_folders(discover(self.root))), [])
        self.assertEqual(list(rule_orphan_test(discover(self.root))), [])


class LanguagesAndApplicationsTest(unittest.TestCase):
    """Covers Python and TypeScript sources and the headers of an application."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_python_mirrors_each_source_file_below_its_package(self) -> None:
        _touch(self.root, "source/sushi_cli/commands/build.py")
        _touch(self.root, "source/sushi_cli/__init__.py")
        _touch(self.root, "source/sushi_cli/_private.py")
        self.assertEqual(
            [test.relative_to(self.root).as_posix() for test in discover(self.root).expected],
            ["tests/commands/test_build.py"],
        )

    def test_typescript_mirrors_each_source_file_of_a_module(self) -> None:
        _touch(self.root, "modules/web/viewer/source/camera.ts")
        _touch(self.root, "modules/web/viewer/source/index.ts")
        self.assertEqual(
            [test.relative_to(self.root).as_posix() for test in discover(self.root).expected],
            ["tests/web/viewer/camera.test.ts"],
        )

    def test_an_application_mirrors_to_tests_applications(self) -> None:
        _touch(self.root, "include/Sushi/sushi.hpp")
        _touch(self.root, "applications/editor/panels/outliner.hpp")
        _touch(self.root, "tests/test_sushi.cpp")
        _touch(self.root, "tests/applications/editor/panels/test_outliner.cpp")
        mirror = discover(self.root)
        findings = [*rule_untested_unit(mirror), *rule_orphan_test(mirror), *rule_mirror_folders(mirror)]
        self.assertEqual(findings, [])


class UnityTest(unittest.TestCase):
    """Covers Assets/<Product>/ mirrored to Assets/<Product>/Tests/ with <Name>Tests.cs."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        (self.root / "ProjectSettings").mkdir()
        _touch(self.root, "Assets/MobileRTS/Core/CommandBus.cs")
        _touch(self.root, "Assets/MobileRTS/Features/Ballistics/Simulation/ProjectileFlight.cs")
        _touch(self.root, "Assets/MobileRTS/Legacy/OldRadar.cs")
        _touch(self.root, "Assets/ThirdParty/Vendor/Thing.cs")
        _touch(self.root, "Assets/MobileRTS/Tests/Core/CommandBusTests.cs")
        _touch(self.root, "Assets/MobileRTS/Tests/Features/Ballistics/Simulation/Fixtures/Shot/a.json")

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _all(self) -> list[str]:
        """Returns the rule and the file name of every finding over the repository."""
        findings = []
        for mirror in collect([self.root]):
            for rule in (rule_untested_unit, rule_orphan_test, rule_mirror_folders):
                findings += [(rule.__name__, issue.path.name) for issue in rule(mirror)]
        return findings

    def test_an_untested_script_is_reported_and_legacy_and_vendors_are_not(self) -> None:
        self.assertEqual(self._all(), [("rule_untested_unit", "ProjectileFlight.cs")])

    def test_a_kind_folder_and_its_test_are_reported(self) -> None:
        _touch(self.root, "Assets/MobileRTS/Features/Ballistics/Simulation/ProjectileFlight.cs")
        _touch(self.root, "Assets/MobileRTS/Tests/Features/Ballistics/Simulation/ProjectileFlightTests.cs")
        _touch(self.root, "Assets/MobileRTS/Tests/EditMode/RadarTests.cs")
        self.assertEqual(self._all(), [
            ("rule_orphan_test", "RadarTests.cs"),
            ("rule_mirror_folders", "EditMode"),
        ])


class SingleModuleTest(unittest.TestCase):
    """Covers include/<Project>/<area>/ mirrored to tests/<area>/."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        _touch(self.root, "include/Sushi/collision/broad_phase.hpp")
        _touch(self.root, "include/Sushi/sushi.hpp")
        _touch(self.root, "tests/collision/test_broad_phase.cpp")
        _touch(self.root, "tests/test_sushi.cpp")

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_a_mirrored_tree_is_clean(self) -> None:
        mirror = discover(self.root)
        findings = [*rule_untested_unit(mirror), *rule_orphan_test(mirror), *rule_mirror_folders(mirror)]
        self.assertEqual(findings, [])

    def test_a_test_in_the_wrong_area_is_reported_twice(self) -> None:
        (self.root / "tests/collision/test_broad_phase.cpp").unlink()
        _touch(self.root, "tests/integration/test_broad_phase.cpp")
        mirror = discover(self.root)
        self.assertEqual(_names(rule_untested_unit(mirror)), ["broad_phase.hpp"])
        self.assertEqual(_names(rule_orphan_test(mirror)), ["test_broad_phase.cpp"])
        self.assertEqual(_names(rule_mirror_folders(mirror)), ["integration"])


if __name__ == "__main__":
    unittest.main()

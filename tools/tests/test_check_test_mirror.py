# test_check_test_mirror.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the test mirror checker: its rules on mirrors built in memory, its command line once on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from testing.check_test_mirror import (  # noqa: E402
    K_CHECKER,
    Entries,
    TestMirror as Mirror,
    mirrors_of,
    rule_mirror_folders,
    rule_orphan_test,
    rule_private_include,
    rule_unit_file_and_folder,
    rule_untested_unit,
)

K_ROOT = Path("repo")
K_MULTI_MODULE = (
    "modules/physics/rigid_body/include/Acme/rigid_body/physics_world.hpp",
    "modules/physics/rigid_body/include/Acme/rigid_body/contact_solver.hpp",
    "modules/physics/rigid_body/include/Acme/rigid_body.hpp",
    "tests/physics/rigid_body/test_physics_world.cpp",
    "tests/physics/rigid_body/test_rigid_body.cpp",
    "tests/physics/rigid_body/fixtures/contact_solver/scene.json",
)
K_PRIVATE_TREE = (
    "modules/physics/rigid_body/include/Acme/rigid_body/physics_world.hpp",
    "modules/physics/rigid_body/source/island_builder.hpp",
    "modules/physics/rigid_body/source/detail/sweep.hpp",
)
K_PHYSICS_WORLD_TEST = "tests/physics/rigid_body/test_physics_world.cpp"
K_SINGLE_MODULE = (
    "include/Acme/collision/broad_phase.hpp",
    "include/Acme/acme.hpp",
    "tests/collision/test_broad_phase.cpp",
    "tests/test_acme.cpp",
)
K_UNITY = (
    "ProjectSettings/ProjectSettings.asset",
    "Assets/MobileRTS/Core/CommandBus.cs",
    "Assets/MobileRTS/Features/Ballistics/Simulation/ProjectileFlight.cs",
    "Assets/MobileRTS/Legacy/OldRadar.cs",
    "Assets/ThirdParty/Vendor/Thing.cs",
    "Assets/MobileRTS/Tests/Core/CommandBusTests.cs",
    "Assets/MobileRTS/Tests/Features/Ballistics/Simulation/Fixtures/Shot/a.json",
)


def entries_of(files: Iterable[str], texts: Mapping[str, str] | None = None,
               ignored: Iterable[str] = ()) -> Entries:
    """Returns the entries of a repository whose files are the given paths below its root."""
    names = tuple(files)
    folders = {K_ROOT.joinpath(*parent.parts) for name in names
               for parent in PurePosixPath(name).parents if parent.parts}
    sources = tuple(SourceFile(K_ROOT / name, text) for name, text in (texts or {}).items())
    return Entries(K_ROOT, frozenset(K_ROOT / name for name in names), frozenset(folders),
                   frozenset(K_ROOT / name for name in ignored), sources)


def mirror_of(files: Iterable[str], texts: Mapping[str, str] | None = None) -> Mirror:
    """Returns the mirror of a single-module or multi-module repository whose files are given."""
    return mirrors_of(entries_of(files, texts))[0]


def expected_tests(mirror: Mirror) -> list[str]:
    """Returns the expected test file of each unit, relative to the root, sorted."""
    return sorted(test.relative_to(K_ROOT).as_posix() for test in mirror.expected)


def findings(mirror: Mirror) -> list[tuple[str, str]]:
    """Returns the rule and the file name of each issue the three tree rules yield, in rule order."""
    return [(rule.__name__, issue.path.name)
            for rule in (rule_untested_unit, rule_orphan_test, rule_mirror_folders)
            for issue in rule(mirror)]


def names(issues) -> list[str]:
    """Returns the file or folder name of each issue."""
    return [issue.path.name for issue in issues]


class TestMultiModule(unittest.TestCase):
    """Covers modules/<tier>/<module>/ mirrored to tests/<tier>/<module>/."""
    K_TEST_SIZE = "small"

    def test_a_header_without_its_test_is_reported(self) -> None:
        self.assertEqual(names(rule_untested_unit(mirror_of(K_MULTI_MODULE))), ["contact_solver.hpp"])

    def test_a_test_with_no_header_is_reported(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/physics/rigid_body/test_old_solver.cpp"))
        self.assertEqual(names(rule_orphan_test(mirror)), ["test_old_solver.cpp"])

    def test_a_kind_folder_is_reported(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/unit/test_physics_world.cpp"))
        self.assertEqual(names(rule_mirror_folders(mirror)), ["unit"])

    def test_an_include_path_with_the_tier_mirrors_like_one_without(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "modules/vision/training/include/Acme/vision/training/trainer.hpp",
                            "tests/vision/training/test_trainer.cpp"))
        self.assertEqual(names(rule_untested_unit(mirror)), ["contact_solver.hpp"])
        self.assertEqual(list(rule_orphan_test(mirror)), [])

    def test_a_folder_of_aspect_files_stands_for_the_unit_test(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/physics/rigid_body/test_contact_solver/test_friction.cpp",
                            "tests/physics/rigid_body/test_contact_solver/test_restitution.cpp"))
        self.assertEqual(findings(mirror), [])

    def test_a_unit_with_both_a_test_file_and_a_folder_is_reported(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/physics/rigid_body/test_physics_world/test_gravity.cpp"))
        self.assertEqual(names(rule_unit_file_and_folder(mirror)), ["test_physics_world.cpp"])

    def test_data_folders_and_the_package_test_are_allowed(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/package/main.cpp"))
        self.assertEqual(list(rule_mirror_folders(mirror)), [])
        self.assertEqual(list(rule_orphan_test(mirror)), [])

    def test_a_compile_check_stands_for_the_unit_test_of_a_header_with_no_runtime_promise(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/physics/rigid_body/compile_contact_solver.cpp"))
        self.assertEqual(findings(mirror), [])

    def test_the_shared_test_support_folder_is_allowed(self) -> None:
        mirror = mirror_of((*K_MULTI_MODULE, "tests/support/tolerance.hpp", "tests/support/test_fixture.hpp"))
        self.assertEqual(list(rule_mirror_folders(mirror)), [])
        self.assertEqual(list(rule_orphan_test(mirror)), [])


class TestPrivateInclude(unittest.TestCase):
    """Covers tests that include a header from a module's or the root's source/ folder."""
    K_TEST_SIZE = "small"

    def private_found(self, path: str, text: str, *extra: str) -> list[tuple[str, int]]:
        """Returns the file name and line of every private include in one test file, with extra files beside it."""
        mirror = mirror_of((*K_PRIVATE_TREE, path, *extra), {path: text})
        return [(issue.path.name, issue.line) for issue in rule_private_include(mirror)]

    def test_a_public_include_is_clean(self) -> None:
        self.assertEqual(self.private_found(K_PHYSICS_WORLD_TEST,
                                            "#include <Acme/rigid_body/physics_world.hpp>\n"), [])

    def test_a_private_header_by_name_is_reported(self) -> None:
        self.assertEqual(self.private_found(K_PHYSICS_WORLD_TEST,
                                            "#include <gtest/gtest.h>\n#include \"island_builder.hpp\"\n"
                                            "#include <detail/sweep.hpp>\n"),
                         [("test_physics_world.cpp", 2), ("test_physics_world.cpp", 3)])

    def test_a_path_through_source_is_reported(self) -> None:
        self.assertEqual(self.private_found(
            "tests/physics/rigid_body/helper.hpp",
            "#include \"../../../modules/physics/rigid_body/source/island_builder.hpp\"\n"), [("helper.hpp", 1)])

    def test_a_helper_beside_the_test_is_not_private(self) -> None:
        self.assertEqual(self.private_found(K_PHYSICS_WORLD_TEST, "#include \"island_builder.hpp\"\n",
                                            "tests/physics/rigid_body/island_builder.hpp"), [])


class TestLanguagesAndApplications(unittest.TestCase):
    """Covers Python and TypeScript sources and the headers of an application."""
    K_TEST_SIZE = "small"

    def test_python_mirrors_each_source_file_below_its_package(self) -> None:
        mirror = mirror_of(("source/acme_cli/commands/build.py", "source/acme_cli/__init__.py",
                            "source/acme_cli/_private.py", "source/acme_cli/bundle/__main__.py"))
        self.assertEqual(expected_tests(mirror), ["tests/bundle/test_main.py", "tests/commands/test_build.py"])

    def test_typescript_mirrors_each_source_file_of_a_module(self) -> None:
        mirror = mirror_of(("modules/web/viewer/source/camera.ts", "modules/web/viewer/source/index.ts"))
        self.assertEqual(expected_tests(mirror), ["tests/web/viewer/camera.test.ts"])

    def test_an_application_mirrors_to_tests_applications(self) -> None:
        mirror = mirror_of(("include/Acme/acme.hpp", "applications/editor/panels/outliner.hpp",
                            "tests/test_acme.cpp", "tests/applications/editor/panels/test_outliner.cpp"))
        self.assertEqual(findings(mirror), [])


class TestUnity(unittest.TestCase):
    """Covers Assets/<Product>/ mirrored to Assets/<Product>/Tests/ with <Name>Tests.cs."""
    K_TEST_SIZE = "small"

    def unity_findings(self, files: Iterable[str]) -> list[tuple[str, str]]:
        """Returns the rule and the file name of every finding over each product of a Unity project."""
        return [finding for mirror in mirrors_of(entries_of(files)) for finding in findings(mirror)]

    def test_an_untested_script_is_reported_and_legacy_and_vendors_are_not(self) -> None:
        self.assertEqual(self.unity_findings(K_UNITY), [("rule_untested_unit", "ProjectileFlight.cs")])

    def test_a_kind_folder_and_its_test_are_reported(self) -> None:
        files = (*K_UNITY, "Assets/MobileRTS/Tests/Features/Ballistics/Simulation/ProjectileFlightTests.cs",
                 "Assets/MobileRTS/Tests/EditMode/RadarTests.cs")
        self.assertEqual(self.unity_findings(files), [
            ("rule_orphan_test", "RadarTests.cs"),
            ("rule_mirror_folders", "EditMode"),
        ])


class TestSingleModule(unittest.TestCase):
    """Covers include/<Project>/<area>/ mirrored to tests/<area>/."""
    K_TEST_SIZE = "small"

    def test_a_mirrored_tree_is_clean(self) -> None:
        self.assertEqual(findings(mirror_of(K_SINGLE_MODULE)), [])

    def test_a_test_in_the_wrong_area_gives_three_findings(self) -> None:
        files = [name for name in K_SINGLE_MODULE if name != "tests/collision/test_broad_phase.cpp"]
        files.append("tests/integration/test_broad_phase.cpp")
        self.assertEqual(findings(mirror_of(files)), [
            ("rule_untested_unit", "broad_phase.hpp"),
            ("rule_orphan_test", "test_broad_phase.cpp"),
            ("rule_mirror_folders", "integration"),
        ])


class TestMirrorCommand(unittest.TestCase):
    """Runs the checker's command line once on each of two trees written to a temporary folder."""
    K_TEST_SIZE = "medium"

    def write(self, root: Path, files: Mapping[str, str]) -> None:
        """Writes each file, with its folders, below the root."""
        for relative, text in files.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text, encoding="utf-8")

    def command(self, files: Mapping[str, str]) -> tuple[int, list[str]]:
        """Returns the exit code and the printed lines of one run over a tree of the given files."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write(root, files)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [str(root)])
        return code, output.getvalue().splitlines()

    def test_command_reports_each_rule_on_a_layered_tree_with_its_line(self) -> None:
        code, lines = self.command({
            "modules/physics/rigid_body/include/Acme/rigid_body/physics_world.hpp": "",
            "modules/physics/rigid_body/include/Acme/rigid_body/contact_solver.hpp": "",
            "modules/physics/rigid_body/source/island_builder.hpp": "",
            "tests/physics/rigid_body/test_physics_world.cpp": '#include "island_builder.hpp"\n',
            "tests/physics/rigid_body/test_old_solver.cpp": "",
            "tests/unit/test_physics_world.cpp": "",
        })
        self.assertEqual((code, lines), (1, [
            "modules/physics/rigid_body/include/Acme/rigid_body/contact_solver.hpp:1: rule_untested_unit: "
            "no test file; expected tests/physics/rigid_body/test_contact_solver.cpp",
            "tests/physics/rigid_body/test_old_solver.cpp:1: rule_orphan_test: "
            "mirrors no header or source file; rename it or move it to its unit's mirror",
            "tests/physics/rigid_body/test_physics_world.cpp:1: rule_private_include: "
            "includes the private header island_builder.hpp; test it through the public surface",
            "tests/unit:1: rule_mirror_folders: folder has no counterpart in the source tree",
            "tests/unit/test_physics_world.cpp:1: rule_orphan_test: "
            "mirrors no header or source file; rename it or move it to its unit's mirror",
        ]))


if __name__ == "__main__":
    unittest.main()

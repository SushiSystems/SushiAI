# test_check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rules of tools/layering/check_layering.py on module trees held in memory, and its command line on a temporary tree."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from layering import check_layering  # noqa: E402
from layering.check_layering import (  # noqa: E402
    K_BUILD_FILE,
    K_CHECKER,
    K_INCLUDE_ROOTS,
    K_PUBLIC_FOLDER,
    K_SOURCE_SUFFIXES,
    Layout,
    Module,
    ModuleFiles,
    ModuleTree,
    rule_cross_domain_includes,
    rule_declared_dependencies,
    rule_declared_libraries,
    rule_declared_tiers,
    rule_private_reach,
    rule_public_headers,
    rule_restricted_includes,
    rule_same_tier_includes,
    rule_unused_dependencies,
    rule_upward_includes,
)

K_ROOT = Path("repo")
K_LAYOUT = Layout(("foundation", "domain"))
K_DOMAINS = Layout(("core", "vision", "language"), domain_tiers=("vision", "language"))
K_BUILD = Layout(("foundation",), module_function="project_add_module")
K_DOORS = Layout(
    ("foundation",),
    include_doors=(("Vendor/", ("modules/foundation/front/", "tests/foundation/front/", "applications/tool/vendor_")),),
)
K_FOREIGN = Layout(
    ("foundation",),
    module_function="project_add_module",
    foreign_roots=(("Runtime", "execution"),),
    foreign_providers=(("runtime_sdk", "Runtime"), ("other_sdk", "Other")),
)
K_GEOMETRY = {"modules/domain/geometry/include/Project/geometry/mesh.hpp": ""}
K_CORE = {
    "modules/foundation/core/include/Project/core/core.hpp": "",
    "modules/foundation/core/CMakeLists.txt": "project_add_module(NAME core INTERNAL)",
    "modules/foundation/log/include/Project/log/log.hpp": "#include <Project/core/core.hpp>",
    "modules/foundation/log/source/log.cpp": "#include <Project/core/core.hpp>",
}


def _sources(files: dict[str, str], folder: PurePosixPath) -> tuple[SourceFile, ...]:
    """Returns the C-family sources below a folder, as the shell reads them, in path order."""
    return tuple(
        SourceFile(K_ROOT / name, text)
        for name, text in sorted(files.items())
        if PurePosixPath(name).is_relative_to(folder)
        and PurePosixPath(name).suffix in K_SOURCE_SUFFIXES
    )


def _module_files(module: Module, files: dict[str, str]) -> ModuleFiles:
    """Returns what the shell reads of one module: its include/ names, sources, tests and build text."""
    folder = PurePosixPath("modules", module.tier, module.name)
    public = folder / K_PUBLIC_FOLDER
    below_public = [
        PurePosixPath(name).relative_to(public)
        for name in files if PurePosixPath(name).is_relative_to(public)
    ]
    names = {
        PurePosixPath(*path.parts[:index]).as_posix()
        for path in below_public
        for index in range(1, len(path.parts) + 1)
    }
    return ModuleFiles(
        module=module,
        has_public_folder=bool(below_public),
        public_names=tuple(sorted(names)),
        sources=_sources(files, folder),
        tests=_sources(files, PurePosixPath("tests", module.tier, module.name)),
        build_text=files.get((folder / K_BUILD_FILE).as_posix()),
        build_files=tuple(
            SourceFile(K_ROOT / name, text)
            for name, text in sorted(files.items())
            if PurePosixPath(name).is_relative_to(folder) and PurePosixPath(name).name == K_BUILD_FILE
        ),
    )


def _tree(files: dict[str, str], layout: Layout = K_LAYOUT) -> ModuleTree:
    """Returns the module tree the shell reads from a repository that holds these files, given as text."""
    keys = sorted({
        (PurePosixPath(name).parts[1], PurePosixPath(name).parts[2])
        for name in files
        if PurePosixPath(name).parts[:1] == ("modules",) and len(PurePosixPath(name).parts) > 3
    })
    restricted = [
        source for folder in K_INCLUDE_ROOTS for source in _sources(files, PurePosixPath(folder))
    ]
    return ModuleTree(K_ROOT, layout, [
        _module_files(Module(tier, name, K_ROOT / "modules" / tier / name), files)
        for tier, name in keys
    ], restricted)


def _messages(issues) -> list[str]:
    """Returns the message of each issue."""
    return [issue.message for issue in issues]


class RuleUpwardIncludesTest(unittest.TestCase):
    """Covers the rule that a module and its tests include nothing from a higher tier."""
    K_TEST_SIZE = "small"

    def test_reports_an_include_of_a_higher_module(self) -> None:
        files = {**K_GEOMETRY, "modules/foundation/blas/include/Project/blas/backend.hpp":
                 "#include <Project/geometry/mesh.hpp>"}
        self.assertEqual(_messages(rule_upward_includes(_tree(files))),
                         ["foundation/blas includes up into domain/geometry"])

    def test_ignores_another_project_with_a_module_of_the_same_name(self) -> None:
        files = {**K_GEOMETRY, "modules/foundation/blas/include/Project/blas/backend.hpp":
                 "#include <Other/geometry/mesh.hpp>"}
        self.assertEqual(list(rule_upward_includes(_tree(files))), [])

    def test_reports_an_upward_include_in_a_module_test(self) -> None:
        files = {**K_GEOMETRY, "modules/foundation/blas/include/Project/blas/backend.hpp": "",
                 "tests/foundation/blas/test_backend.cpp": "#include <Project/geometry/mesh.hpp>"}
        self.assertEqual(_messages(rule_upward_includes(_tree(files))),
                         ["foundation/blas includes up into domain/geometry"])

    def test_allows_a_test_to_include_an_umbrella(self) -> None:
        files = {**K_GEOMETRY, "modules/domain/umbrella/include/Project/Project.hpp": "",
                 "modules/foundation/blas/include/Project/blas/backend.hpp": "",
                 "tests/foundation/blas/test_backend.cpp": "#include <Project/Project.hpp>"}
        self.assertEqual(list(rule_upward_includes(_tree(files))), [])

    def test_resolves_the_tier_form_and_a_root_umbrella(self) -> None:
        files = {
            **K_GEOMETRY,
            "modules/domain/umbrella/include/Project/domain.hpp": "",
            "modules/domain/training/include/Project/domain/training/trainer.hpp": "",
            "modules/foundation/training/include/Project/foundation/training/loop.hpp":
                "#include <Project/domain/training/trainer.hpp>\n#include <Project/domain.hpp>",
        }
        self.assertEqual(_messages(rule_upward_includes(_tree(files))),
                         ["foundation/training includes up into domain/training",
                          "foundation/training includes up into domain/umbrella"])


class RuleDeclaredTiersTest(unittest.TestCase):
    """Covers the declared tier order and domains."""
    K_TEST_SIZE = "small"

    def test_reports_an_empty_tier_order(self) -> None:
        files = {"modules/extra/tool/include/Project/tool/tool.hpp": ""}
        self.assertEqual(len(list(rule_declared_tiers(_tree(files, Layout(()))))), 1)

    def test_reports_an_undeclared_tier_by_name(self) -> None:
        files = {"modules/extra/tool/include/Project/tool/tool.hpp": ""}
        self.assertEqual(_messages(rule_declared_tiers(_tree(files))),
                         ["tier 'extra' is not in K_TIER_ORDER"])

    def test_reports_a_domain_tier_missing_from_the_order_without_crashing_on_a_declared_one(self) -> None:
        layout = Layout(("core", "vision"), domain_tiers=("vision", "language"))
        files = {"modules/vision/camera/include/Project/camera/camera.hpp": ""}
        self.assertEqual(_messages(rule_declared_tiers(_tree(files, layout))),
                         ["domain tier 'language' is not in K_TIER_ORDER"])


class RuleCrossDomainIncludesTest(unittest.TestCase):
    """Covers domain tiers, which share a rank and never include each other."""
    K_TEST_SIZE = "small"

    def test_reports_one_domain_including_another(self) -> None:
        files = {
            "modules/vision/blocks/include/Project/vision/blocks/conv.hpp": "",
            "modules/language/blocks/include/Project/language/blocks/attention.hpp":
                "#include <Project/vision/blocks/conv.hpp>",
        }
        self.assertEqual(_messages(rule_cross_domain_includes(_tree(files, K_DOMAINS))),
                         ["language/blocks of one domain includes vision/blocks"])

    def test_allows_a_domain_to_include_a_lower_tier(self) -> None:
        files = {
            "modules/core/tensor/include/Project/core/tensor/tensor.hpp": "",
            "modules/language/blocks/include/Project/language/blocks/attention.hpp":
                "#include <Project/core/tensor/tensor.hpp>",
        }
        tree = _tree(files, K_DOMAINS)
        self.assertEqual(list(rule_cross_domain_includes(tree)) + list(rule_upward_includes(tree)), [])


class DomainNameResolutionTest(unittest.TestCase):
    """Covers a bare build name that a domain module shares with a lower one."""
    K_TEST_SIZE = "small"

    def test_a_bare_name_means_the_module_outside_the_domains(self) -> None:
        files = {
            "modules/core/umbrella/include/Project/core.hpp": "",
            "modules/vision/umbrella/include/Project/vision.hpp": "",
            "modules/vision/blocks/CMakeLists.txt": "project_add_module(NAME blocks DEPENDS umbrella)",
            "modules/vision/blocks/include/Project/vision/blocks/conv.hpp": "#include <Project/core.hpp>",
        }
        layout = Layout(("core", "vision"), domain_tiers=("vision",), module_function="project_add_module")
        tree = _tree(files, layout)
        self.assertEqual(list(rule_declared_dependencies(tree)) + list(rule_unused_dependencies(tree)), [])


class RuleSameTierIncludesTest(unittest.TestCase):
    """Covers the listed edges between modules of one tier."""
    K_TEST_SIZE = "small"

    def test_reports_an_edge_that_is_not_listed(self) -> None:
        files = {
            "modules/foundation/core/include/Project/core/core.hpp": "",
            "modules/foundation/log/include/Project/log/log.hpp": "#include <Project/core/core.hpp>",
        }
        self.assertEqual(len(list(rule_same_tier_includes(_tree(files)))), 1)
        listed = Layout(("foundation",), same_tier_edges=frozenset({("log", "core")}))
        self.assertEqual(list(rule_same_tier_includes(_tree(files, listed))), [])

    def test_lets_an_umbrella_include_its_tier(self) -> None:
        files = {
            "modules/foundation/core/include/Project/core/core.hpp": "",
            "modules/foundation/umbrella/include/Project/foundation.hpp": "#include <Project/core/core.hpp>",
        }
        self.assertEqual(list(rule_same_tier_includes(_tree(files))), [])


class RulePrivateReachTest(unittest.TestCase):
    """Covers reaching into another module's source/ folder."""
    K_TEST_SIZE = "small"

    def test_a_quoted_include_into_source_is_reported(self) -> None:
        files = {"modules/foundation/log/source/log.cpp": '#include "../../core/source/detail.hpp"'}
        self.assertEqual(_messages(rule_private_reach(_tree(files))),
                         ["quoted include leaves the module or names a source/ folder"])

    def test_a_build_line_into_another_modules_source_is_reported(self) -> None:
        files = {
            "modules/foundation/core/source/core.cpp": "",
            "modules/foundation/log/CMakeLists.txt": "target_sources(log PRIVATE ../core/source/core.cpp)",
        }
        self.assertEqual(_messages(rule_private_reach(_tree(files))), ["reaches into core/source"])

    def test_an_angle_include_of_a_public_header_is_clean(self) -> None:
        self.assertEqual(list(rule_private_reach(_tree(K_CORE))), [])


class BuildDeclarationTest(unittest.TestCase):
    """Covers the rules that read a module's declaring call in its build file."""
    K_TEST_SIZE = "small"

    def test_reports_an_include_the_build_file_does_not_list(self) -> None:
        files = {**K_CORE, "modules/foundation/log/CMakeLists.txt":
                 "project_add_module(NAME log PRIVATE_DEPENDS core)"}
        self.assertEqual(_messages(rule_declared_dependencies(_tree(files, K_BUILD))),
                         ["log includes core, which its CMakeLists.txt does not list in DEPENDS"])

    def test_reports_a_listed_module_nothing_includes(self) -> None:
        files = {**K_CORE, "modules/foundation/log/CMakeLists.txt":
                 "project_add_module(NAME log DEPENDS core PRIVATE_DEPENDS clock)"}
        self.assertEqual(_messages(rule_unused_dependencies(_tree(files, K_BUILD))),
                         ["log lists clock in PRIVATE_DEPENDS and no file includes it"])

    def test_a_further_private_keyword_declares_a_source_include(self) -> None:
        layout = Layout(("foundation",), module_function="project_add_module",
                        private_depends_also=("COMPILE_ONLY_DEPENDS",))
        files = {**K_CORE, "modules/foundation/log/include/Project/log/log.hpp": "",
                 "modules/foundation/log/CMakeLists.txt": "project_add_module(NAME log COMPILE_ONLY_DEPENDS core clock)"}
        tree = _tree(files, layout)
        self.assertEqual(_messages(rule_declared_dependencies(tree)), [])
        self.assertEqual(_messages(rule_unused_dependencies(tree)),
                         ["log lists clock in PRIVATE_DEPENDS and no file includes it"])

    def test_an_include_of_a_generated_header_uses_a_module_with_no_include_tree(self) -> None:
        files = {
            **K_CORE,
            "modules/foundation/shaders/CMakeLists.txt": "project_add_module(NAME shaders)",
            "modules/foundation/shaders/triangle.vert": "",
            "modules/foundation/log/source/draw.cpp": "#include <Project/shaders/triangle.vert.h>",
            "modules/foundation/log/CMakeLists.txt":
                "project_add_module(NAME log DEPENDS core PRIVATE_DEPENDS shaders clock)",
        }
        self.assertEqual(_messages(rule_unused_dependencies(_tree(files, K_BUILD))),
                         ["log lists clock in PRIVATE_DEPENDS and no file includes it"])

    def test_reads_same_tier_edges_from_the_build_file(self) -> None:
        files = {**K_CORE, "modules/foundation/log/CMakeLists.txt": "project_add_module(NAME log)"}
        self.assertEqual(_messages(rule_same_tier_includes(_tree(files, K_BUILD))),
                         ["foundation/log includes core, an edge its CMakeLists.txt does not list"] * 2)
        files["modules/foundation/log/CMakeLists.txt"] = "project_add_module(NAME log DEPENDS core)"
        self.assertEqual(list(rule_same_tier_includes(_tree(files, K_BUILD))), [])

    def test_resolves_a_tier_qualified_name_and_keeps_same_named_modules_apart(self) -> None:
        files = {
            **K_CORE,
            "modules/domain/core/include/Project/domain/core/core.hpp": "",
            "modules/foundation/log/CMakeLists.txt": "project_add_module(NAME log DEPENDS foundation/core)",
        }
        layout = Layout(("foundation", "domain"), module_function="project_add_module")
        self.assertEqual(list(rule_declared_dependencies(_tree(files, layout))), [])
        self.assertEqual(list(rule_unused_dependencies(_tree(files, layout))), [])
        files["modules/foundation/log/CMakeLists.txt"] = "project_add_module(NAME log DEPENDS domain/core)"
        self.assertEqual(len(list(rule_declared_dependencies(_tree(files, layout)))), 2)

    def test_reports_an_installed_header_that_includes_an_internal_module(self) -> None:
        files = {**K_CORE, "modules/foundation/log/CMakeLists.txt": "project_add_module(NAME log DEPENDS core)"}
        self.assertEqual(_messages(rule_public_headers(_tree(files, K_BUILD))),
                         ["log is installed and includes core, an internal module whose headers are not"])

    def test_skips_the_build_rules_when_no_function_is_named(self) -> None:
        tree = _tree(K_CORE, Layout(("foundation",)))
        self.assertEqual(list(rule_declared_dependencies(tree)) + list(rule_public_headers(tree)), [])


class RuleDeclaredLibrariesTest(unittest.TestCase):
    """Covers the rule that every library under modules/ is declared through the module function."""
    K_TEST_SIZE = "small"

    def _issues(self, files: dict[str, str], layout: Layout = K_BUILD) -> list[tuple[str, int]]:
        """Returns the file and line of each finding."""
        return [(issue.path.relative_to(K_ROOT).as_posix(), issue.line)
                for issue in rule_declared_libraries(_tree(files, layout))]

    def test_reports_a_plain_add_library_with_its_file_and_line(self) -> None:
        files = {"modules/domain/physics/CMakeLists.txt":
                 "project_add_module(NAME physics)\nadd_library(physics_cooking STATIC cook.cpp)"}
        self.assertEqual(self._issues(files), [("modules/domain/physics/CMakeLists.txt", 2)])

    def test_reports_any_case_and_any_compiled_type(self) -> None:
        files = {"modules/domain/physics/CMakeLists.txt":
                 "ADD_LIBRARY(a OBJECT x.cpp)\nAdd_Library (b STATIC z.cpp)\nadd_library(c SHARED y.cpp)"}
        self.assertEqual([line for _path, line in self._issues(files)], [1, 2, 3])

    def test_ignores_a_target_that_compiles_nothing(self) -> None:
        files = {"modules/domain/physics/CMakeLists.txt":
                 "add_library(sdk INTERFACE)\nadd_library(vendor SHARED IMPORTED GLOBAL)"}
        self.assertEqual(self._issues(files), [])

    def test_reports_a_build_file_below_the_module_folder(self) -> None:
        files = {"modules/domain/physics/tools/CMakeLists.txt": "add_library(cook STATIC cook.cpp)"}
        self.assertEqual(self._issues(files), [("modules/domain/physics/tools/CMakeLists.txt", 1)])

    def test_ignores_the_module_function_a_comment_and_a_longer_name(self) -> None:
        files = {"modules/domain/physics/CMakeLists.txt":
                 "project_add_module(NAME physics)\n# add_library(x STATIC y.cpp)\n"
                 "project_add_library(physics_cooking)\nadd_library_alias(z)"}
        self.assertEqual(self._issues(files), [])

    def test_is_silent_when_the_layout_names_no_module_function(self) -> None:
        files = {"modules/domain/physics/CMakeLists.txt": "add_library(a STATIC x.cpp)"}
        self.assertEqual(self._issues(files, K_LAYOUT), [])


class RuleRestrictedIncludesTest(unittest.TestCase):
    """Covers the rule that a restricted include prefix appears only in the listed paths."""
    K_TEST_SIZE = "small"

    def _issues(self, files: dict[str, str], layout: Layout = K_DOORS) -> list:
        """Returns the findings of the rule on a tree with these files."""
        return list(rule_restricted_includes(_tree(files, layout)))

    def test_include_outside_the_door_is_refused(self) -> None:
        files = {"modules/foundation/front/include/Project/front/front.hpp": "#include <Vendor/capture.hpp>",
                 "modules/foundation/other/source/other.cpp": "#include <Vendor/capture.hpp>"}
        messages = _messages(self._issues(files))
        self.assertEqual(len(messages), 1)
        self.assertIn("Vendor/capture.hpp", messages[0])

    def test_reports_the_file_and_line_of_the_include(self) -> None:
        files = {"modules/foundation/other/source/other.cpp": "// first\n#include <Vendor/capture.hpp>"}
        self.assertEqual([(issue.path.name, issue.line) for issue in self._issues(files)], [("other.cpp", 2)])

    def test_reads_applications_and_tests_as_well_as_modules(self) -> None:
        files = {"applications/tool/main.cpp": '#include "Vendor/capture.hpp"',
                 "tests/foundation/other/test_other.cpp": "#include <Vendor/capture.hpp>"}
        self.assertEqual(len(self._issues(files)), 2)

    def test_allows_the_listed_paths_and_a_file_name_prefix(self) -> None:
        files = {"modules/foundation/front/source/front.cpp": "#include <Vendor/capture.hpp>",
                 "tests/foundation/front/test_front.cpp": "#include <Vendor/capture.hpp>",
                 "applications/tool/vendor_session.cpp": "#include <Vendor/capture.hpp>"}
        self.assertEqual(self._issues(files), [])

    def test_ignores_another_include(self) -> None:
        files = {"modules/foundation/other/source/other.cpp": "#include <Other/capture.hpp>\n#include <vector>"}
        self.assertEqual(self._issues(files), [])

    def test_ignores_a_layout_with_no_door(self) -> None:
        files = {"modules/foundation/third/source/third.cpp": "#include <Vendor/capture.hpp>"}
        self.assertEqual(self._issues(files, K_LAYOUT), [])


class ForeignRootTest(unittest.TestCase):
    """Covers an include under a foreign root: it names the door module unless the includer links a provider."""
    K_TEST_SIZE = "small"

    def _messages(self, call: str, include: str = "#include <Runtime/api.hpp>", door: bool = True) -> list[str]:
        """Returns the declared-dependency findings of a user module whose build call is `call`."""
        files = {"modules/foundation/user/source/user.cpp": include,
                 "modules/foundation/user/CMakeLists.txt": call}
        if door:
            files["modules/foundation/execution/CMakeLists.txt"] = "project_add_module(NAME execution)"
        return _messages(rule_declared_dependencies(_tree(files, K_FOREIGN)))

    def test_names_the_door_module_when_the_includer_does_not_list_it(self) -> None:
        self.assertEqual(len(self._messages("project_add_module(NAME user)")), 1)

    def test_accepts_the_door_module_when_listed(self) -> None:
        self.assertEqual(self._messages("project_add_module(NAME user PRIVATE_DEPENDS execution)"), [])

    def test_accepts_a_linked_provider_in_any_case(self) -> None:
        self.assertEqual(self._messages("project_add_module(NAME user EXTERNAL RUNTIME_SDK)"), [])

    def test_ignores_a_provider_of_another_root(self) -> None:
        self.assertEqual(len(self._messages("project_add_module(NAME user EXTERNAL other_sdk)")), 1)

    def test_names_nothing_when_the_door_module_is_absent(self) -> None:
        self.assertEqual(self._messages("project_add_module(NAME user)", door=False), [])

    def test_leaves_an_include_of_another_root_alone(self) -> None:
        self.assertEqual(self._messages("project_add_module(NAME user)", "#include <Elsewhere/api.hpp>"), [])

    def test_leaves_a_bare_root_name_alone(self) -> None:
        self.assertEqual(self._messages("project_add_module(NAME user)", "#include <Runtime>"), [])


class ShellTest(unittest.TestCase):
    """Runs the command line on a tree on disk, so the files the shell reads reach every rule."""
    K_TEST_SIZE = "medium"

    def test_cli_reads_the_module_tree_and_reports_each_finding(self) -> None:
        """Prints the findings of a module and its test in path order, and exits non-zero."""
        layout = Layout(("foundation", "domain"), module_function="project_add_module")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in {
                "modules/domain/geometry/include/Project/geometry/mesh.hpp": "",
                "modules/foundation/blas/include/Project/blas/backend.hpp": "#include <Project/geometry/mesh.hpp>",
                "modules/foundation/blas/CMakeLists.txt": "project_add_module(NAME blas)",
                "tests/foundation/blas/test_backend.cpp": "#include <Project/geometry/mesh.hpp>",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", encoding="utf-8")
            output = io.StringIO()
            with mock.patch.object(check_layering, "K_LAYOUT", layout), contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [folder])
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue().splitlines(), [
                "modules/foundation/blas/include/Project/blas/backend.hpp:1: rule_declared_dependencies: "
                "blas includes geometry, which its CMakeLists.txt does not list in DEPENDS",
                "modules/foundation/blas/include/Project/blas/backend.hpp:1: rule_upward_includes: "
                "foundation/blas includes up into domain/geometry",
                "tests/foundation/blas/test_backend.cpp:1: rule_upward_includes: "
                "foundation/blas includes up into domain/geometry",
            ])


if __name__ == "__main__":
    unittest.main()

# test_check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/layering/check_layering.py on trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_layering import (  # noqa: E402
    Layout,
    ModuleTree,
    rule_cross_domain_includes,
    rule_declared_dependencies,
    rule_declared_tiers,
    rule_public_headers,
    rule_same_tier_includes,
    rule_unused_dependencies,
    rule_upward_includes,
)

K_LAYOUT = Layout(("foundation", "domain"))
K_DOMAINS = Layout(("core", "vision", "language"), domain_tiers=("vision", "language"))
K_BUILD = Layout(("foundation",), module_function="project_add_module")


def _write(root: Path, relative: str, text: str = "\n") -> None:
    """Writes a file and its folders below root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _messages(issues) -> list[str]:
    """Returns the message of each issue."""
    return [issue.message for issue in issues]


class LayeringTest(unittest.TestCase):
    """Builds a temporary repository for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()


class RuleUpwardIncludesTest(LayeringTest):
    """Covers the rule that a module and its tests include nothing from a higher tier."""

    def setUp(self) -> None:
        super().setUp()
        _write(self.root, "modules/domain/geometry/include/Project/geometry/mesh.hpp")

    def test_reports_an_include_of_a_higher_module(self) -> None:
        _write(self.root, "modules/foundation/blas/include/Project/blas/backend.hpp",
               "#include <Project/geometry/mesh.hpp>")
        self.assertEqual(_messages(rule_upward_includes(ModuleTree(self.root, K_LAYOUT))),
                         ["foundation/blas includes up into domain/geometry"])

    def test_ignores_another_project_with_a_module_of_the_same_name(self) -> None:
        _write(self.root, "modules/foundation/blas/include/Project/blas/backend.hpp",
               "#include <Other/geometry/mesh.hpp>")
        self.assertEqual(list(rule_upward_includes(ModuleTree(self.root, K_LAYOUT))), [])

    def test_reports_an_upward_include_in_a_module_test(self) -> None:
        _write(self.root, "modules/foundation/blas/include/Project/blas/backend.hpp")
        _write(self.root, "tests/foundation/blas/test_backend.cpp", "#include <Project/geometry/mesh.hpp>")
        self.assertEqual(_messages(rule_upward_includes(ModuleTree(self.root, K_LAYOUT))),
                         ["foundation/blas includes up into domain/geometry"])

    def test_allows_a_test_to_include_an_umbrella(self) -> None:
        _write(self.root, "modules/domain/umbrella/include/Project/Project.hpp")
        _write(self.root, "modules/foundation/blas/include/Project/blas/backend.hpp")
        _write(self.root, "tests/foundation/blas/test_backend.cpp", "#include <Project/Project.hpp>")
        self.assertEqual(list(rule_upward_includes(ModuleTree(self.root, K_LAYOUT))), [])

    def test_resolves_the_tier_form_and_a_root_umbrella(self) -> None:
        _write(self.root, "modules/domain/umbrella/include/Project/domain.hpp")
        _write(self.root, "modules/domain/training/include/Project/domain/training/trainer.hpp")
        _write(self.root, "modules/foundation/training/include/Project/foundation/training/loop.hpp",
               "#include <Project/domain/training/trainer.hpp>\n#include <Project/domain.hpp>")
        self.assertEqual(_messages(rule_upward_includes(ModuleTree(self.root, K_LAYOUT))),
                         ["foundation/training includes up into domain/training",
                          "foundation/training includes up into domain/umbrella"])


class RuleDeclaredTiersTest(LayeringTest):
    """Covers the declared tier order and domains."""

    def test_reports_an_empty_order_and_an_undeclared_tier(self) -> None:
        _write(self.root, "modules/extra/tool/include/Project/tool/tool.hpp")
        self.assertEqual(len(list(rule_declared_tiers(ModuleTree(self.root, Layout(()))))), 1)
        self.assertEqual(_messages(rule_declared_tiers(ModuleTree(self.root, K_LAYOUT))),
                         ["tier 'extra' is not in K_TIER_ORDER"])


class RuleCrossDomainIncludesTest(LayeringTest):
    """Covers domain tiers, which share a rank and never include each other."""

    def test_reports_one_domain_including_another_and_allows_a_lower_tier(self) -> None:
        _write(self.root, "modules/core/tensor/include/Project/core/tensor/tensor.hpp")
        _write(self.root, "modules/vision/blocks/include/Project/vision/blocks/conv.hpp")
        _write(self.root, "modules/language/blocks/include/Project/language/blocks/attention.hpp",
               "#include <Project/vision/blocks/conv.hpp>\n#include <Project/core/tensor/tensor.hpp>")
        tree = ModuleTree(self.root, K_DOMAINS)
        self.assertEqual(_messages(rule_cross_domain_includes(tree)),
                         ["language/blocks of one domain includes vision/blocks"])
        self.assertEqual(list(rule_upward_includes(tree)), [])


class RuleSameTierIncludesTest(LayeringTest):
    """Covers the listed edges between modules of one tier."""

    def test_reports_an_edge_that_is_not_listed(self) -> None:
        _write(self.root, "modules/foundation/core/include/Project/core/core.hpp")
        _write(self.root, "modules/foundation/log/include/Project/log/log.hpp", "#include <Project/core/core.hpp>")
        self.assertEqual(len(list(rule_same_tier_includes(ModuleTree(self.root, K_LAYOUT)))), 1)
        listed = Layout(("foundation",), same_tier_edges=frozenset({("log", "core")}))
        self.assertEqual(list(rule_same_tier_includes(ModuleTree(self.root, listed))), [])

    def test_lets_an_umbrella_include_its_tier(self) -> None:
        _write(self.root, "modules/foundation/core/include/Project/core/core.hpp")
        _write(self.root, "modules/foundation/umbrella/include/Project/foundation.hpp", "#include <Project/core/core.hpp>")
        self.assertEqual(list(rule_same_tier_includes(ModuleTree(self.root, K_LAYOUT))), [])


class BuildDeclarationTest(LayeringTest):
    """Covers the rules that read a module's declaring call in its build file."""

    def setUp(self) -> None:
        super().setUp()
        _write(self.root, "modules/foundation/core/include/Project/core/core.hpp")
        _write(self.root, "modules/foundation/core/CMakeLists.txt", "project_add_module(NAME core INTERNAL)")
        _write(self.root, "modules/foundation/log/include/Project/log/log.hpp", "#include <Project/core/core.hpp>")
        _write(self.root, "modules/foundation/log/source/log.cpp", "#include <Project/core/core.hpp>")

    def test_reports_an_include_the_build_file_does_not_list(self) -> None:
        _write(self.root, "modules/foundation/log/CMakeLists.txt", "project_add_module(NAME log PRIVATE_DEPENDS core)")
        self.assertEqual(_messages(rule_declared_dependencies(ModuleTree(self.root, K_BUILD))),
                         ["log includes core, which its CMakeLists.txt does not list in DEPENDS"])

    def test_reports_a_listed_module_nothing_includes(self) -> None:
        _write(self.root, "modules/foundation/log/CMakeLists.txt",
               "project_add_module(NAME log DEPENDS core PRIVATE_DEPENDS clock)")
        self.assertEqual(_messages(rule_unused_dependencies(ModuleTree(self.root, K_BUILD))),
                         ["log lists clock in PRIVATE_DEPENDS and no file includes it"])

    def test_reports_an_installed_header_that_includes_an_internal_module(self) -> None:
        _write(self.root, "modules/foundation/log/CMakeLists.txt", "project_add_module(NAME log DEPENDS core)")
        self.assertEqual(_messages(rule_public_headers(ModuleTree(self.root, K_BUILD))),
                         ["log is installed and includes core, an internal module whose headers are not"])

    def test_skips_the_build_rules_when_no_function_is_named(self) -> None:
        tree = ModuleTree(self.root, Layout(("foundation",)))
        self.assertEqual(list(rule_declared_dependencies(tree)) + list(rule_public_headers(tree)), [])


if __name__ == "__main__":
    unittest.main()

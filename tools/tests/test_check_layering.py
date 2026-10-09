# test_check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the tier, domain and private reach rules of the layering checker."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import run  # noqa: E402
from layering.check_layering import (  # noqa: E402
    K_CHECKER,
    K_LAYOUT,
    Layout,
    SourceTree,
    rule_cross_domain_includes,
    rule_declared_tiers,
    rule_private_reach,
    rule_upward_includes,
)

K_REPOSITORY = Path(__file__).resolve().parents[2]
K_TEST_LAYOUT = Layout(
    tier_order=("core", "layers", "domains", "programs"),
    domain_tier="domains",
    module_tiers={"core": "core", "nn": "layers", "vision": "domains", "language": "domains"},
    library_roots=("include/SushiAI", "src"),
    scoped_roots={
        "tests/common": "layers",
        "tests/functional": "layers",
        "tests/vision": "domains",
        "apps": "programs",
    },
)


def _tree(folder: str, files: dict[str, str], layout: Layout = K_TEST_LAYOUT) -> SourceTree:
    """Returns a tree holding the given sources and a folder for every entry of the layout."""
    root = Path(folder)
    for module in layout.module_tiers:
        (root / layout.library_roots[0] / module).mkdir(parents=True, exist_ok=True)
    for label in layout.scoped_roots:
        (root / label).mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return SourceTree(root, layout)


def _places(tree: SourceTree, issues: list) -> list[tuple[str, int]]:
    """Returns the file and the line of each issue, the file relative to the tree's root."""
    return [(issue.path.relative_to(tree.root).as_posix(), issue.line) for issue in issues]


def _run(folder: str, *flags: str) -> tuple[int, str]:
    """Returns the exit code and the printed text of one command line run over a folder."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = run(K_CHECKER, [folder, *flags])
    return code, printed.getvalue()


class UpwardIncludeTest(unittest.TestCase):
    """Checks that an include may point down or sideways and never up."""

    def test_is_registered(self) -> None:
        """Runs every rule as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_declared_tiers"], rule_declared_tiers)
        self.assertIs(K_CHECKER.rules["rule_upward_includes"], rule_upward_includes)
        self.assertIs(K_CHECKER.rules["rule_cross_domain_includes"], rule_cross_domain_includes)
        self.assertIs(K_CHECKER.rules["rule_private_reach"], rule_private_reach)

    def test_a_downward_include_is_clean(self) -> None:
        """Reports nothing for a layer that includes core, its own tier and the umbrella."""
        files = {
            "include/SushiAI/nn/linear.hpp": (
                "#include <SushiAI/core/status.hpp>\n"
                "#include <SushiAI/nn/layer.hpp>\n"
                "#include <SushiAI/SushiAI.h>\n"
                "#include <vector>\n"
            ),
            "src/vision/detect/iou.cpp": "#include <SushiAI/nn/layer.hpp>\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_upward_includes(_tree(folder, files))), [])

    def test_an_upward_include_is_reported(self) -> None:
        """Reports the one include of a layer that names a domain, with its file and line."""
        files = {
            "src/nn/model_graph.cpp": (
                "#include <SushiAI/core/status.hpp>\n#include <SushiAI/vision/blocks/c3k.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_upward_includes(tree))
            self.assertEqual(_places(tree, issues), [("src/nn/model_graph.cpp", 2)])
            self.assertEqual(issues[0].message, "nn (layers) includes up into vision (domains)")

    def test_a_core_module_including_a_layer_is_reported(self) -> None:
        """Reports a header of tier 1 that names a module of tier 2."""
        files = {"include/SushiAI/core/shape.hpp": "#  include <SushiAI/nn/layer.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(
                _places(tree, list(rule_upward_includes(tree))),
                [("include/SushiAI/core/shape.hpp", 1)],
            )

    def test_a_core_test_including_a_domain_is_reported(self) -> None:
        """Reports a test of the shared binary and a shared test header that name vision."""
        files = {
            "tests/functional/unit/test_shape.cpp": "#include <SushiAI/vision/detect/iou.hpp>\n",
            "tests/common/test_utils.hpp": "\n#include <SushiAI/vision/imaging/image.hpp>\n",
            "tests/vision/test_iou.cpp": "#include <SushiAI/vision/detect/iou.hpp>\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(
                _places(tree, list(rule_upward_includes(tree))),
                [("tests/common/test_utils.hpp", 2), ("tests/functional/unit/test_shape.cpp", 1)],
            )

    def test_a_program_may_include_anything(self) -> None:
        """Reports nothing for an application that names a domain."""
        files = {"apps/domains_vision.cpp": "#include <SushiAI/vision/registration.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_upward_includes(_tree(folder, files))), [])

    def test_a_commented_include_is_skipped(self) -> None:
        """Reports nothing for an include that a line comment holds."""
        files = {"src/nn/a.cpp": "// #include <SushiAI/vision/blocks/c3k.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_upward_includes(_tree(folder, files))), [])


class CrossDomainTest(unittest.TestCase):
    """Checks that a domain names no other domain."""

    def test_a_domain_including_another_domain_is_reported(self) -> None:
        """Reports a file of language that includes vision, and no upward include for it."""
        files = {
            "src/language/tokens.cpp": (
                "#include <SushiAI/language/tokens.hpp>\n"
                "#include <SushiAI/vision/imaging/image.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_cross_domain_includes(tree))
            self.assertEqual(_places(tree, issues), [("src/language/tokens.cpp", 2)])
            self.assertEqual(list(rule_upward_includes(tree)), [])

    def test_a_domain_test_including_another_domain_is_reported(self) -> None:
        """Reports a test under tests/vision that includes language."""
        files = {"tests/vision/test_image.cpp": "#include <SushiAI/language/tokens.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(
                _places(tree, list(rule_cross_domain_includes(tree))),
                [("tests/vision/test_image.cpp", 1)],
            )

    def test_a_domain_including_itself_and_lower_tiers_is_clean(self) -> None:
        """Reports nothing for vision files that include vision, nn and core."""
        files = {
            "include/SushiAI/vision/blocks/c3k.hpp": (
                "#include <SushiAI/vision/blocks/bottleneck.hpp>\n"
                "#include <SushiAI/nn/layer.hpp>\n"
                "#include <SushiAI/core/status.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_cross_domain_includes(_tree(folder, files))), [])


class PrivateReachTest(unittest.TestCase):
    """Checks that a quoted include of a module file stays in its module's folder."""

    def test_a_quoted_include_inside_the_module_is_clean(self) -> None:
        """Reports nothing for a neighbour and for a file one folder up inside the module."""
        files = {
            "src/vision/blocks/c3k.cpp": (
                '#include "block_attributes.hpp"\n#include "../imaging/bilinear.hpp"\n'
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_private_reach(_tree(folder, files))), [])

    def test_a_quoted_include_that_leaves_the_module_is_reported(self) -> None:
        """Reports a quoted path that ends in another module's folder."""
        files = {"src/nn/conv2d.cpp": '#include <vector>\n#include "../core/private.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_private_reach(tree))
            self.assertEqual(_places(tree, issues), [("src/nn/conv2d.cpp", 2)])
            self.assertEqual(
                issues[0].message, '"../core/private.hpp" leaves the folder of module nn'
            )

    def test_a_test_may_quote_the_shared_headers(self) -> None:
        """Reports nothing for a test that quotes a header of tests/common."""
        files = {"tests/functional/unit/test_a.cpp": '#include "../../common/test_utils.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_private_reach(_tree(folder, files))), [])

    def test_a_root_header_belongs_to_no_module(self) -> None:
        """Gives a file directly under a library root no scope, so no rule reads it."""
        files = {
            "include/SushiAI/SushiAI.h": '#include "nn/layer.hpp"\n#include "../other.hpp"\n',
            "src/version.cpp": "#include <SushiAI/vision/registration.hpp>\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual([len(includes) for includes in tree.includes.values() if includes], [])
            self.assertNotIn("SushiAI.h", tree.module_folders)
            for rule in K_CHECKER.rules.values():
                self.assertEqual(list(rule(tree)), [])


class DeclaredTiersTest(unittest.TestCase):
    """Checks that the tables and the folders of the tree agree."""

    def test_a_tree_that_matches_the_tables_is_clean(self) -> None:
        """Reports nothing when every entry has a folder and every folder an entry."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_declared_tiers(_tree(folder, {}))), [])

    def test_an_undeclared_module_folder_is_reported(self) -> None:
        """Reports a folder under a library root that the module table lacks."""
        files = {"src/physics/world.cpp": "#include <SushiAI/core/status.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_declared_tiers(tree))
            self.assertEqual(_places(tree, issues), [("src/physics", 1)])
            self.assertEqual(issues[0].message, "module folder 'physics' is not in K_MODULE_TIERS")

    def test_a_table_entry_without_a_folder_is_reported(self) -> None:
        """Reports a declared module and a declared scoped root that the tree lacks."""
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, {})
            (tree.root / "include/SushiAI/language").rmdir()
            (tree.root / "apps").rmdir()
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(tree)],
                [
                    "module 'language' is declared and has no folder",
                    "scoped root 'apps' is declared and missing",
                ],
            )

    def test_a_tier_outside_the_order_is_reported(self) -> None:
        """Reports a module and a scoped root whose tier the order does not hold."""
        layout = Layout(
            tier_order=("core", "layers"),
            domain_tier="domains",
            module_tiers={"core": "core", "nn": "middle"},
            library_roots=("include/SushiAI", "src"),
            scoped_roots={"apps": "programs"},
        )
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(_tree(folder, {}, layout))],
                [
                    "tier 'middle' of 'nn' is not in K_TIER_ORDER",
                    "tier 'programs' of 'apps' is not in K_TIER_ORDER",
                ],
            )

    def test_a_domain_test_root_named_after_no_domain_is_reported(self) -> None:
        """Reports a scoped root of the domain tier whose folder name is no domain module."""
        layout = Layout(
            tier_order=("core", "domains"),
            domain_tier="domains",
            module_tiers={"core": "core", "vision": "domains"},
            library_roots=("include/SushiAI", "src"),
            scoped_roots={"tests/pictures": "domains"},
        )
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(_tree(folder, {}, layout))],
                ["scoped root 'tests/pictures' is named after no domain"],
            )

    def test_the_declared_layout_follows_the_design(self) -> None:
        """Places train and io beside nn, vision above them and the shared tests below vision."""
        self.assertEqual(K_LAYOUT.module_tiers["train"], K_LAYOUT.module_tiers["nn"])
        self.assertEqual(K_LAYOUT.module_tiers["io"], K_LAYOUT.module_tiers["nn"])
        self.assertEqual(K_LAYOUT.module_tiers["vision"], K_LAYOUT.domain_tier)
        self.assertLess(
            K_LAYOUT.rank(K_LAYOUT.scoped_roots["tests/common"]), K_LAYOUT.rank("domains")
        )


class ExitCodeTest(unittest.TestCase):
    """Checks what the command line returns."""

    def test_exits_with_findings_and_names_the_rule(self) -> None:
        """Returns 1 and prints the path, the line and the rule of an upward include."""
        files = {"src/core/shape.cpp": "#include <SushiAI/nn/layer.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            _tree(folder, files, K_LAYOUT)
            code, printed = _run(folder)
            self.assertEqual(code, 1)
            self.assertIn("src/core/shape.cpp:1: rule_upward_includes:", printed)

    def test_reports_counts_and_exits_clean(self) -> None:
        """Returns 0 under --report and prints one count per rule."""
        files = {"src/core/shape.cpp": "#include <SushiAI/nn/layer.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            _tree(folder, files, K_LAYOUT)
            code, printed = _run(folder, "--report")
            self.assertEqual(code, 0)
            self.assertIn("rule_upward_includes: 1", printed)

    def test_this_repository_is_clean(self) -> None:
        """Returns 0 and prints nothing for the repository that holds this checker."""
        code, printed = _run(str(K_REPOSITORY))
        self.assertEqual((code, printed), (0, ""))


if __name__ == "__main__":
    unittest.main()

# test_check_layering.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the tier, domain and private reach rules of the layering checker."""

from __future__ import annotations

import contextlib
import io
import re
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
K_TIER_TABLE = K_REPOSITORY / "cmake" / "SushiAILayers.cmake"
K_TEST_LAYOUT = Layout(
    tier_order=("core", "layers", "vision", "language"),
    domain_tiers=("vision", "language"),
    module_tiers={
        "core": "core",
        "nn": "layers",
        "vision/blocks": "vision",
        "vision/imaging": "vision",
        "language/tokens": "language",
    },
    scoped_roots={"tests/common": "layers", "tests": None, "applications": None},
)


def _module_folder(key: str, tier: str) -> str:
    """Returns the folder of a module key, relative to the repository root."""
    return f"modules/{tier}/{key.split('/')[-1]}"


def _tree(folder: str, files: dict[str, str], layout: Layout = K_TEST_LAYOUT) -> SourceTree:
    """Returns a tree holding the given sources and a folder for every entry of the layout.

    Every module gets the public header folder its key names under include/SushiAI.
    """
    root = Path(folder)
    for key, tier in layout.module_tiers.items():
        public = root / _module_folder(key, tier) / "include" / "SushiAI" / key
        public.mkdir(parents=True, exist_ok=True)
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


def _cmake_list(name: str) -> list[str]:
    """Returns the words of one set() call of the build's tier table."""
    text = re.sub(r"#.*", "", K_TIER_TABLE.read_text(encoding="utf-8"))
    match = re.search(rf"set\(\s*{name}\s([^)]*)\)", text)
    return match.group(1).split() if match else []


class ModuleDiscoveryTest(unittest.TestCase):
    """Checks which folders are modules and which module a header belongs to."""

    def test_a_domain_module_is_keyed_by_tier_and_name(self) -> None:
        """Keys a module of a domain tier `<tier>/<module>` and any other by its name."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                sorted(_tree(folder, {}).modules),
                ["core", "language/tokens", "nn", "vision/blocks", "vision/imaging"],
            )

    def test_two_domains_may_hold_a_module_of_one_name(self) -> None:
        """Keeps vision/blocks and language/blocks apart and resolves each one's headers."""
        files = {
            "modules/language/blocks/include/SushiAI/language/blocks/a.hpp": "",
            "modules/vision/blocks/include/SushiAI/vision/blocks/a.hpp": "",
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(tree.resolve("SushiAI/language/blocks/a.hpp").key, "language/blocks")
            self.assertEqual(tree.resolve("SushiAI/vision/blocks/a.hpp").key, "vision/blocks")

    def test_a_header_resolves_to_the_module_that_owns_its_folder(self) -> None:
        """Resolves a header by the nearest folder one module alone holds, else to None."""
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, {})
            self.assertEqual(tree.resolve("SushiAI/nn/layer.hpp").key, "nn")
            self.assertEqual(tree.resolve("SushiAI/vision/imaging/a/b.hpp").key, "vision/imaging")
            self.assertIsNone(tree.resolve("SushiAI/vision/unknown.hpp"))
            self.assertIsNone(tree.resolve("vector"))

    def test_the_single_headers_of_this_repository_resolve(self) -> None:
        """Resolves the umbrellas, the version header and the registration header."""
        tree = SourceTree(K_REPOSITORY, K_LAYOUT)
        expected = {
            "SushiAI/SushiAI.h": "umbrella",
            "SushiAI/version.hpp": "version",
            "SushiAI/vision/vision.h": "vision/umbrella",
            "SushiAI/vision/registration.hpp": "vision/registration",
            "SushiAI/nn/conv2d.hpp": "nn",
            "SushiAI/vision/detect/iou.hpp": "vision/detect",
        }
        for target, key in expected.items():
            owner = tree.resolve(target)
            self.assertEqual(owner.key if owner else None, key, target)


class UpwardIncludeTest(unittest.TestCase):
    """Checks that an include may point down or sideways and never up."""

    def test_is_registered(self) -> None:
        """Runs every rule as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_declared_tiers"], rule_declared_tiers)
        self.assertIs(K_CHECKER.rules["rule_upward_includes"], rule_upward_includes)
        self.assertIs(K_CHECKER.rules["rule_cross_domain_includes"], rule_cross_domain_includes)
        self.assertIs(K_CHECKER.rules["rule_private_reach"], rule_private_reach)

    def test_a_downward_include_is_clean(self) -> None:
        """Reports nothing for a layer that includes core and itself, or a domain a layer."""
        files = {
            "modules/layers/nn/include/SushiAI/nn/linear.hpp": (
                "#include <SushiAI/core/status.hpp>\n"
                "#include <SushiAI/nn/layer.hpp>\n"
                "#include <vector>\n"
            ),
            "modules/vision/blocks/source/c3k.cpp": "#include <SushiAI/nn/layer.hpp>\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_upward_includes(_tree(folder, files))), [])

    def test_an_upward_include_is_reported(self) -> None:
        """Reports the one include of a layer that names a domain, with its file and line."""
        files = {
            "modules/layers/nn/source/model_graph.cpp": (
                "#include <SushiAI/core/status.hpp>\n#include <SushiAI/vision/blocks/c3k.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_upward_includes(tree))
            self.assertEqual(
                _places(tree, issues), [("modules/layers/nn/source/model_graph.cpp", 2)]
            )
            self.assertEqual(
                issues[0].message, "nn (layers) includes up into vision/blocks (vision)"
            )

    def test_a_core_module_including_a_layer_is_reported(self) -> None:
        """Reports a header of tier 1 that names a module of tier 2."""
        name = "modules/core/core/include/SushiAI/core/shape.hpp"
        files = {name: "#  include <SushiAI/nn/layer.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(_places(tree, list(rule_upward_includes(tree))), [(name, 1)])

    def test_a_quoted_include_of_a_public_header_is_ranked_too(self) -> None:
        """Reports an upward include that is written with quotes."""
        name = "modules/core/core/source/shape.cpp"
        files = {name: '#include "SushiAI/nn/layer.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(_places(tree, list(rule_upward_includes(tree))), [(name, 1)])

    def test_a_module_test_is_bound_to_its_module_tier(self) -> None:
        """Reports a test in a layer module's tests/ that names a domain."""
        name = "modules/layers/nn/tests/test_conv2d.cpp"
        files = {name: "#include <SushiAI/nn/conv2d.hpp>\n#include <SushiAI/vision/blocks/a.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(_places(tree, list(rule_upward_includes(tree))), [(name, 2)])

    def test_the_shared_test_headers_name_no_domain(self) -> None:
        """Reports a header of tests/common that names a domain, and no other root test."""
        files = {
            "tests/common/test_utils.hpp": (
                "#include <SushiAI/nn/layer.hpp>\n#include <SushiAI/vision/imaging/image.hpp>\n"
            ),
            "tests/unit/test_shape.cpp": "#include <SushiAI/vision/imaging/image.hpp>\n",
            "tests/integration/test_all.cpp": "#include <SushiAI/language/tokens/a.hpp>\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_upward_includes(tree))
            self.assertEqual(_places(tree, issues), [("tests/common/test_utils.hpp", 2)])
            self.assertEqual(
                issues[0].message,
                "tests/common (layers) includes up into vision/imaging (vision)",
            )

    def test_a_program_may_include_anything(self) -> None:
        """Reports nothing for an application that names two domains."""
        files = {
            "applications/programs/source/domains.cpp": (
                "#include <SushiAI/vision/blocks/a.hpp>\n#include <SushiAI/language/tokens/a.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            for rule in K_CHECKER.rules.values():
                self.assertEqual(list(rule(tree)), [])

    def test_a_commented_include_is_skipped(self) -> None:
        """Reports nothing for an include that a line comment holds."""
        files = {"modules/layers/nn/source/a.cpp": "// #include <SushiAI/vision/blocks/c3k.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_upward_includes(_tree(folder, files))), [])


class CrossDomainTest(unittest.TestCase):
    """Checks that a domain names no other domain."""

    def test_a_domain_including_another_domain_is_reported(self) -> None:
        """Reports a file of language that includes vision, and no upward include for it."""
        name = "modules/language/tokens/source/tokens.cpp"
        files = {
            name: (
                "#include <SushiAI/language/tokens/tokens.hpp>\n"
                "#include <SushiAI/vision/imaging/image.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_cross_domain_includes(tree))
            self.assertEqual(_places(tree, issues), [(name, 2)])
            self.assertEqual(
                issues[0].message, "language/tokens of domain language includes vision/imaging"
            )
            self.assertEqual(list(rule_upward_includes(tree)), [])

    def test_the_lower_domain_of_the_order_may_not_name_the_higher(self) -> None:
        """Reports vision naming language as a cross domain include, not as an upward one."""
        name = "modules/vision/blocks/source/a.cpp"
        files = {name: "#include <SushiAI/language/tokens/tokens.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(_places(tree, list(rule_cross_domain_includes(tree))), [(name, 1)])
            self.assertEqual(list(rule_upward_includes(tree)), [])

    def test_a_domain_test_including_another_domain_is_reported(self) -> None:
        """Reports a test in a vision module's tests/ that includes language."""
        name = "modules/vision/imaging/tests/test_image.cpp"
        files = {name: "#include <SushiAI/language/tokens/tokens.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(_places(tree, list(rule_cross_domain_includes(tree))), [(name, 1)])

    def test_a_domain_including_itself_and_lower_tiers_is_clean(self) -> None:
        """Reports nothing for a vision file that includes vision, nn and core."""
        files = {
            "modules/vision/blocks/include/SushiAI/vision/blocks/c3k.hpp": (
                "#include <SushiAI/vision/blocks/bottleneck.hpp>\n"
                "#include <SushiAI/vision/imaging/image.hpp>\n"
                "#include <SushiAI/nn/layer.hpp>\n"
                "#include <SushiAI/core/status.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(list(rule_cross_domain_includes(tree)), [])
            self.assertEqual(list(rule_upward_includes(tree)), [])


class PrivateReachTest(unittest.TestCase):
    """Checks that no include names another module's source/ or tests/ or leaves its module."""

    def test_a_quoted_include_inside_the_module_is_clean(self) -> None:
        """Reports nothing for a neighbour and for the module's own source/ seen from tests/."""
        files = {
            "modules/vision/blocks/source/c3k.cpp": '#include "block_attributes.hpp"\n',
            "modules/vision/blocks/tests/test_c3k.cpp": '#include "../source/attributes.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_private_reach(_tree(folder, files))), [])

    def test_a_quoted_include_into_another_source_folder_is_reported(self) -> None:
        """Reports a relative path that ends in the source/ of a module of the same domain."""
        name = "modules/vision/blocks/source/c3k.cpp"
        files = {name: '#include <vector>\n#include "../../imaging/source/bilinear.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_private_reach(tree))
            self.assertEqual(_places(tree, issues), [(name, 2)])
            self.assertEqual(
                issues[0].message,
                '"../../imaging/source/bilinear.hpp" reaches source/ of module vision/imaging',
            )

    def test_an_angle_include_into_a_private_folder_is_reported(self) -> None:
        """Reports angle includes that name another module's source/ and its tests/."""
        name = "modules/layers/nn/source/conv2d.cpp"
        files = {
            name: (
                "#include <core/source/private.hpp>\n"
                "#include <modules/vision/blocks/tests/fixture.hpp>\n"
                "#include <nn/source/own.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_private_reach(tree))
            self.assertEqual(_places(tree, issues), [(name, 1), (name, 2)])
            self.assertEqual(
                issues[1].message,
                '"modules/vision/blocks/tests/fixture.hpp" reaches tests/ of module vision/blocks',
            )

    def test_a_quoted_include_that_leaves_the_module_is_reported(self) -> None:
        """Reports a relative path that ends outside the module, here in a public header."""
        name = "modules/layers/nn/source/conv2d.cpp"
        files = {name: '#include "../../../core/core/include/SushiAI/core/status.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_private_reach(tree))
            self.assertEqual(_places(tree, issues), [(name, 1)])
            self.assertEqual(
                issues[0].message,
                '"../../../core/core/include/SushiAI/core/status.hpp" '
                "leaves the folder of module nn",
            )

    def test_a_root_test_may_quote_the_shared_headers(self) -> None:
        """Reports nothing for a root test that quotes a header of tests/common."""
        files = {"tests/unit/test_a.cpp": '#include "../common/test_utils.hpp"\n'}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_private_reach(_tree(folder, files))), [])

    def test_a_root_test_reaching_a_source_folder_is_reported(self) -> None:
        """Reports a root test and a program that quote a file of a module's source/."""
        files = {
            "tests/unit/test_a.cpp": '#include "../../modules/layers/nn/source/private.hpp"\n',
            "applications/programs/source/a.cpp": '#include "nn/source/private.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(
                _places(tree, list(rule_private_reach(tree))),
                [("tests/unit/test_a.cpp", 1), ("applications/programs/source/a.cpp", 1)],
            )

    def test_a_file_of_tests_common_is_read_once(self) -> None:
        """Gives a file under tests/common to that scope alone, not to tests/ as well."""
        files = {"tests/common/test_utils.hpp": "#include <SushiAI/core/status.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            self.assertEqual(len(tree.includes["tests/common"]), 1)
            self.assertEqual(len(tree.includes["tests"]), 0)


class DeclaredTiersTest(unittest.TestCase):
    """Checks that the tables and the folders of the tree agree."""

    def test_a_tree_that_matches_the_tables_is_clean(self) -> None:
        """Reports nothing when every entry has a folder and every folder an entry."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_declared_tiers(_tree(folder, {}))), [])

    def test_an_undeclared_module_folder_is_reported(self) -> None:
        """Reports a folder under a tier that the module table lacks, by its key."""
        files = {
            "modules/core/physics/source/world.cpp": "#include <SushiAI/core/status.hpp>\n",
            "modules/vision/video/source/clip.cpp": "",
        }
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_declared_tiers(tree))
            self.assertEqual(
                _places(tree, issues), [("modules/core/physics", 1), ("modules/vision/video", 1)]
            )
            self.assertEqual(
                [issue.message for issue in issues],
                [
                    "module folder 'physics' is not in K_MODULE_TIERS",
                    "module folder 'vision/video' is not in K_MODULE_TIERS",
                ],
            )

    def test_a_module_under_another_tier_than_declared_is_reported(self) -> None:
        """Reports a module the table puts in layers whose folder sits under core."""
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, {})
            (tree.root / "modules/layers/nn").rename(tree.root / "modules/core/nn")
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(tree)],
                ["module 'nn' is declared layers and sits under core"],
            )

    def test_a_module_name_held_by_two_tiers_is_reported(self) -> None:
        """Reports the second folder of a key that two tiers below the domains share."""
        files = {"modules/layers/core/source/a.cpp": ""}
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, files)
            issues = list(rule_declared_tiers(tree))
            self.assertEqual(_places(tree, issues), [("modules/layers/core", 1)])
            self.assertEqual(issues[0].message, "module key 'core' names a second folder")

    def test_a_table_entry_without_a_folder_is_reported(self) -> None:
        """Reports a declared module and a declared scoped root that the tree lacks."""
        with tempfile.TemporaryDirectory() as folder:
            tree = _tree(folder, {})
            for path in sorted((tree.root / "modules/language").rglob("*"), reverse=True):
                path.rmdir()
            (tree.root / "applications").rmdir()
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(tree)],
                [
                    "module 'language/tokens' has no folder",
                    "scoped root 'applications' is declared and missing",
                ],
            )

    def test_a_tier_outside_the_order_is_reported(self) -> None:
        """Reports a domain, a module and a scoped root whose tier the order does not hold."""
        layout = Layout(
            tier_order=("core", "layers"),
            domain_tiers=("vision",),
            module_tiers={"core": "core", "nn": "middle"},
            scoped_roots={"tests/common": "upper", "tests": None},
        )
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(_tree(folder, {}, layout))],
                [
                    "domain tier 'vision' is not in K_TIER_ORDER",
                    "tier 'middle' of 'nn' is not in K_TIER_ORDER",
                    "tier 'upper' of 'tests/common' is not in K_TIER_ORDER",
                ],
            )

    def test_a_scoped_root_bound_to_a_domain_is_reported(self) -> None:
        """Reports a root outside the modules that the table ties to one domain."""
        layout = Layout(
            tier_order=("core", "vision"),
            domain_tiers=("vision",),
            module_tiers={"core": "core"},
            scoped_roots={"tests/common": "vision"},
        )
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                [issue.message for issue in rule_declared_tiers(_tree(folder, {}, layout))],
                ["scoped root 'tests/common' is bound to domain 'vision'"],
            )

    def test_the_tier_table_agrees_with_the_build(self) -> None:
        """Holds the order, the domains and the module tiers cmake/SushiAILayers.cmake holds."""
        pairs = _cmake_list("SUSHIAI_MODULE_TIERS")
        self.assertEqual(tuple(_cmake_list("SUSHIAI_TIER_ORDER")), K_LAYOUT.tier_order)
        self.assertEqual(tuple(_cmake_list("SUSHIAI_DOMAIN_TIERS")), K_LAYOUT.domain_tiers)
        self.assertEqual(dict(zip(pairs[0::2], pairs[1::2])), dict(K_LAYOUT.module_tiers))
        self.assertGreater(len(pairs), 0)

    def test_the_declared_layout_follows_the_design(self) -> None:
        """Places train and io beside nn, vision above them and the shared tests below vision."""
        self.assertEqual(K_LAYOUT.module_tiers["train"], K_LAYOUT.module_tiers["nn"])
        self.assertEqual(K_LAYOUT.module_tiers["io"], K_LAYOUT.module_tiers["nn"])
        self.assertIn(K_LAYOUT.module_tiers["vision/detect"], K_LAYOUT.domain_tiers)
        self.assertLess(
            K_LAYOUT.rank(K_LAYOUT.scoped_roots["tests/common"]), K_LAYOUT.rank("vision")
        )
        self.assertIsNone(K_LAYOUT.scoped_roots["tests"])
        self.assertIsNone(K_LAYOUT.scoped_roots["applications"])


class ExitCodeTest(unittest.TestCase):
    """Checks what the command line returns."""

    def test_exits_with_findings_and_names_the_rule(self) -> None:
        """Returns 1 and prints the path, the line and the rule of an upward include."""
        files = {"modules/core/core/source/shape.cpp": "#include <SushiAI/nn/layer.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            _tree(folder, files, K_LAYOUT)
            code, printed = _run(folder)
            self.assertEqual(code, 1)
            self.assertIn(
                "modules/core/core/source/shape.cpp:1: rule_upward_includes:", printed
            )

    def test_reports_counts_and_exits_clean(self) -> None:
        """Returns 0 under --report and prints one count per rule."""
        files = {"modules/core/core/source/shape.cpp": "#include <SushiAI/nn/layer.hpp>\n"}
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

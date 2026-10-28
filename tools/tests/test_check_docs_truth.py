# test_check_docs_truth.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests each rule of the docs truth checker on a manual it must report and one it must not."""

from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_docs_layout import K_DESIGN_LINE_CEILING  # noqa: E402
from documentation.check_docs_truth import (  # noqa: E402
    K_CHECKER,
    K_LINE_CEILINGS,
    K_PASSAGE_FLOOR,
    K_SOURCE_FOLDERS,
    Manual,
    PageKind,
    main,
    rule_allow_list,
    rule_cited_paths,
    rule_duplicated_passages,
    rule_em_dash,
    rule_example_source,
    rule_history_words,
    rule_named_symbols,
    rule_page_length,
)

K_SKILL = Path(__file__).resolve().parents[2] / "skills" / "documentation" / "SKILL.md"
K_CEILING_ROW = re.compile(r"^\| ([^|]+?) \| ([\d ]+) lines \|", re.MULTILINE)
K_SOURCE = {
    "modules/core/engine/include/Project/engine/world.hpp": (
        "namespace Project { class PhysicsWorld { void add_body(); }; }\n"
        "#define PR_ENABLE_TRACE 1\nconstexpr int K_BODY_CEILING = 8;\n"
    ),
    "applications/examples/quickstart.cpp": (
        "#include <Project/engine/world.hpp>\nint main()\n{\n    Project::PhysicsWorld world;\n"
        "    world.add_body();\n}\n"
    ),
}
K_PASSAGE = "The world advances every body by one fixed step and then settles contacts. " * 5


def _manual(folder: str, files: dict[str, str]) -> Manual:
    """Returns the manual of a repository that holds the test source and the given files."""
    root = Path(folder)
    for name, text in {**K_SOURCE, **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Manual(root, K_SOURCE_FOLDERS)


def _found(
    rule: Callable[[Manual], Iterator[Issue]], files: dict[str, str]
) -> list[tuple[str, int]]:
    """Returns the root-relative path and the line of each issue a rule yields for the files."""
    with tempfile.TemporaryDirectory() as folder:
        manual = _manual(folder, files)
        return [
            (issue.path.relative_to(manual.root).as_posix(), issue.line)
            for issue in rule(manual)
        ]


class RuleTableTest(unittest.TestCase):
    """Checks that every rule runs as part of the checker."""

    def test_registers_every_rule(self) -> None:
        """Holds each rule function under its own name in the rule table."""
        rules = (
            rule_named_symbols, rule_cited_paths, rule_example_source, rule_history_words,
            rule_page_length, rule_duplicated_passages, rule_em_dash, rule_allow_list,
        )
        self.assertEqual(dict(K_CHECKER.rules), {rule.__name__: rule for rule in rules})


class NamedSymbolsTest(unittest.TestCase):
    """Checks which names in code count as names the source must hold."""

    def test_reports_each_shape_of_absent_name(self) -> None:
        """Reports a qualified name, a call, a type, a snake name, a constant and a macro."""
        page = (
            "# Guide\n\n`Project::RuntimeContext`\n\n`try_run()`\n\n`SushiException`\n\n"
            "`get_queue`\n\n`K_GONE_CEILING`\n\n`K_BODY_CEILING` and `PR_ENABLE_TRACE`\n"
        )
        found = _found(rule_named_symbols, {"docs/guides/GUIDE.md": page})
        self.assertEqual(found, [("docs/guides/GUIDE.md", line) for line in (3, 5, 7, 9, 11)])

    def test_accepts_names_the_source_holds(self) -> None:
        """Reports nothing for names of the source, of the standard library or of a path."""
        page = (
            "# Guide\n\n`Project::PhysicsWorld::add_body()` takes a `std::vector<int>`\n"
            "and `sycl::queue`; see `docs/stale_name.md`, `stale_name.hpp`, `--stale_flag`\n"
            "and `rule_<what>`.\n\n```console\n$ stale_command --run StaleType\n```\n"
        )
        self.assertEqual(_found(rule_named_symbols, {"docs/guides/GUIDE.md": page}), [])

    def test_reads_a_source_block_without_its_own_names(self) -> None:
        """Reports the absent call in a block, not the names the block declares or comments."""
        page = (
            "# Core\n\n```cpp\n// calls stale_helper() first\nstruct MyKernel {};\n"
            "auto my_world = Project::PhysicsWorld();\nmy_world.remove_body(local_index);\n```\n"
        )
        found = _found(rule_named_symbols, {"modules/core/engine/README.md": page})
        self.assertEqual(found, [("modules/core/engine/README.md", 7)])

    def test_leaves_a_design_document_to_its_intent(self) -> None:
        """Reports nothing for a name a design document plans and the source lacks."""
        page = "# Topic\n**Status:** Open\n\n`PlannedScheduler` arrives in phase 2.\n"
        self.assertEqual(_found(rule_named_symbols, {"docs/design/TOPIC.md": page}), [])

    def test_accepts_an_allowed_name(self) -> None:
        """Reports nothing for a name an allow-list pattern matches whole."""
        files = {
            "docs/guides/GUIDE.md": "# Guide\n\n`vk::ImageView` and `pr_engine_tests`\n",
            "docs/.docs_truth_allow": "vk::.*  # Vulkan's own\npr_\\w+_tests  # built by CMake\n",
        }
        self.assertEqual(_found(rule_named_symbols, files), [])


class CitedPathsTest(unittest.TestCase):
    """Checks which cited paths must exist."""

    def test_reports_a_missing_path(self) -> None:
        """Reports a path under a top-level folder, an include and a path in a text block."""
        page = (
            "# Guide\n\n`modules/core/engine/source/gone.cpp`\n\n```cpp\n"
            "#include <Project/engine/gone.hpp>\n```\n\n```text\napplications/gone/\n```\n"
        )
        found = _found(rule_cited_paths, {"docs/guides/GUIDE.md": page})
        self.assertEqual(found, [("docs/guides/GUIDE.md", line) for line in (3, 6, 10)])

    def test_accepts_existing_and_foreign_paths(self) -> None:
        """Accepts a root path, an include path, a pattern, a foreign path and the allow-list."""
        page = (
            "# Guide\n\n`applications/examples/quickstart.cpp:4`, `Project/engine/world.hpp`,\n"
            "`modules/<tier>/<module>/`, `~/.claude/skills/`, `https://example.org/a.html`,\n"
            "`build/bin/quickstart.exe`, `read/write`, `engine/include/` and the optional\n"
            "`docs/.docs_truth_allow` and `<new root>/.lock`.\n"
        )
        self.assertEqual(_found(rule_cited_paths, {"docs/guides/GUIDE.md": page}), [])

    def test_resolves_a_module_readme_path_beside_it(self) -> None:
        """Accepts a path a module README writes from its own folder."""
        page = "# Engine\n\n`include/Project/engine/world.hpp`\n"
        self.assertEqual(_found(rule_cited_paths, {"modules/core/engine/README.md": page}), [])

    def test_reads_a_page_beside_a_test_file(self) -> None:
        """Reads every page under a source folder, not only its README."""
        page = "# test_world\n\n`modules/core/engine/source/gone.cpp`\n"
        self.assertEqual(_found(rule_cited_paths, {"tests/core/engine/test_world.md": page}),
                         [("tests/core/engine/test_world.md", 3)])


class ExampleSourceTest(unittest.TestCase):
    """Checks that an example in a manual page is taken from a file."""

    def test_reports_an_uncited_and_a_drifted_example(self) -> None:
        """Reports a block without a citation, a line its file lacks and a missing file."""
        page = (
            "# Guide\n\n```cpp\nworld.add_body();\n```\n\n```cpp\nworld.add_body();\n"
            "world.step();\n```\n\nSource: `applications/examples/quickstart.cpp`\n\n"
            "```cpp\nint x;\n```\nSource: `applications/examples/gone.cpp`\n"
        )
        found = _found(rule_example_source, {"docs/guides/GUIDE.md": page})
        self.assertEqual(found, [("docs/guides/GUIDE.md", line) for line in (3, 9, 17)])

    def test_accepts_a_cited_excerpt_and_a_marked_block(self) -> None:
        """Accepts lines of the cited file with an elision, and a block marked not compiled."""
        page = (
            "# Guide\n\n```cpp\nint main()\n{\n    // ...\n    world.add_body();\n}\n```\n\n"
            "Source: `applications/examples/quickstart.cpp`\n\n```cpp\nfuture_call();\n```\n"
            "Source: not compiled\n\n```console\n$ pr build\n```\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            manual = _manual(folder, {"docs/guides/GUIDE.md": page})
            self.assertEqual(list(rule_example_source(manual)), [])
            self.assertEqual(manual.unverified_examples(), 1)

    def test_exempts_a_module_readme(self) -> None:
        """Reports nothing for an uncited block in a module README."""
        page = "# Engine\n\n```cpp\nworld.add_body();\n```\n"
        self.assertEqual(_found(rule_example_source, {"modules/core/engine/README.md": page}), [])


class HistoryWordsTest(unittest.TestCase):
    """Checks which prose narrates history."""

    def test_reports_each_history_phrase(self) -> None:
        """Reports one issue per line that says what the code was."""
        page = (
            "# Guide\n\nThe world previously stepped twice.\nIt no longer does.\n"
            "It used to sort bodies.\nIt was renamed in March.\nFormerly a list.\n"
            "As of v0.3 it sorts.\nNew in v0.4.\nUp to v0.2 it did not.\nDeprecated since spring.\n"
        )
        found = _found(rule_history_words, {"docs/guides/GUIDE.md": page})
        self.assertEqual(found, [("docs/guides/GUIDE.md", line) for line in range(3, 12)])

    def test_accepts_present_tense_code_quotes_and_known_issues(self) -> None:
        """Accepts 'is used to', a phrase in code or quotes, and the backlog."""
        files = {
            "docs/guides/GUIDE.md": (
                "# Guide\n\nThe queue is used to submit work; `previously` is a word.\n"
                'A comment never says "we used to do X".\n\n```text\nno longer\n```\n'
            ),
            "docs/backlog/detail/KI-1.md": "# KI-1\n\nThe mixer no longer clips.\n",
        }
        self.assertEqual(_found(rule_history_words, files), [])


class PageLengthTest(unittest.TestCase):
    """Checks the line ceiling of each kind of page."""

    def test_reports_a_page_one_line_above_its_ceiling(self) -> None:
        """Reports the root README, a manual page and a module README above their ceilings."""
        files = {
            "README.md": "x\n" * (K_LINE_CEILINGS[PageKind.ROOT_README] + 1),
            "docs/guides/GUIDE.md": "x\n" * (K_LINE_CEILINGS[PageKind.MANUAL_PAGE] + 1),
            "modules/core/engine/README.md": "x\n" * (K_LINE_CEILINGS[PageKind.MODULE_README] + 1),
        }
        found = _found(rule_page_length, files)
        self.assertEqual([name for name, _ in found], list(files))

    def test_accepts_a_page_at_its_ceiling_and_leaves_design_to_its_checker(self) -> None:
        """Accepts pages at their ceilings and a design document of any length."""
        files = {
            "README.md": "x\n" * K_LINE_CEILINGS[PageKind.ROOT_README],
            "docs/guides/GUIDE.md": "x\n" * K_LINE_CEILINGS[PageKind.MANUAL_PAGE],
            "docs/design/TOPIC.md": "x\n" * (K_DESIGN_LINE_CEILING + 1),
        }
        self.assertEqual(_found(rule_page_length, files), [])

    @unittest.skipUnless(K_SKILL.is_file(), "the documentation skill is not in this repository")
    def test_ceilings_match_the_documentation_skill(self) -> None:
        """Holds the same four ceilings as the table in the documentation skill."""
        stated = {
            label: int(number.replace(" ", ""))
            for label, number in K_CEILING_ROW.findall(K_SKILL.read_text(encoding="utf-8"))
        }
        held = {kind.value: ceiling for kind, ceiling in K_LINE_CEILINGS.items()}
        held[PageKind.DESIGN_DOCUMENT.value] = K_DESIGN_LINE_CEILING
        self.assertEqual(stated, held)


class DuplicatedPassagesTest(unittest.TestCase):
    """Checks that a long paragraph has one home."""

    def test_reports_the_second_copy_whatever_its_wrapping(self) -> None:
        """Reports the later page of two that hold one paragraph wrapped differently."""
        files = {
            "docs/guides/A.md": f"# A\n\n{K_PASSAGE}\n",
            "docs/guides/B.md": "# B\n\n" + K_PASSAGE.replace(". ", ".\n") + "\n",
        }
        self.assertEqual(_found(rule_duplicated_passages, files), [("docs/guides/B.md", 3)])

    def test_reports_a_rule_restated_in_a_second_skill(self) -> None:
        """Reports a skill that copies a paragraph of another skill, and skips research notes."""
        files = {
            "skills/commits/SKILL.md": f"# Commits\n\n{K_PASSAGE}\n",
            "skills/git-workflow/SKILL.md": f"# Git\n\n{K_PASSAGE}\n",
            "skills/humanizer/research/SOURCES.md": f"# Sources\n\n{K_PASSAGE}\n",
        }
        self.assertEqual(_found(rule_duplicated_passages, files), [("skills/git-workflow/SKILL.md", 3)])

    def test_accepts_a_short_repeat_and_a_repeated_code_block(self) -> None:
        """Accepts a repeated paragraph under the floor and a block repeated in a fence."""
        short = K_PASSAGE[: K_PASSAGE_FLOOR - 1].strip()
        files = {
            "docs/guides/A.md": f"# A\n\n{short}\n\n```text\n{K_PASSAGE}\n```\n",
            "docs/guides/B.md": f"# B\n\n{short}\n\n```text\n{K_PASSAGE}\n```\n",
        }
        self.assertEqual(_found(rule_duplicated_passages, files), [])

    def test_accepts_a_generated_block_in_two_module_readmes(self) -> None:
        """Accepts the designs block the views write into every module a topic owns."""
        block = f"<!-- designs:begin -->\n{K_PASSAGE}\n<!-- designs:end -->"
        files = {
            "docs/guides/A.md": f"# A\n\n{block}\n",
            "docs/guides/B.md": f"# B\n\n{block}\n",
        }
        self.assertEqual(_found(rule_duplicated_passages, files), [])


class EmDashTest(unittest.TestCase):
    """Checks where an em dash is refused."""

    def test_reports_an_em_dash_in_prose(self) -> None:
        """Reports the prose line that holds the dash."""
        page = "# Guide\n\nThe world steps — and then settles.\n"
        found = _found(rule_em_dash, {"docs/guides/GUIDE.md": page})
        self.assertEqual(found, [("docs/guides/GUIDE.md", 3)])

    def test_accepts_code_the_status_line_and_the_changelog(self) -> None:
        """Accepts a dash in a span, in a fence, in a design status line and in the changelog."""
        files = {
            "docs/guides/GUIDE.md": "# Guide\n\n`a — b`\n\n```text\na — b\n```\n",
            "docs/design/TOPIC.md": "# Topic\n**Status:** Open — phase 2 of 4\n",
            "docs/reference/CHANGELOG.md": "# Changelog\n\n- 2026-09-15 — core: Added it (`x`).\n",
        }
        self.assertEqual(_found(rule_em_dash, files), [])


class AllowListTest(unittest.TestCase):
    """Checks that the allow-list stays justified and in use."""

    def test_reports_a_bare_a_broken_and_an_unused_entry(self) -> None:
        """Reports an entry without a reason, one that does not compile and one nothing uses."""
        files = {
            "docs/guides/GUIDE.md": "# Guide\n\n`vk::ImageView`\n",
            "docs/.docs_truth_allow": "vk::.*\n(  # broken\nglm::.*  # the maths library\n",
        }
        found = _found(rule_allow_list, files)
        self.assertEqual(found, [("docs/.docs_truth_allow", line) for line in (1, 2, 3)])

    def test_accepts_a_reasoned_entry_in_use_and_comment_lines(self) -> None:
        """Accepts an entry with a reason that a page uses, beside a comment and a blank line."""
        files = {
            "docs/guides/GUIDE.md": "# Guide\n\n`vk::ImageView`\n",
            "docs/.docs_truth_allow": "# third-party namespaces\n\nvk::.*  # Vulkan's own\n",
        }
        self.assertEqual(_found(rule_allow_list, files), [])


class CommandLineTest(unittest.TestCase):
    """Checks the options the checker reads beside the shared ones."""

    def test_source_option_replaces_the_default_folders(self) -> None:
        """Finds a name only when --source names the folder that holds it."""
        with tempfile.TemporaryDirectory() as folder:
            _manual(folder, {
                "docs/guides/GUIDE.md": "# Guide\n\n`custom_entry_point`\n",
                "runtime/entry.cpp": "void custom_entry_point();\n",
            })
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([folder, "--rule", "rule_named_symbols"]), 1)
                self.assertEqual(
                    main([folder, "--rule", "rule_named_symbols", "--source", "runtime"]), 0,
                )

    def test_checks_a_page_given_by_path(self) -> None:
        """Checks a page passed on the command line, whatever its name."""
        with tempfile.TemporaryDirectory() as folder:
            _manual(folder, {".git/HEAD": "", "notes/probe.md": "# Probe\n\n`modules/nope/nothing.hpp`\n"})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(Path(folder) / "notes/probe.md"), "--rule", "rule_cited_paths"]), 1)

    def test_report_counts_the_unverified_examples(self) -> None:
        """Prints the number of blocks marked not compiled after the rule counts."""
        page = "# Guide\n\n```cpp\nfuture_call();\n```\nSource: not compiled\n"
        with tempfile.TemporaryDirectory() as folder:
            _manual(folder, {"docs/guides/GUIDE.md": page})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([folder, "--report"]), 0)
            self.assertEqual(output.getvalue().splitlines()[-1], "unverified examples: 1")


if __name__ == "__main__":
    unittest.main()

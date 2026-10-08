# test_check_contract.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the site scanner, the rules and the notes of the error contract checker."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.check_contract import (  # noqa: E402
    K_CHECKER,
    K_COLUMNS,
    K_HEAD_LENGTH,
    K_SITE_TABLE,
    Repository,
    count_bridges,
    find_sites,
    is_frozen,
    main,
    module_of,
    note_bridge_count,
    note_frozen_growth,
    note_open_question,
    note_unclassified,
    rule_ratchet,
    rule_stale_row,
    rule_table_shape,
    rule_unlisted_site,
)

K_ONE_SITE = 'void f(int rank)\n{\n    SA_THROW_IF(rank > 4, "rank {} is too high", rank);\n}\n'
K_SECOND_SITE = 'void g(int rank)\n{\n    SA_THROW_IF(rank < 0, "rank {} is negative", rank);\n}\n'
K_TWO_SITES = K_ONE_SITE + K_SECOND_SITE


def _row(
    file: str,
    line: int,
    head: str,
    kind: str = "macro",
    mechanism: str = "status",
    question: str = "",
) -> str:
    """Returns one table row for a site."""
    cause = "unclassified" if mechanism == "unclassified" else "user-content"
    return "\t".join((file, str(line), "f", kind, head, cause, mechanism, question, "A reason."))


def _repository(folder: str, files: dict[str, str], rows: list[str] | None) -> Repository:
    """Returns a repository holding the given sources and, unless rows is None, a class table."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if rows is not None:
        table = root / K_SITE_TABLE
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("\n".join(["\t".join(K_COLUMNS), *rows]) + "\n", encoding="utf-8")
    return Repository(root)


def _run(folder: str, *flags: str) -> tuple[int, str]:
    """Returns the exit code and the printed text of one command line run over a folder."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main([folder, *flags])
    return code, printed.getvalue()


class SiteScannerTest(unittest.TestCase):
    """Checks what counts as a failure site and how its message head is cut."""

    def test_finds_a_macro_site_with_its_line_and_head(self) -> None:
        """Reports the call's line and the format string as the head."""
        sites = find_sites("src/core/a.cpp", K_ONE_SITE)
        self.assertEqual(
            [(site.line, site.kind, site.head) for site in sites],
            [(3, "macro", "rank {} is too high")],
        )

    def test_joins_adjacent_literals_and_cuts_the_head(self) -> None:
        """Joins the literals of a wrapped format string and keeps the first characters."""
        literals = '"' + "x" * 40 + '"\n    " and ' + "y" * 40 + '"'
        text = "SA_THROW_IF(a(b, c) || d,\n    " + literals + ", a);\n"
        head = find_sites("src/core/a.cpp", text)[0].head
        self.assertEqual(head, ("x" * 40 + " and " + "y" * 40)[:K_HEAD_LENGTH])

    def test_unescapes_a_quote_in_the_head(self) -> None:
        """Counts an escaped quote as one character of the head."""
        text = 'SA_THROW_IF(bad, "the name \\"{}\\" is taken", name);\n'
        self.assertEqual(find_sites("src/core/a.cpp", text)[0].head, 'the name "{}" is taken')

    def test_finds_a_throw_statement(self) -> None:
        """Reports a raw throw with the first literal of its statement."""
        text = 'catch (...)\n{\n    throw Error(format("no arena of {} bytes", size));\n}\n'
        sites = find_sites("src/core/a.cpp", text)
        self.assertEqual(
            [(site.line, site.kind, site.head) for site in sites],
            [(3, "throw", "no arena of {} bytes")],
        )

    def test_skips_comments_literals_and_the_definition(self) -> None:
        """Reports no site for a mention in a comment, in a literal or in a #define block."""
        text = (
            "// unreachable: SA_THROW_IF(true) always throws\n"
            "/* throw nothing */\n"
            "#define SA_THROW_IF(condition, ...) \\\n"
            "    do { if (condition) { throw ::Error(format(__VA_ARGS__)); } } while (0)\n"
            'const char* text = "they throw when SA_THROW_IF(x) fires";\n'
        )
        self.assertEqual(find_sites("include/SushiAI/core/common.hpp", text), [])

    def test_counts_a_macro_that_wraps_the_site_macro(self) -> None:
        """Reports the SA_THROW_IF in the body of another #define, continuation lines included."""
        text = (
            "#define MY_CHECK(condition) SA_THROW_IF(condition, \"wrapped\")\n"
            "#define MY_LONG_CHECK(condition) \\\n"
            "    SA_THROW_IF(condition, \"wrapped twice\")\n"
        )
        sites = find_sites("src/core/a.cpp", text)
        self.assertEqual(
            [(site.line, site.head) for site in sites], [(1, "wrapped"), (3, "wrapped twice")]
        )

    def test_keeps_a_digit_separator_out_of_literals(self) -> None:
        """Finds the site that follows a number written with digit separators."""
        text = "const int limit = 1'000'000;\n" + K_ONE_SITE
        self.assertEqual(len(find_sites("src/core/a.cpp", text)), 1)

K_TEST_FROZEN = ("src/graph/lowering.cpp", "src/optim/")
K_FROZEN_NAME = "contract.check_contract.K_FROZEN"


class PathTest(unittest.TestCase):
    """Checks the frozen list and the module of a path."""

    @mock.patch(K_FROZEN_NAME, K_TEST_FROZEN)
    def test_names_frozen_files_and_folders(self) -> None:
        """Treats a listed file and every file under a listed folder as frozen."""
        self.assertTrue(is_frozen("src/graph/lowering.cpp"))
        self.assertTrue(is_frozen("src/optim/adamw.cpp"))
        self.assertFalse(is_frozen("src/graph/builder.cpp"))

    def test_reads_the_module_from_the_path(self) -> None:
        """Returns the folder under src/ or under include/SushiAI/."""
        self.assertEqual(module_of("src/graph/builder.cpp"), "graph")
        self.assertEqual(module_of("include/SushiAI/nn/module.hpp"), "nn")
        self.assertEqual(module_of("include/SushiAI/SushiAI.h"), "(root)")
        self.assertEqual(module_of("apps/train.cpp"), "apps")
        self.assertEqual(module_of("examples/demo/main.cpp"), "examples")


class RatchetTest(unittest.TestCase):
    """Checks that a file may not hold more SA_THROW_IF sites than rows."""

    def test_is_registered(self) -> None:
        """Runs every rule as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_ratchet"], rule_ratchet)
        self.assertIs(K_CHECKER.rules["rule_unlisted_site"], rule_unlisted_site)
        self.assertIs(K_CHECKER.rules["rule_stale_row"], rule_stale_row)
        self.assertIs(K_CHECKER.rules["rule_table_shape"], rule_table_shape)

    def test_accepts_a_file_with_one_row_per_site(self) -> None:
        """Reports nothing when the counts agree."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high")]
            self.assertEqual(
                list(rule_ratchet(_repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows))), []
            )

    def test_reports_a_site_added_after_the_table(self) -> None:
        """Reports the first surplus site of a file that outgrew its rows."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high")]
            issues = list(rule_ratchet(_repository(folder, {"src/core/a.cpp": K_TWO_SITES}, rows)))
            self.assertEqual([(issue.path.name, issue.line) for issue in issues], [("a.cpp", 7)])

    def test_reads_the_apps_and_the_examples(self) -> None:
        """Reports a site without a row under apps/ and examples/, and none under tests/."""
        files = {
            "apps/train.cpp": K_ONE_SITE,
            "examples/demo.cpp": K_ONE_SITE,
            "tests/unit/test_a.cpp": K_ONE_SITE,
        }
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files, [])
            self.assertEqual(
                [issue.path.name for issue in rule_ratchet(repository)], ["train.cpp", "demo.cpp"]
            )
            self.assertEqual(len(list(rule_unlisted_site(repository))), 2)

    def test_reports_a_wrapper_macro_added_after_the_table(self) -> None:
        """Fails the ratchet for a new macro whose body calls SA_THROW_IF."""
        text = K_ONE_SITE + "#define MY_CHECK(condition) SA_THROW_IF(condition, \"wrapped\")\n"
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high")]
            repository = _repository(folder, {"src/core/a.cpp": text}, rows)
            self.assertEqual([issue.line for issue in rule_ratchet(repository)], [5])

    def test_does_not_count_a_throw_row_as_room_for_a_macro(self) -> None:
        """Compares SA_THROW_IF sites with the rows of kind macro only."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high", kind="throw")]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows)
            self.assertEqual(len(list(rule_ratchet(repository))), 1)

    @mock.patch(K_FROZEN_NAME, K_TEST_FROZEN)
    def test_only_notes_a_frozen_file(self) -> None:
        """Leaves a frozen file to the note and out of the failing rules."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/optim/a.cpp", 3, "rank {} is too high")]
            repository = _repository(folder, {"src/optim/a.cpp": K_TWO_SITES}, rows)
            self.assertEqual(list(rule_ratchet(repository)), [])
            self.assertEqual(list(rule_unlisted_site(repository)), [])
            self.assertEqual([issue.line for issue in note_frozen_growth(repository)], [7, 7])


class CoverageTest(unittest.TestCase):
    """Checks that every site has a row and every row a site."""

    def test_reports_a_site_without_a_row(self) -> None:
        """Reports the site whose head no row carries."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, [])
            issues = list(rule_unlisted_site(repository))
            self.assertEqual(
                [(issue.line, issue.message) for issue in issues],
                [(3, "no row for 'rank {} is too high'")],
            )

    def test_reports_a_row_without_a_site(self) -> None:
        """Reports a stale row at its line of the table."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row("src/core/a.cpp", 3, "rank {} is too high"),
                _row("src/core/gone.cpp", 9, "gone"),
            ]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows)
            issues = list(rule_stale_row(repository))
            self.assertEqual(
                [(issue.path.name, issue.line) for issue in issues], [(K_SITE_TABLE.name, 3)]
            )

    def test_matches_a_moved_site_by_its_head(self) -> None:
        """Accepts a row whose line hint is out of date."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 300, "rank {} is too high")]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows)
            self.assertEqual(
                list(rule_unlisted_site(repository)) + list(rule_stale_row(repository)), []
            )

    def test_reports_a_reworded_site_twice(self) -> None:
        """Reports a changed message as one unlisted site and one stale row."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is far too high")]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows)
            self.assertEqual(len(list(rule_unlisted_site(repository))), 1)
            self.assertEqual(len(list(rule_stale_row(repository))), 1)

    def test_pairs_equal_heads_one_to_one(self) -> None:
        """Reports the second of two equal sites when the table holds one row for them."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high")]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE + K_ONE_SITE}, rows)
            self.assertEqual([issue.line for issue in rule_unlisted_site(repository)], [7])


class TableShapeTest(unittest.TestCase):
    """Checks the header, the cells and the order of the class table."""

    def test_reports_a_missing_table(self) -> None:
        """Reports a repository that carries no table."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(list(rule_table_shape(_repository(folder, {}, None)))), 1)

    def test_accepts_a_sorted_table(self) -> None:
        """Reports nothing for well-formed rows in file and line order."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/a.cpp", 3, "a"), _row("src/a.cpp", 9, "b"), _row("src/b.cpp", 1, "c")]
            self.assertEqual(list(rule_table_shape(_repository(folder, {}, rows))), [])

    def test_reports_a_row_out_of_order(self) -> None:
        """Reports the row that sorts before its predecessor."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/a.cpp", 9, "b"), _row("src/a.cpp", 3, "a")]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [3]
            )

    def test_reports_a_malformed_row_and_an_unknown_class(self) -> None:
        """Reports a row with too few cells and a row with a mechanism the table does not know."""
        with tempfile.TemporaryDirectory() as folder:
            rows = ["src/a.cpp\t3\tf", _row("src/b.cpp", 1, "c", mechanism="maybe")]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [2, 3]
            )


    def test_accepts_a_row_that_waits_for_a_class(self) -> None:
        """Reports nothing for a row whose cause and mechanism are unclassified."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/a.cpp", 3, "a", mechanism="unclassified")]
            self.assertEqual(list(rule_table_shape(_repository(folder, {}, rows))), [])

    def test_reports_an_open_question_that_is_no_key(self) -> None:
        """Accepts keys joined by a semicolon and reports a cell that holds a sentence."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row("src/a.cpp", 3, "a", question="hand_built_graph;layout_proof"),
                _row("src/a.cpp", 9, "b", question="Who builds it?"),
            ]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [3]
            )


class NoteTest(unittest.TestCase):
    """Checks the notes, which are printed and never fail the run."""

    def test_counts_bridge_calls_outside_comments(self) -> None:
        """Counts a call and ignores a mention in a comment."""
        text = "// Detail::bridge(result) throws\nauto x = Detail::bridge(make());\n"
        self.assertEqual(count_bridges(text), 1)

    def test_counts_no_other_function_named_bridge(self) -> None:
        """Ignores a member call and an unqualified call of the same name."""
        self.assertEqual(count_bridges("auto x = other.bridge(a);\nauto y = bridge(b);\n"), 0)

    def test_reports_the_bridge_count_per_module(self) -> None:
        """Yields one line per module that bridges and one line with the total."""
        files = {
            "src/graph/a.cpp": "auto x = Detail::bridge(a());\nauto y = Detail::bridge(b());\n",
            "include/SushiAI/nn/b.hpp": "auto z = Detail::bridge(c());\n",
            "src/core/c.cpp": "int d = 0;\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            messages = [
                issue.message for issue in note_bridge_count(_repository(folder, files, []))
            ]
            self.assertEqual(messages, [
                "module graph: 2 bridge calls",
                "module nn: 1 bridge calls",
                "3 bridge calls; the last wave needs 0",
            ])

    def test_lists_a_row_that_waits_for_a_class(self) -> None:
        """Yields a note for a row whose mechanism is unclassified."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row("src/core/a.cpp", 3, "rank {} is too high", mechanism="unclassified")]
            repository = _repository(folder, {"src/core/a.cpp": K_ONE_SITE}, rows)
            self.assertEqual(len(list(note_unclassified(repository))), 1)


    def test_counts_the_rows_of_each_open_question(self) -> None:
        """Yields one note per key, at the first row that carries it, with its row count."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row("src/a.cpp", 3, "a"),
                _row("src/a.cpp", 5, "b", question="output_span"),
                _row("src/a.cpp", 9, "c", question="hand_built_graph;output_span"),
            ]
            notes = list(note_open_question(_repository(folder, {}, rows)))
            self.assertEqual(
                [(note.line, note.message) for note in notes],
                [
                    (4, "hand_built_graph: 1 rows wait for an answer"),
                    (3, "output_span: 2 rows wait for an answer"),
                ],
            )


class ExitCodeTest(unittest.TestCase):
    """Checks what the command line returns."""

    @mock.patch(K_FROZEN_NAME, K_TEST_FROZEN)
    def test_exits_clean_when_the_table_covers_the_tree(self) -> None:
        """Returns 0 for a tree whose only remarks are notes."""
        files = {
            "src/core/a.cpp": K_ONE_SITE + "auto x = Detail::bridge(a());\n",
            "src/optim/b.cpp": K_ONE_SITE,
        }
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, files, [_row("src/core/a.cpp", 3, "rank {} is too high")])
            code, printed = _run(folder)
            self.assertEqual(code, 0)
            self.assertIn("note: src/optim/b.cpp:3: note_frozen_growth", printed)
            self.assertIn("1 bridge calls", printed)

    def test_exits_with_findings_when_a_site_was_added(self) -> None:
        """Returns 1 and names the ratchet when a file outgrew its rows."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(
                folder,
                {"src/core/a.cpp": K_TWO_SITES},
                [_row("src/core/a.cpp", 3, "rank {} is too high")],
            )
            code, printed = _run(folder)
            self.assertEqual(code, 1)
            self.assertIn("src/core/a.cpp:7: rule_ratchet:", printed)

    def test_reports_counts_and_exits_clean(self) -> None:
        """Returns 0 under --report and prints one count per rule."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(
                folder,
                {"src/core/a.cpp": K_TWO_SITES},
                [_row("src/core/a.cpp", 3, "rank {} is too high")],
            )
            code, printed = _run(folder, "--report")
            self.assertEqual(code, 0)
            self.assertIn("rule_ratchet: 1", printed)


if __name__ == "__main__":
    unittest.main()

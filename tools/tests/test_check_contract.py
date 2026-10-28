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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.check_contract import (  # noqa: E402
    K_CHECKER,
    K_COLUMNS,
    K_HEAD_LENGTH,
    K_SITE_TABLE,
    Repository,
    find_removed_names,
    find_sites,
    main,
    note_open_question,
    note_unclassified,
    rule_removed_surface,
    rule_stale_row,
    rule_table_shape,
    rule_unlisted_site,
)

K_ONE_SITE = 'void f(int rank)\n{\n    throw Failure("rank {} is too high");\n}\n'
K_SECOND_SITE = 'void g(int rank)\n{\n    throw Failure("rank {} is negative");\n}\n'
K_TWO_SITES = K_ONE_SITE + K_SECOND_SITE
K_FILE = "modules/core/core/source/a.cpp"
K_OTHER_FILE = "modules/core/core/source/b.cpp"


def _row(
    file: str,
    line: int,
    head: str,
    kind: str = "throw",
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

    def test_finds_a_throw_site_with_its_line_and_head(self) -> None:
        """Reports the statement's line and its first literal as the head."""
        sites = find_sites(K_FILE, K_ONE_SITE)
        self.assertEqual(
            [(site.line, site.kind, site.head) for site in sites],
            [(3, "throw", "rank {} is too high")],
        )

    def test_joins_adjacent_literals_and_cuts_the_head(self) -> None:
        """Joins the literals of a wrapped message and keeps the first characters."""
        literals = '"' + "x" * 40 + '"\n    " and ' + "y" * 40 + '"'
        text = "throw Failure(\n    " + literals + ", a);\n"
        head = find_sites(K_FILE, text)[0].head
        self.assertEqual(head, ("x" * 40 + " and " + "y" * 40)[:K_HEAD_LENGTH])

    def test_unescapes_a_quote_in_the_head(self) -> None:
        """Counts an escaped quote as one character of the head."""
        text = 'throw Failure("the name \\"{}\\" is taken");\n'
        self.assertEqual(find_sites(K_FILE, text)[0].head, 'the name "{}" is taken')

    def test_finds_a_throw_statement(self) -> None:
        """Reports a raw throw with the first literal of its statement."""
        text = 'catch (...)\n{\n    throw Error(format("no arena of {} bytes", size));\n}\n'
        sites = find_sites(K_FILE, text)
        self.assertEqual(
            [(site.line, site.kind, site.head) for site in sites],
            [(3, "throw", "no arena of {} bytes")],
        )

    def test_skips_comments_and_literals(self) -> None:
        """Reports no site for a mention of throw in a comment or in a literal."""
        text = (
            '// unreachable: throw Failure("x") always runs\n'
            "/* throw nothing */\n"
            'const char* text = "they throw when x fires";\n'
        )
        self.assertEqual(find_sites("modules/core/core/include/SushiAI/core/a.hpp", text), [])

    def test_keeps_a_digit_separator_out_of_literals(self) -> None:
        """Finds the site that follows a number written with digit separators."""
        text = "const int limit = 1'000'000;\n" + K_ONE_SITE
        self.assertEqual(len(find_sites(K_FILE, text)), 1)


class RemovedSurfaceTest(unittest.TestCase):
    """Checks that SA_THROW_IF and the rest of the throwing surface stay removed."""

    def test_is_registered(self) -> None:
        """Runs every rule as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_removed_surface"], rule_removed_surface)
        self.assertIs(K_CHECKER.rules["rule_unlisted_site"], rule_unlisted_site)
        self.assertIs(K_CHECKER.rules["rule_stale_row"], rule_stale_row)
        self.assertIs(K_CHECKER.rules["rule_table_shape"], rule_table_shape)

    def test_finds_a_call_a_definition_and_an_include(self) -> None:
        """Reports each removed name with its line, comments and literals aside."""
        text = (
            "#include <SushiAI/failure/error.hpp>\n"
            "// SA_THROW_IF is gone\n"
            'const char* note = "SA_THROW_IF";\n'
            'void f() { SA_THROW_IF(true, "x"); }\n'
            "#define SA_THROW_IF(condition, ...) do {} while (0)\n"
            "auto r = Detail::bridge_throwing(make());\n"
            "throw SushiAI::Error(message);\n"
        )
        self.assertEqual(
            find_removed_names(text),
            [
                (1, "failure/error.hpp"),
                (4, "SA_THROW_IF"),
                (5, "SA_THROW_IF"),
                (6, "bridge_throwing"),
                (7, "SushiAI::Error"),
            ],
        )

    def test_leaves_the_error_of_another_library_alone(self) -> None:
        """Accepts SushiRuntime::Error and an enumerator named Error."""
        text = "const SushiRuntime::Error* refused = tasks.refusal();\nauto l = LogLevel::Error;\n"
        self.assertEqual(find_removed_names(text), [])

    def test_reports_a_new_site_under_the_library_the_programs_and_the_tests(self) -> None:
        """Reports SA_THROW_IF in a module, an application and a tests folder, not elsewhere."""
        text = 'void f() { SA_THROW_IF(true, "x"); }\n'
        files = {
            "modules/core/core/source/a.cpp": text,
            "applications/programs/source/train.cpp": text,
            "tests/unit/test_a.cpp": text,
            "modules/core/stray.cpp": text,
            "tools/other.cpp": text,
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_removed_surface(_repository(folder, files, [])))
            self.assertEqual(
                sorted(issue.path.name for issue in issues), ["a.cpp", "test_a.cpp", "train.cpp"]
            )

    def test_accepts_a_tree_without_the_removed_names(self) -> None:
        """Reports nothing for a file that fails through a status."""
        text = 'Status f() { SA_FAIL_IF(true, StatusCode::InvalidDescription, "x"); }\n'
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_FILE: text}, [])
            self.assertEqual(list(rule_removed_surface(repository)), [])

    def test_reads_the_applications_and_no_tests_folder_for_throw_sites(self) -> None:
        """Reports a throw without a row under applications/, and none under a tests/ folder."""
        files = {
            "applications/programs/source/train.cpp": K_ONE_SITE,
            "applications/examples/source/demo.cpp": K_ONE_SITE,
            "tests/unit/test_a.cpp": K_ONE_SITE,
            "modules/core/core/tests/test_b.cpp": K_ONE_SITE,
            "modules/core/stray.cpp": K_ONE_SITE,
        }
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files, [])
            self.assertEqual(
                [issue.path.name for issue in rule_unlisted_site(repository)],
                ["demo.cpp", "train.cpp"],
            )


class CoverageTest(unittest.TestCase):
    """Checks that every site has a row and every row a site."""

    def test_reports_a_site_without_a_row(self) -> None:
        """Reports the site whose head no row carries."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_FILE: K_ONE_SITE}, [])
            issues = list(rule_unlisted_site(repository))
            self.assertEqual(
                [(issue.line, issue.message) for issue in issues],
                [(3, "no row for 'rank {} is too high'")],
            )

    def test_reports_a_row_without_a_site(self) -> None:
        """Reports a stale row at its line of the table."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row(K_FILE, 3, "rank {} is too high"),
                _row("modules/core/core/source/gone.cpp", 9, "gone"),
            ]
            repository = _repository(folder, {K_FILE: K_ONE_SITE}, rows)
            issues = list(rule_stale_row(repository))
            self.assertEqual(
                [(issue.path.name, issue.line) for issue in issues], [(K_SITE_TABLE.name, 3)]
            )

    def test_matches_a_moved_site_by_its_head(self) -> None:
        """Accepts a row whose line hint is out of date."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 300, "rank {} is too high")]
            repository = _repository(folder, {K_FILE: K_ONE_SITE}, rows)
            self.assertEqual(
                list(rule_unlisted_site(repository)) + list(rule_stale_row(repository)), []
            )

    def test_reports_a_reworded_site_twice(self) -> None:
        """Reports a changed message as one unlisted site and one stale row."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 3, "rank {} is far too high")]
            repository = _repository(folder, {K_FILE: K_ONE_SITE}, rows)
            self.assertEqual(len(list(rule_unlisted_site(repository))), 1)
            self.assertEqual(len(list(rule_stale_row(repository))), 1)

    def test_pairs_equal_heads_one_to_one(self) -> None:
        """Reports the second of two equal sites when the table holds one row for them."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 3, "rank {} is too high")]
            repository = _repository(folder, {K_FILE: K_ONE_SITE + K_ONE_SITE}, rows)
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
            rows = [_row(K_FILE, 3, "a"), _row(K_FILE, 9, "b"), _row(K_OTHER_FILE, 1, "c")]
            self.assertEqual(list(rule_table_shape(_repository(folder, {}, rows))), [])

    def test_reports_a_row_out_of_order(self) -> None:
        """Reports the row that sorts before its predecessor."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 9, "b"), _row(K_FILE, 3, "a")]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [3]
            )

    def test_reports_a_malformed_row_and_an_unknown_class(self) -> None:
        """Reports a row with too few cells and a row with a mechanism the table does not know."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [K_FILE + "\t3\tf", _row(K_OTHER_FILE, 1, "c", mechanism="maybe")]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [2, 3]
            )


    def test_accepts_a_row_that_waits_for_a_class(self) -> None:
        """Reports nothing for a row whose cause and mechanism are unclassified."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 3, "a", mechanism="unclassified")]
            self.assertEqual(list(rule_table_shape(_repository(folder, {}, rows))), [])

    def test_reports_an_open_question_that_is_no_key(self) -> None:
        """Accepts keys joined by a semicolon and reports a cell that holds a sentence."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row(K_FILE, 3, "a", question="hand_built_graph;layout_proof"),
                _row(K_FILE, 9, "b", question="Who builds it?"),
            ]
            self.assertEqual(
                [issue.line for issue in rule_table_shape(_repository(folder, {}, rows))], [3]
            )


class NoteTest(unittest.TestCase):
    """Checks the notes, which are printed and never fail the run."""

    def test_lists_a_row_that_waits_for_a_class(self) -> None:
        """Yields a note for a row whose mechanism is unclassified."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(K_FILE, 3, "rank {} is too high", mechanism="unclassified")]
            repository = _repository(folder, {K_FILE: K_ONE_SITE}, rows)
            self.assertEqual(len(list(note_unclassified(repository))), 1)


    def test_counts_the_rows_of_each_open_question(self) -> None:
        """Yields one note per key, at the first row that carries it, with its row count."""
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                _row(K_FILE, 3, "a"),
                _row(K_FILE, 5, "b", question="output_span"),
                _row(K_FILE, 9, "c", question="hand_built_graph;output_span"),
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

    def test_exits_clean_when_the_table_covers_the_tree(self) -> None:
        """Returns 0 for a tree whose throw sites all have rows."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {K_FILE: K_ONE_SITE}, [_row(K_FILE, 3, "rank {} is too high")])
            code, printed = _run(folder)
            self.assertEqual(code, 0)
            self.assertEqual(printed, "")

    def test_exits_with_findings_when_a_throw_was_added(self) -> None:
        """Returns 1 and names the rule when a file holds a throw no row describes."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(
                folder,
                {K_FILE: K_TWO_SITES},
                [_row(K_FILE, 3, "rank {} is too high")],
            )
            code, printed = _run(folder)
            self.assertEqual(code, 1)
            self.assertIn("modules/core/core/source/a.cpp:7: rule_unlisted_site:", printed)

    def test_exits_with_findings_when_the_removed_macro_returns(self) -> None:
        """Returns 1 and names the rule for a new SA_THROW_IF."""
        text = 'void f() { SA_THROW_IF(true, "x"); }\n'
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {K_FILE: text}, [])
            code, printed = _run(folder)
            self.assertEqual(code, 1)
            self.assertIn("modules/core/core/source/a.cpp:1: rule_removed_surface:", printed)

    def test_reports_counts_and_exits_clean(self) -> None:
        """Returns 0 under --report and prints one count per rule."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(
                folder,
                {K_FILE: K_TWO_SITES},
                [_row(K_FILE, 3, "rank {} is too high")],
            )
            code, printed = _run(folder, "--report")
            self.assertEqual(code, 0)
            self.assertIn("rule_unlisted_site: 1", printed)


if __name__ == "__main__":
    unittest.main()

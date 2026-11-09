# test_check_contract.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the site scanner, the rules and the notes of the error contract checker on trees held in memory."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Finding  # noqa: E402
from contract.check_contract import (  # noqa: E402
    K_CHECKER,
    K_COLUMNS,
    K_HEAD_LENGTH,
    K_SITE_TABLE,
    DiskTree,
    MemoryTree,
    Repository,
    find_removed_names,
    find_sites,
    main,
    open_question_notes,
    render_notes,
    rows_of,
    rule_removed_surface,
    rule_stale_row,
    rule_table_shape,
    rule_unlisted_site,
    table_problems,
    unclassified_notes,
    unmatched_of,
)

K_ONE_SITE = 'void f(int rank)\n{\n    throw Failure("rank {} is too high");\n}\n'
K_SECOND_SITE = 'void g(int rank)\n{\n    throw Failure("rank {} is negative");\n}\n'
K_TWO_SITES = K_ONE_SITE + K_SECOND_SITE
K_FILE = "modules/core/core/source/a.cpp"
K_OTHER_FILE = "modules/core/core/source/b.cpp"
K_ROOT = Path("repo")
K_TABLE_KEY = K_SITE_TABLE.as_posix()


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


def _table(rows: list[str]) -> list[str]:
    """Returns the lines of a class table: its header, then the rows."""
    return ["\t".join(K_COLUMNS), *rows]


def _repository(files: dict[str, str], rows: list[str] | None) -> Repository:
    """Returns a repository over a tree in memory holding the sources and, unless rows is None, a table."""
    held = dict(files)
    if rows is not None:
        held[K_TABLE_KEY] = "\n".join(_table(rows)) + "\n"
    return Repository(K_ROOT, MemoryTree(held))


def _findings(repository: Repository) -> list[Finding]:
    """Returns the findings of every rule over a repository, as the runner names them."""
    return [
        Finding(issue.path, issue.line, name, issue.message)
        for name, rule in K_CHECKER.rules.items()
        for issue in rule(repository)
    ]


def _shown(repository: Repository) -> list[str]:
    """Returns the lines the command line prints for the findings of a repository."""
    return [finding.format(K_ROOT) for finding in _findings(repository)]


def _write(root: Path, files: dict[str, str]) -> None:
    """Writes the files under a folder on disk."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class SiteScannerTest(unittest.TestCase):
    """Checks what counts as a failure site and how its message head is cut."""

    K_TEST_SIZE = "small"

    def test_finds_a_throw_site_with_its_line_and_head(self) -> None:
        """Reports the statement's line and its first literal as the head, wherever the throw sits."""
        rows = (
            (K_ONE_SITE, [(3, "throw", "rank {} is too high")]),
            (
                'catch (...)\n{\n    throw Error(format("no arena of {} bytes", size));\n}\n',
                [(3, "throw", "no arena of {} bytes")],
            ),
        )
        for text, expected in rows:
            with self.subTest(text=text):
                sites = find_sites(K_FILE, text)
                self.assertEqual([(site.line, site.kind, site.head) for site in sites], expected)

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

    K_TEST_SIZE = "small"

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
        issues = list(rule_removed_surface(_repository(files, [])))
        self.assertEqual(
            sorted(issue.path.name for issue in issues), ["a.cpp", "test_a.cpp", "train.cpp"]
        )

    def test_accepts_a_tree_without_the_removed_names(self) -> None:
        """Reports nothing for a file that fails through a status."""
        text = 'Status f() { SA_FAIL_IF(true, StatusCode::InvalidDescription, "x"); }\n'
        repository = _repository({K_FILE: text}, [])
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
        repository = _repository(files, [])
        self.assertEqual(
            [issue.path.name for issue in rule_unlisted_site(repository)],
            ["demo.cpp", "train.cpp"],
        )


class CoverageTest(unittest.TestCase):
    """Checks that every site has a row and every row a site."""

    K_TEST_SIZE = "small"

    def test_reports_a_site_without_a_row(self) -> None:
        """Reports the site whose head no row carries."""
        issues = list(rule_unlisted_site(_repository({K_FILE: K_ONE_SITE}, [])))
        self.assertEqual(
            [(issue.line, issue.message) for issue in issues],
            [(3, "no row for 'rank {} is too high'")],
        )

    def test_reports_a_row_without_a_site(self) -> None:
        """Reports a stale row at its line of the table."""
        rows = [
            _row(K_FILE, 3, "rank {} is too high"),
            _row("modules/core/core/source/gone.cpp", 9, "gone"),
        ]
        issues = list(rule_stale_row(_repository({K_FILE: K_ONE_SITE}, rows)))
        self.assertEqual(
            [(issue.path.name, issue.line) for issue in issues], [(K_SITE_TABLE.name, 3)]
        )

    def test_matches_a_moved_site_by_its_head(self) -> None:
        """Accepts a row whose line hint is out of date."""
        repository = _repository({K_FILE: K_ONE_SITE}, [_row(K_FILE, 300, "rank {} is too high")])
        self.assertEqual(
            list(rule_unlisted_site(repository)) + list(rule_stale_row(repository)), []
        )

    def test_reports_a_reworded_site_as_unlisted(self) -> None:
        """Reports a changed message as a site no row carries."""
        repository = _repository({K_FILE: K_ONE_SITE}, [_row(K_FILE, 3, "rank {} is far too high")])
        self.assertEqual(len(list(rule_unlisted_site(repository))), 1)

    def test_reports_a_reworded_site_as_a_stale_row(self) -> None:
        """Reports the row of the old message as a row no site carries."""
        repository = _repository({K_FILE: K_ONE_SITE}, [_row(K_FILE, 3, "rank {} is far too high")])
        self.assertEqual(len(list(rule_stale_row(repository))), 1)

    def test_pairs_equal_heads_one_to_one(self) -> None:
        """Reports the second of two equal sites when the table holds one row for them."""
        repository = _repository(
            {K_FILE: K_ONE_SITE + K_ONE_SITE}, [_row(K_FILE, 3, "rank {} is too high")]
        )
        self.assertEqual([issue.line for issue in rule_unlisted_site(repository)], [7])

    def test_unmatched_of_pairs_the_sites_and_rows_it_is_given(self) -> None:
        """Returns the lone site and the lone row of values given without a tree."""
        sites = {K_FILE: find_sites(K_FILE, K_TWO_SITES)}
        rows = rows_of(_table([_row(K_FILE, 3, "rank {} is too high")]))
        lone_sites, lone_rows = unmatched_of(sites, rows)
        self.assertEqual(
            ([site.head for site in lone_sites], lone_rows), (["rank {} is negative"], [])
        )


class TableShapeTest(unittest.TestCase):
    """Checks the header, the cells and the order of the class table."""

    K_TEST_SIZE = "small"

    def test_reports_a_missing_table(self) -> None:
        """Reports a repository that carries no table."""
        self.assertEqual([line for line, _ in table_problems([])], [1])

    def test_accepts_a_sorted_table(self) -> None:
        """Reports nothing for well-formed rows in file and line order."""
        rows = [_row(K_FILE, 3, "a"), _row(K_FILE, 9, "b"), _row(K_OTHER_FILE, 1, "c")]
        self.assertEqual(table_problems(_table(rows)), [])

    def test_reports_a_row_out_of_order(self) -> None:
        """Reports the row that sorts before its predecessor."""
        rows = [_row(K_FILE, 9, "b"), _row(K_FILE, 3, "a")]
        self.assertEqual([line for line, _ in table_problems(_table(rows))], [3])

    def test_reports_a_row_with_too_few_cells(self) -> None:
        """Reports a row that holds fewer cells than the header."""
        rows = [K_FILE + "\t3\tf"]
        self.assertEqual([line for line, _ in table_problems(_table(rows))], [2])

    def test_reports_a_mechanism_the_table_does_not_know(self) -> None:
        """Reports a row whose mechanism is none of the known ones."""
        rows = [_row(K_OTHER_FILE, 1, "c", mechanism="maybe")]
        self.assertEqual([line for line, _ in table_problems(_table(rows))], [2])

    def test_accepts_a_row_that_waits_for_a_class(self) -> None:
        """Reports nothing for a row whose cause and mechanism are unclassified."""
        rows = [_row(K_FILE, 3, "a", mechanism="unclassified")]
        self.assertEqual(table_problems(_table(rows)), [])

    def test_accepts_open_question_keys_joined_by_a_semicolon(self) -> None:
        """Reports nothing for a cell that holds snake_case keys joined by semicolons."""
        rows = [_row(K_FILE, 3, "a", question="hand_built_graph;layout_proof")]
        self.assertEqual(table_problems(_table(rows)), [])

    def test_reports_an_open_question_that_is_a_sentence(self) -> None:
        """Reports a cell that holds a sentence in place of keys."""
        rows = [_row(K_FILE, 9, "b", question="Who builds it?")]
        self.assertEqual([line for line, _ in table_problems(_table(rows))], [2])

    def test_a_repository_reports_the_problems_at_its_table(self) -> None:
        """Gives each problem the path of the table and the line of the row."""
        rows = [_row(K_FILE, 9, "b"), _row(K_FILE, 3, "a")]
        issues = list(rule_table_shape(_repository({}, rows)))
        self.assertEqual(
            [(issue.path, issue.line) for issue in issues], [(K_ROOT / K_SITE_TABLE, 3)]
        )


class NoteTest(unittest.TestCase):
    """Checks the notes, which are printed and never fail the run."""

    K_TEST_SIZE = "small"

    def test_lists_a_row_that_waits_for_a_class(self) -> None:
        """Yields a note for a row whose mechanism is unclassified."""
        rows = rows_of(_table([_row(K_FILE, 3, "rank {} is too high", mechanism="unclassified")]))
        self.assertEqual(unclassified_notes(rows), [(2, f"{K_FILE}:3 has no class yet")])

    def test_counts_the_rows_of_each_open_question(self) -> None:
        """Yields one note per key, at the first row that carries it, with its row count."""
        rows = rows_of(
            _table(
                [
                    _row(K_FILE, 3, "a"),
                    _row(K_FILE, 5, "b", question="output_span"),
                    _row(K_FILE, 9, "c", question="hand_built_graph;output_span"),
                ]
            )
        )
        self.assertEqual(
            open_question_notes(rows),
            [
                (4, "hand_built_graph: 1 rows wait for an answer"),
                (3, "output_span: 2 rows wait for an answer"),
            ],
        )


class ExitCodeTest(unittest.TestCase):
    """Checks what the command line prints, from the findings and the notes it holds."""

    K_TEST_SIZE = "small"

    def test_exits_clean_when_the_table_covers_the_tree(self) -> None:
        """Gives no finding and no note for a tree whose throw sites all have rows."""
        repository = _repository({K_FILE: K_ONE_SITE}, [_row(K_FILE, 3, "rank {} is too high")])
        self.assertEqual((_shown(repository), render_notes(repository)), ([], []))

    def test_exits_with_findings_when_a_throw_was_added(self) -> None:
        """Names the rule and the line of a throw no row describes."""
        repository = _repository({K_FILE: K_TWO_SITES}, [_row(K_FILE, 3, "rank {} is too high")])
        self.assertEqual(
            _shown(repository),
            [f"{K_FILE}:7: rule_unlisted_site: no row for 'rank {{}} is negative'"],
        )

    def test_exits_with_findings_when_the_removed_macro_returns(self) -> None:
        """Names the rule for a new SA_THROW_IF."""
        repository = _repository({K_FILE: 'void f() { SA_THROW_IF(true, "x"); }\n'}, [])
        self.assertTrue(_shown(repository)[0].startswith(f"{K_FILE}:1: rule_removed_surface:"))


class DiskTreeTest(unittest.TestCase):
    """Checks that the tree on disk reads what the tree in memory holds."""

    K_TEST_SIZE = "medium"

    def test_reads_the_same_sources_and_table_as_the_tree_in_memory(self) -> None:
        """Returns, for the folders given, the files of the memory tree with the same names and texts."""
        files = {
            K_FILE: K_ONE_SITE,
            "applications/programs/source/train.cpp": K_ONE_SITE,
            "modules/core/stray.cpp": K_ONE_SITE,
            "tools/other.cpp": K_ONE_SITE,
            K_TABLE_KEY: "\n".join(_table([_row(K_FILE, 3, "rank {} is too high")])) + "\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            _write(Path(folder), files)
            disk, memory = DiskTree(Path(folder)), MemoryTree(files)
            folders = ("modules/*/*/include", "modules/*/*/source", "applications", "tests")
            self.assertEqual(disk.texts(folders), memory.texts(folders))
            self.assertEqual(disk.table_text(), memory.table_text())

    def test_has_no_table_when_the_file_is_absent(self) -> None:
        """Returns None for a folder without a class table."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(DiskTree(Path(folder)).table_text())


class CommandLineTest(unittest.TestCase):
    """Checks the command line over a folder on disk."""

    K_TEST_SIZE = "medium"

    def test_reports_counts_and_exits_clean(self) -> None:
        """Returns 0 under --report and prints one count per rule."""
        files = {
            K_FILE: K_TWO_SITES,
            K_TABLE_KEY: "\n".join(_table([_row(K_FILE, 3, "rank {} is too high")])) + "\n",
        }
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            _write(Path(folder), files)
            with contextlib.redirect_stdout(printed):
                code = main([folder, "--report"])
        self.assertEqual(code, 0)
        self.assertIn("rule_unlisted_site: 1", printed.getvalue())


if __name__ == "__main__":
    unittest.main()

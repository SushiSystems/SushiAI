# test_check_source_comments.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the source comment rules on values in memory, and the command line on a tiny tree."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from documentation import check_source_comments  # noqa: E402
from documentation.check_source_comments import (  # noqa: E402
    K_CHECKER,
    K_PART_OF,
    CheckedSource,
    rule_block_ceiling,
    rule_comment_runs,
    rule_file_header,
    rule_history,
    rule_license_block,
    rule_no_date,
    rule_separators,
    split_source,
)
from documentation.licence import Licence  # noqa: E402

K_ACME = Licence(
    holder="Jane Doe & Acme Corp",
    company="Acme Corp",
    site="https://acme.example",
    open_lines=("Licensed under the Acme Licence 1.0. See LICENSE.", "Commercial use requires a licence from Acme Corp."),
    closed_lines=("All rights reserved. No licence is granted.",),
    open_box_lines=("Licensed under the Acme Licence 1.0.", "", "    https://acme.example/licence", "",
                    "Commercial use requires a licence from Acme Corp; see COMMERCIAL.md."),
    closed_box_lines=("All rights reserved.", "", "No licence is granted by Acme Corp."),
    closed=False,
)
K_PROJECT_NAME = "AcmeSkills"
K_PROJECT_URL = "https://github.com/AcmeCorp/AcmeSkills"
K_PROJECT_LINE = f"{K_PROJECT_NAME} - {K_PROJECT_URL}"
K_COPYRIGHT = "Copyright (c) 2026-present Jane Doe & Acme Corp"
K_OWN_LINES = (K_COPYRIGHT, *K_ACME.hash_lines)
K_OWN_BOX_LINES = (K_COPYRIGHT, "", *K_ACME.box_lines)
K_LACKS_COPYRIGHT = f"license block lacks '{K_ACME.copyright_shape}'"
K_FOREIGN_LICENCE = "Licensed under the Apache License, Version 2.0. See LICENSE."
K_OLD_HOLDER = "Copyright (c) 2026 Acme Corp"
K_BOM = "﻿"


def _source(name: str, text: str) -> CheckedSource:
    """Returns a file of the given name and text, split and held against the made-up licence."""
    return split_source(SourceFile(Path(name), text), K_ACME)


def _messages(name: str, text: str) -> list[str]:
    """Returns the license block messages for a file with the given name and content."""
    return [issue.message for issue in rule_license_block(_source(name, text))]


def _lacks(lines: tuple[str, ...]) -> list[str]:
    """Returns the messages the rule gives when every row of the given lines is missing."""
    return [f"license block lacks '{line.strip()}'" for line in lines if line.strip()]


def _boxed(name: str, rows: tuple[str, ...] = K_OWN_BOX_LINES) -> str:
    """Returns a C-family file opening with the three-section box and a complete file block."""
    edge = "/" + "*" * 74 + "/"
    project = [f"/*{row.center(72)}*/" for row in (K_PART_OF, K_PROJECT_NAME, K_PROJECT_URL)]
    licence = [f"/* {row.ljust(70)} */" for row in rows]
    box = [edge, f"/* {name.ljust(70)} */", edge, *project, edge, *licence, edge]
    file_block = f"/**\n * @file {name}\n * @brief Declares the file.\n * @author Jane Doe\n */\n"
    return "\n".join(box) + "\n\n" + file_block


def _hashed(name: str, rows: tuple[str, ...] = K_OWN_LINES, first: str = "") -> str:
    """Returns a Python file opening with the `#` form of the block, after an optional line."""
    body = "\n".join(f"# {row}" for row in (name, K_PROJECT_LINE, *rows))
    return first + body + '\n"""Does nothing."""\n'


def _foreign(text: str) -> str:
    """Returns the message the rule gives for a line the licence does not allow."""
    return f"license block carries a line this repository's licence does not allow: '{text}'"


class RuleTableTest(unittest.TestCase):
    """Checks that the license block rule runs as part of the checker."""
    K_TEST_SIZE = "small"

    def test_registers_the_license_block_rule(self) -> None:
        """Holds the license block rule under its own name in the rule table."""
        self.assertIs(K_CHECKER.rules["rule_license_block"], rule_license_block)


class LicenseBlockTest(unittest.TestCase):
    """Checks which file openings the license block rule accepts."""
    K_TEST_SIZE = "small"

    def test_accepts_boxed_block(self) -> None:
        """Accepts a C++ header that opens with the full boxed block."""
        self.assertEqual(_messages("graph.hpp", _boxed("graph.hpp")), [])

    def test_accepts_python_block(self) -> None:
        """Accepts a Python file that opens with the `#` block."""
        self.assertEqual(_messages("graph.py", _hashed("graph.py")), [])

    def test_accepts_block_after_coding_line(self) -> None:
        """Accepts a Python file whose block follows a shebang line or a source encoding line."""
        for first in ("#!/usr/bin/env python3\n", "# -*- coding: utf-8 -*-\n"):
            with self.subTest(first=first):
                self.assertEqual(_messages("graph.py", _hashed("graph.py", first=first)), [])

    def test_accepts_block_behind_a_byte_order_mark(self) -> None:
        """Accepts a correct block whose file begins with a UTF-8 byte order mark."""
        self.assertEqual(_messages("graph.hpp", K_BOM + _boxed("graph.hpp")), [])

    def test_accepts_upstream_lines(self) -> None:
        """Accepts extra upstream notice lines below the repository's own lines."""
        rows = (*K_OWN_BOX_LINES, "", "Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        self.assertEqual(_messages("lapjv.cpp", _boxed("lapjv.cpp", rows)), [])

    def test_demands_own_lines_beside_upstream_lines(self) -> None:
        """Reports a ported file that carries only the upstream notice."""
        rows = ("Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        expected = [K_LACKS_COPYRIGHT, *_lacks(K_ACME.box_lines)]
        self.assertEqual(_messages("lapjv.cpp", _boxed("lapjv.cpp", rows)), expected)

    def test_reports_a_box_without_the_project_section(self) -> None:
        """Reports a boxed block that lacks the centred project section."""
        text = _boxed("graph.hpp").replace(f"/*{K_PART_OF.center(72)}*/\n", "")
        self.assertEqual(_messages("graph.hpp", text), [f"license block lacks '{K_PART_OF}'"])

    def test_reports_missing_block(self) -> None:
        """Reports a file that opens with code."""
        self.assertEqual(
            _messages("graph.hpp", "#pragma once\n"),
            ["file must open with the license block"],
        )

    def test_reports_wrong_file_name(self) -> None:
        """Reports a block whose first line names another file."""
        self.assertEqual(
            _messages("graph.hpp", _boxed("node.hpp")),
            ["license block must open with graph.hpp"],
        )

    def test_reports_old_block(self) -> None:
        """Reports a block that carries the earlier holder line and the earlier licence."""
        expected = [K_LACKS_COPYRIGHT, *_lacks(K_ACME.hash_lines)]
        expected += [_foreign(K_OLD_HOLDER), _foreign(K_FOREIGN_LICENCE)]
        text = _hashed("graph.py", (K_OLD_HOLDER, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.py", text), expected)

    def test_reports_second_licence_beside_the_right_one(self) -> None:
        """Reports a licence line left in a block that also carries the right lines."""
        text = _boxed("graph.hpp", (*K_OWN_BOX_LINES, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(K_FOREIGN_LICENCE)])

    def test_reports_reserved_rights_beside_a_grant(self) -> None:
        """Reports a rights statement the repository's licence lines do not hold."""
        other = "All rights reserved. Nothing is granted to anyone."
        text = _boxed("graph.hpp", (*K_OWN_BOX_LINES, other))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(other)])


class PythonCommentTest(unittest.TestCase):
    """Checks which Python lines the comment rules read as comments."""
    K_TEST_SIZE = "small"

    def _runs(self, body: str) -> list[int]:
        """Returns the line numbers the comment-run rule reports for a Python file body."""
        source = _source("sample.py", _hashed("sample.py").rstrip("\n") + "\n" + body)
        return [issue.line for issue in rule_comment_runs(source)]

    def _head_lines(self) -> int:
        """Returns how many lines the license block and the docstring take."""
        return len(_hashed("sample.py").splitlines())

    def test_hash_lines_inside_a_string_are_not_comments(self) -> None:
        """Reports nothing for lines that start with # inside a string literal."""
        body = 'K_EXAMPLE = """\n#include <stdio.h>\n#include "api.h"\nint main(void);\n"""\n'
        self.assertEqual(self._runs(body), [])

    def test_a_real_run_of_comments_is_still_reported(self) -> None:
        """Reports two comment lines in a row, on the line of the first."""
        body = "x = 1\n# first reason\n# second reason\ny = 2\n"
        self.assertEqual(self._runs(body), [self._head_lines() + 2])

    def test_a_file_that_does_not_tokenize_falls_back_to_the_line_test(self) -> None:
        """Still reports a comment run in a file with an unclosed bracket."""
        body = "# first reason\n# second reason\nx = (1,\n"
        self.assertEqual(self._runs(body), [self._head_lines() + 1])


class DateTagTest(unittest.TestCase):
    """Checks that both spellings of the date tag are reported, in shader files too."""
    K_TEST_SIZE = "small"

    def test_reports_both_spellings_in_a_ray_tracing_shader(self) -> None:
        """Reports the at and the backslash spelling in a `.rgen` file."""
        body = "/** @date 2026 */\nint a;\n/** \\date 2026 */\nint b;\n"
        found = list(rule_no_date(_source("trace.rgen", _boxed("trace.rgen") + body)))
        self.assertEqual(len(found), 2)


def _found(rule, name: str, text: str) -> list[str]:
    """Returns the messages one rule gives for a file with the given name and content."""
    return [issue.message for issue in rule(_source(name, text))]


class FileHeaderTest(unittest.TestCase):
    """Covers the file block and the module docstring a file opens with."""

    K_TEST_SIZE = "small"

    def test_a_complete_file_block_is_clean(self) -> None:
        self.assertEqual(_found(rule_file_header, "a.hpp", _boxed("a.hpp")), [])

    def test_a_file_block_naming_another_file_is_reported(self) -> None:
        self.assertEqual(_found(rule_file_header, "a.hpp", _boxed("a.hpp").replace("@file a.hpp", "@file b.hpp")),
                         ["@file must name a.hpp"])

    def test_a_python_file_without_a_docstring_is_reported(self) -> None:
        text = _hashed("a.py").replace('"""Does nothing."""', "K_VALUE = 1")
        self.assertEqual(_found(rule_file_header, "a.py", text), ["module must open with a docstring"])


class BlockCeilingTest(unittest.TestCase):
    """Covers the length of a symbol comment block."""

    K_TEST_SIZE = "small"

    def test_a_symbol_block_over_the_ceiling_is_reported(self) -> None:
        block = "/**\n" + " * @brief Does it.\n" * 8 + " */\nint f();\n"
        self.assertEqual(_found(rule_block_ceiling, "a.hpp", _boxed("a.hpp") + block),
                         ["comment block exceeds 8 lines"])

    def test_a_short_symbol_block_is_clean(self) -> None:
        block = "/** @brief Does it. */\nint f();\n"
        self.assertEqual(_found(rule_block_ceiling, "a.hpp", _boxed("a.hpp") + block), [])


class SeparatorTest(unittest.TestCase):
    """Covers separator lines in comments."""

    K_TEST_SIZE = "small"

    def test_a_separator_comment_is_reported(self) -> None:
        self.assertEqual(_found(rule_separators, "a.hpp", _boxed("a.hpp") + "// ----------\nint f();\n"),
                         ["separator line outside the license block"])

    def test_the_license_box_is_not_a_separator(self) -> None:
        self.assertEqual(_found(rule_separators, "a.hpp", _boxed("a.hpp")), [])


class HistoryTest(unittest.TestCase):
    """Covers backlog tags and history words in comments."""

    K_TEST_SIZE = "small"

    def test_a_todo_tag_is_reported(self) -> None:
        self.assertEqual(_found(rule_history, "a.hpp", _boxed("a.hpp") + "// TODO speed up\nint f();\n"),
                         ["'TODO' belongs in the backlog or git"])

    def test_a_comment_that_says_what_a_thing_does_is_clean(self) -> None:
        self.assertEqual(_found(rule_history, "a.hpp", _boxed("a.hpp") + "// Adds two numbers.\nint f();\n"), [])


class CommandLineTest(unittest.TestCase):
    """Runs the checker on a tiny tree in a temporary folder, reading its files from the disk."""
    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        """Makes the made-up company's licence the one the checker enforces for these runs."""
        patch = mock.patch.object(check_source_comments, "K_LICENCE", K_ACME)
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, folder: str, *arguments: str) -> tuple[int, list[str]]:
        """Returns the exit code and the lines the checker prints for a folder."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, [folder, *arguments])
        return code, output.getvalue().splitlines()

    def test_accepts_a_byte_order_mark_on_disk(self) -> None:
        """Reads a correct file saved with a byte order mark and reports nothing."""
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "graph.hpp").write_text(K_BOM + _boxed("graph.hpp"), encoding="utf-8")
            self.assertEqual(self._run(folder), (0, []))

    def test_reports_both_spellings_in_a_ray_tracing_shader_on_disk(self) -> None:
        """Reads a `.rgen` file from the disk and reports the at and the backslash spelling."""
        body = "/** @date 2026 */\nint a;\n/** \\date 2026 */\nint b;\n"
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "trace.rgen").write_text(_boxed("trace.rgen") + body, encoding="utf-8")
            code, lines = self._run(folder, "--rule", "rule_no_date")
        self.assertEqual(
            (code, [line.split(": ", 1)[1] for line in lines]),
            (1, ["rule_no_date: @date is forbidden; git records dates"] * 2),
        )


if __name__ == "__main__":
    unittest.main()

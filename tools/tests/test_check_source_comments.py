# test_check_source_comments.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the license block rule of the source comment checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.check_source_comments import (  # noqa: E402
    K_BOX_LICENSE_LINES,
    K_CHECKER,
    K_COPYRIGHT_SHAPE,
    K_LICENSE_LINES,
    K_PART_OF,
    collect,
    rule_comment_runs,
    rule_no_date,
    rule_license_block,
)

K_PROJECT_NAME = "SushiSkills"
K_PROJECT_URL = "https://github.com/SushiSystems/SushiSkills"
K_PROJECT_LINE = f"{K_PROJECT_NAME} - {K_PROJECT_URL}"
K_COPYRIGHT = "Copyright (c) 2026-present Mustafa Garip & Sushi Systems"
K_SUSHI_LINES = (K_COPYRIGHT, *K_LICENSE_LINES)
K_SUSHI_BOX_LINES = (K_COPYRIGHT, "", *K_BOX_LICENSE_LINES)
K_LACKS_COPYRIGHT = f"license block lacks '{K_COPYRIGHT_SHAPE}'"
K_FOREIGN_LICENCE = "Licensed under the Apache License, Version 2.0. See LICENSE."
K_OLD_HOLDER = "Copyright (c) 2026 Sushi Systems"


def _messages(name: str, text: str, encoding: str = "utf-8") -> list[str]:
    """Returns the license block messages for a file with the given name and content."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_text(text, encoding=encoding)
        return [issue.message for source in collect([path]) for issue in rule_license_block(source)]


def _lacks(lines: tuple[str, ...]) -> list[str]:
    """Returns the messages the rule gives when every row of the given lines is missing."""
    return [f"license block lacks '{line.strip()}'" for line in lines if line.strip()]


def _boxed(name: str, rows: tuple[str, ...] = K_SUSHI_BOX_LINES) -> str:
    """Returns a C-family file opening with the three-section box that names the given file."""
    edge = "/" + "*" * 74 + "/"
    project = [f"/*{row.center(72)}*/" for row in (K_PART_OF, K_PROJECT_NAME, K_PROJECT_URL)]
    licence = [f"/* {row.ljust(70)} */" for row in rows]
    box = [edge, f"/* {name.ljust(70)} */", edge, *project, edge, *licence, edge]
    return "\n".join(box) + f"\n\n/**\n * @file {name}\n */\n"


def _hashed(name: str, rows: tuple[str, ...] = K_SUSHI_LINES, first: str = "") -> str:
    """Returns a Python file opening with the `#` form of the block, after an optional line."""
    body = "\n".join(f"# {row}" for row in (name, K_PROJECT_LINE, *rows))
    return first + body + '\n"""Does nothing."""\n'


def _foreign(text: str) -> str:
    """Returns the message the rule gives for a line the licence does not allow."""
    return f"license block carries a line this repository's licence does not allow: '{text}'"


class LicenseBlockTest(unittest.TestCase):
    """Checks which file openings the license block rule accepts."""

    def test_is_registered(self) -> None:
        """Runs as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_license_block"], rule_license_block)

    def test_accepts_boxed_block(self) -> None:
        """Accepts a C++ header that opens with the full boxed block."""
        self.assertEqual(_messages("graph.hpp", _boxed("graph.hpp")), [])

    def test_accepts_python_block(self) -> None:
        """Accepts a Python file that opens with the `#` block."""
        self.assertEqual(_messages("graph.py", _hashed("graph.py")), [])

    def test_accepts_block_after_shebang(self) -> None:
        """Accepts a Python file whose block follows a shebang line."""
        text = _hashed("graph.py", first="#!/usr/bin/env python3\n")
        self.assertEqual(_messages("graph.py", text), [])

    def test_accepts_block_after_coding_line(self) -> None:
        """Accepts a Python file whose block follows a source encoding line."""
        text = _hashed("graph.py", first="# -*- coding: utf-8 -*-\n")
        self.assertEqual(_messages("graph.py", text), [])

    def test_accepts_block_behind_a_byte_order_mark(self) -> None:
        """Accepts a correct block in a file saved with a UTF-8 byte order mark."""
        self.assertEqual(_messages("graph.hpp", _boxed("graph.hpp"), encoding="utf-8-sig"), [])

    def test_accepts_upstream_lines(self) -> None:
        """Accepts extra upstream notice lines below the Sushi lines."""
        rows = (*K_SUSHI_BOX_LINES, "", "Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        self.assertEqual(_messages("lapjv.cpp", _boxed("lapjv.cpp", rows)), [])

    def test_demands_sushi_lines_beside_upstream_lines(self) -> None:
        """Reports a ported file that carries only the upstream notice."""
        rows = ("Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        expected = [K_LACKS_COPYRIGHT, *_lacks(K_BOX_LICENSE_LINES)]
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
        expected = [K_LACKS_COPYRIGHT, *_lacks(K_LICENSE_LINES)]
        expected += [_foreign(K_OLD_HOLDER), _foreign(K_FOREIGN_LICENCE)]
        text = _hashed("graph.py", (K_OLD_HOLDER, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.py", text), expected)

    def test_reports_second_licence_beside_the_right_one(self) -> None:
        """Reports a licence line left in a block that also carries the right lines."""
        text = _boxed("graph.hpp", (*K_SUSHI_BOX_LINES, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(K_FOREIGN_LICENCE)])

    def test_reports_reserved_rights_beside_a_grant(self) -> None:
        """Reports a rights statement the repository's licence lines do not hold."""
        other = "All rights reserved. Nothing is granted to anyone."
        text = _boxed("graph.hpp", (*K_SUSHI_BOX_LINES, other))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(other)])


class PythonCommentTest(unittest.TestCase):
    """Checks which Python lines the comment rules read as comments."""

    def _runs(self, body: str) -> list[int]:
        """Returns the line numbers the comment-run rule reports for a Python file body."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.py"
            path.write_text(_hashed("sample.py").rstrip("\n") + "\n" + body, encoding="utf-8")
            return [issue.line for source in collect([path]) for issue in rule_comment_runs(source)]

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

    def test_reports_both_spellings_in_a_ray_tracing_shader(self) -> None:
        """Reports the at and the backslash spelling in a `.rgen` file."""
        body = "/** @date 2026 */\nint a;\n/** \\date 2026 */\nint b;\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.rgen"
            path.write_text(_boxed("trace.rgen") + body, encoding="utf-8")
            found = [issue for source in collect([path]) for issue in rule_no_date(source)]
        self.assertEqual(len(found), 2)


if __name__ == "__main__":
    unittest.main()

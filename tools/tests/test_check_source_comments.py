# test_check_source_comments.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
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
    K_CHECKER,
    K_LICENSE_LINES,
    collect,
    rule_license_block,
)

K_PROJECT_LINE = "SushiSkills - https://github.com/SushiSystems/SushiSkills"
K_SUSHI_LINES = ("Copyright (c) 2026 Sushi Systems", *K_LICENSE_LINES)
K_FOREIGN_LICENCE = "Licensed under the Apache License, Version 2.0. See LICENSE."
K_JOINT_HOLDER = "Copyright (c) 2026-present Mustafa Garip & Sushi Systems"


def _messages(name: str, text: str, encoding: str = "utf-8") -> list[str]:
    """Returns the license block messages for a file with the given name and content."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_text(text, encoding=encoding)
        return [issue.message for source in collect([path]) for issue in rule_license_block(source)]


def _boxed(name: str, rows: tuple[str, ...] = K_SUSHI_LINES) -> str:
    """Returns a C-family file opening with a boxed block that names the given file."""
    edge = "/" + "*" * 70 + "/"
    body = "\n".join(f"/* {row.ljust(66)} */" for row in (name, K_PROJECT_LINE, *rows))
    return f"{edge}\n{body}\n{edge}\n\n/**\n * @file {name}\n */\n"


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
        rows = (*K_SUSHI_LINES, "Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        self.assertEqual(_messages("lapjv.cpp", _boxed("lapjv.cpp", rows)), [])

    def test_demands_sushi_lines_beside_upstream_lines(self) -> None:
        """Reports a ported file that carries only the upstream notice."""
        rows = ("Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        expected = ["license block lacks 'Copyright (c) <year> Sushi Systems'"]
        expected += [f"license block lacks '{line}'" for line in K_LICENSE_LINES]
        self.assertEqual(_messages("lapjv.cpp", _boxed("lapjv.cpp", rows)), expected)

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
        """Reports a block that carries the joint holder and the earlier licence."""
        expected = ["license block lacks 'Copyright (c) <year> Sushi Systems'"]
        expected += [f"license block lacks '{line}'" for line in K_LICENSE_LINES]
        expected += [_foreign(K_JOINT_HOLDER), _foreign(K_FOREIGN_LICENCE)]
        text = _hashed("graph.py", (K_JOINT_HOLDER, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.py", text), expected)

    def test_reports_second_licence_beside_the_right_one(self) -> None:
        """Reports a licence line left in a block that also carries the right lines."""
        text = _boxed("graph.hpp", (*K_SUSHI_LINES, K_FOREIGN_LICENCE))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(K_FOREIGN_LICENCE)])

    def test_reports_reserved_rights_beside_a_grant(self) -> None:
        """Reports a rights statement the repository's licence lines do not hold."""
        other = "All rights reserved. Nothing is granted to anyone."
        text = _boxed("graph.hpp", (*K_SUSHI_LINES, other))
        self.assertEqual(_messages("graph.hpp", text), [_foreign(other)])


if __name__ == "__main__":
    unittest.main()

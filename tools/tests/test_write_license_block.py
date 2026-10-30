# test_write_license_block.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the license block writer on each way a source file can open."""

from __future__ import annotations

import contextlib
import io
import subprocess
from datetime import date
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation import check_source_comments  # noqa: E402
from documentation.check_source_comments import collect, rule_license_block  # noqa: E402
from documentation.licence import Licence  # noqa: E402
from licensing import write_license_block  # noqa: E402
from licensing.write_license_block import (  # noqa: E402
    Header,
    first_years,
    is_source,
    main,
    rewrite,
)

K_ACME = Licence(
    holder="Jane Doe & Acme Corp",
    company="Acme Corp",
    site="https://acme.example",
    open_lines=("Licensed under the Acme Licence 1.0. See LICENSE.", "Commercial use requires a licence from Acme Corp."),
    closed_lines=("All rights reserved. No licence is granted.",),
    open_box_lines=("Licensed under the Acme Licence 1.0.", "", "    https://acme.example/licence", "",
                    "Commercial use requires a licence from Acme Corp; see COMMERCIAL.md."),
    closed_box_lines=("All rights reserved.", "", "No licence is granted."),
    closed=False,
)
K_PROJECT = "AcmeRuntime - https://github.com/AcmeCorp/AcmeRuntime"
K_COPYRIGHT = "Copyright (c) 2026-present Jane Doe & Acme Corp"
K_HEADER = Header(K_PROJECT)
K_EDGE = "/" + "*" * 74 + "/"


def setUpModule() -> None:
    """Makes the made-up company's licence the one the checker and the writer use for every test here."""
    for module in (check_source_comments, write_license_block):
        patch = mock.patch.object(module, "K_LICENCE", K_ACME)
        patch.start()
        unittest.addModuleCleanup(patch.stop)


def _left(text: str, field: int = 70) -> str:
    """Returns a box row with the text at the left."""
    return f"/* {text.ljust(field)} */"


def _middle(text: str) -> str:
    """Returns a box row with the text centred, the odd space on the left."""
    padding = 72 - len(text)
    left = (padding + 1) // 2
    return "/*" + " " * left + text + " " * (padding - left) + "*/"


K_BOX = "\n".join(
    [
        K_EDGE,
        _left("graph.hpp"),
        K_EDGE,
        _middle("This file is part of:"),
        _middle("AcmeRuntime"),
        _middle("https://github.com/AcmeCorp/AcmeRuntime"),
        _middle("https://acme.example"),
        K_EDGE,
        _left(K_COPYRIGHT),
        _left(""),
        *[_left(row) for row in K_ACME.open_box_lines],
        K_EDGE,
    ]
)
K_HASHES = "\n".join(["# graph.py", f"# {K_PROJECT}", f"# {K_COPYRIGHT}", *[f"# {row}" for row in K_ACME.open_lines]])
K_OLD_BOX = "\n".join(
    [
        K_EDGE,
        _left("graph.hpp"),
        K_EDGE,
        _middle("This file is part of:"),
        _middle("AcmeRuntime"),
        K_EDGE,
        _left(K_COPYRIGHT),
        _left(""),
        _left('Licensed under the Apache License, Version 2.0 (the "License");'),
        K_EDGE,
    ]
)
K_DOC = "/**\n * @file graph.hpp\n * @brief Declares the graph.\n */\n#pragma once\n"
K_OLD_HASHES = f"# {K_COPYRIGHT}\n# Licensed under the Apache License, Version 2.0. See LICENSE.\n"
K_DOCSTRING = '"""Builds the graph."""\n\nimport sys\n'


def _main(arguments: list[str]) -> int:
    """Runs the writer from its command line, keeping its listing out of the test output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(arguments)


def _write(name: str, text: str, header: Header = K_HEADER, year: int = 2026) -> str:
    """Returns the text the writer produces for a file of the given name."""
    return rewrite(name, text, header, year)


class RewriteTest(unittest.TestCase):
    """Checks the text the writer produces for each file opening."""

    def test_replaces_an_old_box(self) -> None:
        """Replaces a boxed Apache block and keeps the file block one blank line below."""
        self.assertEqual(_write("graph.hpp", f"{K_OLD_BOX}\n\n{K_DOC}"), f"{K_BOX}\n\n{K_DOC}")

    def test_replaces_a_box_without_copyright(self) -> None:
        """Replaces a boxed block that names no holder."""
        box = "/********/\n/* graph.hpp */\n/********/\n"
        self.assertEqual(_write("graph.hpp", f"{box}{K_DOC}"), f"{K_BOX}\n\n{K_DOC}")

    def test_replaces_a_slash_block(self) -> None:
        """Replaces a leading `//` run that holds a copyright line."""
        old = "// Copyright (c) 2026 Somebody\n// Licensed under Apache.\n\n"
        self.assertEqual(_write("graph.hpp", f"{old}{K_DOC}"), f"{K_BOX}\n\n{K_DOC}")

    def test_inserts_above_a_file_block(self) -> None:
        """Inserts the block above a file that opens with its Doxygen block."""
        self.assertEqual(_write("graph.hpp", K_DOC), f"{K_BOX}\n\n{K_DOC}")

    def test_keeps_a_leading_comment_that_is_no_licence(self) -> None:
        """Keeps a leading comment and a directive below the block."""
        text = "/* eslint-disable */\n'use server'\n"
        self.assertEqual(_write("graph.hpp", text), f"{K_BOX}\n\n{text}")

    def test_replaces_old_hash_lines(self) -> None:
        """Replaces the two-line Python header and leaves the docstring directly below."""
        text = f"{K_OLD_HASHES}{K_DOCSTRING}"
        self.assertEqual(_write("graph.py", text), f"{K_HASHES}\n{K_DOCSTRING}")

    def test_keeps_shebang_and_encoding_lines_first(self) -> None:
        """Keeps a shebang and an encoding line above the block."""
        lead = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"
        text = f"{lead}{K_OLD_HASHES}{K_DOCSTRING}"
        self.assertEqual(_write("graph.py", text), f"{lead}{K_HASHES}\n{K_DOCSTRING}")

    def test_inserts_above_a_docstring(self) -> None:
        """Inserts the block directly above a module docstring."""
        self.assertEqual(_write("graph.py", K_DOCSTRING), f"{K_HASHES}\n{K_DOCSTRING}")

    def test_keeps_an_ordinary_leading_comment(self) -> None:
        """Keeps a CMake file's own leading comment one blank line below the block."""
        text = "# Composes the policy targets.\nadd_library(policy INTERFACE)\n"
        expected = K_HASHES.replace("graph.py", "CMakeLists.txt") + "\n\n" + text
        self.assertEqual(_write("CMakeLists.txt", text), expected)

    def test_writes_an_empty_file(self) -> None:
        """Writes the block alone into an empty file."""
        self.assertEqual(_write("graph.py", ""), f"{K_HASHES}\n")

    def test_keeps_newline_style_and_byte_order_mark(self) -> None:
        """Keeps CRLF newlines and a leading byte order mark."""
        text = "﻿" + f"{K_OLD_BOX}\n\n{K_DOC}".replace("\n", "\r\n")
        expected = "﻿" + f"{K_BOX}\n\n{K_DOC}".replace("\n", "\r\n")
        self.assertEqual(_write("graph.hpp", text), expected)

    def test_is_idempotent(self) -> None:
        """Changes nothing on a second run, for every opening."""
        cases = {
            "graph.hpp": [f"{K_OLD_BOX}\n\n{K_DOC}", K_DOC, "/* eslint-disable */\n'use server'\n"],
            "graph.py": [f"{K_OLD_HASHES}{K_DOCSTRING}", K_DOCSTRING, "", "#!/bin/env python\nx = 1\n"],
            "CMakeLists.txt": ["# Composes the policy targets.\nadd_library(policy INTERFACE)\n"],
        }
        for name, texts in cases.items():
            for text in texts:
                once = _write(name, text)
                self.assertEqual(_write(name, once), once, msg=f"{name}: {text[:30]!r}")

    def test_writes_the_closed_lines(self) -> None:
        """Writes the reserved-rights line and no grant in a closed repository."""
        text = _write("graph.hpp", K_DOC, Header("AcmeWeb - https://acme.example", closed=True))
        self.assertIn("/* All rights reserved. ", text)
        self.assertIn("No licence is granted.", text)
        self.assertEqual(text.count("https://acme.example"), 1)
        self.assertNotIn("Licence 1.0", text)
        self.assertNotIn("Commercial use", text)

    def test_appends_upstream_lines(self) -> None:
        """Appends a ported file's upstream notice below the repository's own lines."""
        upstream = ("Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        header = Header(K_PROJECT, upstream={"source/lapjv.cpp": upstream})
        rows = rewrite("source/lapjv.cpp", K_DOC, header, 2026).split("\n\n")[0].splitlines()
        self.assertEqual(rows[1].strip("/* "), "lapjv.cpp")
        self.assertEqual([row.strip("/* ") for row in rows[-4:-1]], ["", *upstream])
        self.assertNotIn("Tomas", rewrite("source/other.cpp", K_DOC, header, 2026))

    def test_keeps_upstream_lines_on_a_plain_rerun(self) -> None:
        """Keeps a ported file's upstream rows when a later run names no upstream for it."""
        upstream = ("Portions Copyright (c) 2012 Tomas Kazmar", "BSD-2-Clause")
        ported = Header(K_PROJECT, upstream={"lapjv.cpp": upstream})
        for name, body in (("lapjv.cpp", K_DOC), ("lapjv.py", K_DOCSTRING)):
            first = rewrite(name, body, Header(K_PROJECT, upstream={name: upstream}), 2026)
            self.assertIn("Tomas Kazmar", first)
            self.assertEqual(rewrite(name, first, K_HEADER, 2026), first)
        replaced = rewrite("lapjv.cpp", rewrite("lapjv.cpp", K_DOC, ported, 2026),
                           Header(K_PROJECT, upstream={"lapjv.cpp": ("MIT",)}), 2026)
        self.assertNotIn("Tomas Kazmar", replaced)
        self.assertIn("/* MIT ", replaced)

    def test_drops_foreign_rows_of_an_old_block(self) -> None:
        """Carries nothing over from a block that lacks this repository's licence lines."""
        self.assertNotIn("Apache", _write("graph.hpp", f"{K_OLD_BOX}\n\n{K_DOC}"))

    def test_keeps_upstream_rows_of_the_narrow_box(self) -> None:
        """Keeps the upstream rows of the one-section box the writer produced before."""
        narrow = "\n".join([
            "/" + "*" * 64 + "/",
            "/* lapjv.cpp                                                    */",
            "/* Copyright (c) 2026 Acme Corp                                 */",
            "/* Licensed under the Acme Licence 1.0. See LICENSE.            */",
            "/* Commercial use requires a licence from Acme Corp.            */",
            "/* Portions Copyright (c) 2012 Tomas Kazmar                     */",
            "/" + "*" * 64 + "/",
        ])
        written = _write("lapjv.cpp", f"{narrow}\n\n{K_DOC}")
        self.assertIn("/* Portions Copyright (c) 2012 Tomas Kazmar ", written)
        self.assertEqual(written.count("Copyright (c)"), 2)

    def test_reads_templates_and_include_fragments_as_c_family(self) -> None:
        """Gives a configure template and an include fragment the block of the file they become."""
        for name in ("version.hpp.in", "kernels.inc", "config.h.in"):
            self.assertTrue(is_source(name), msg=name)
            self.assertTrue(_write(name, K_DOC).startswith("/***"), msg=name)
        self.assertTrue(_write("settings.py.in", K_DOCSTRING).startswith("# settings.py.in"))
        self.assertFalse(is_source("notes.txt.in"))

    def test_replaces_an_indented_old_box(self) -> None:
        """Replaces an old boxed block whose rows are indented."""
        indented = "\n".join("  " + row for row in K_OLD_BOX.splitlines())
        self.assertEqual(_write("graph.hpp", f"{indented}\n\n{K_DOC}"), f"{K_BOX}\n\n{K_DOC}")

    def test_widens_the_box_for_a_long_line(self) -> None:
        """Keeps every row and both edges the same width when a line is long."""
        header = Header("AcmeRuntime - " + "x" * 80)
        rows = rewrite("graph.hpp", K_DOC, header, 2026).split("\n\n")[0].splitlines()
        self.assertEqual(len({len(row) for row in rows}), 1)

    def test_uses_the_year_it_is_given(self) -> None:
        """Writes the file's first year into the copyright line."""
        self.assertIn("Copyright (c) 2024-present Jane Doe & Acme Corp", _write("graph.py", "", year=2024))

    def test_output_passes_the_checker(self) -> None:
        """Produces blocks the license block rule accepts."""
        for name, text in (("graph.hpp", K_DOC), ("graph.py", K_DOCSTRING)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / name
                path.write_text(rewrite(name, text, Header(K_PROJECT, closed=K_ACME.closed), 2026), encoding="utf-8")
                issues = [issue for source in collect([path]) for issue in rule_license_block(source)]
                self.assertEqual(issues, [])


def _git(root: Path, *arguments: str) -> None:
    """Runs one git command in a repository with a fixed identity and date."""
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "-C", str(root), *identity, *arguments], check=True, capture_output=True)


class RepositoryTest(unittest.TestCase):
    """Checks the writer against a real git repository."""

    def _repository(self, folder: str) -> Path:
        """Returns a repository with one file added in 2024 and renamed in 2026."""
        root = Path(folder)
        _git(root, "init", "-q")
        (root / "old.py").write_text(K_DOCSTRING, encoding="utf-8")
        _git(root, "add", "old.py")
        _git(root, "commit", "-q", "-m", "add", "--date=2024-03-01T00:00:00")
        _git(root, "mv", "old.py", "graph.py")
        (root / "third_party").mkdir()
        (root / "third_party" / "vendored.py").write_text("x = 1\n", encoding="utf-8")
        (root / "notes.md").write_text("# Notes\n", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "rename", "--date=2026-03-01T00:00:00")
        return root

    def test_first_year_follows_a_rename(self) -> None:
        """Dates a renamed file from the commit that first added it."""
        with tempfile.TemporaryDirectory() as folder:
            years = first_years(self._repository(folder))
            self.assertEqual(years["graph.py"], 2024)

    def test_report_writes_nothing_and_run_converges(self) -> None:
        """Reports a pending file without touching it, then writes it once."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._repository(folder)
            arguments = [str(root), "--project", K_PROJECT]
            self.assertEqual(_main([*arguments, "--report"]), 1)
            self.assertEqual((root / "graph.py").read_text(encoding="utf-8"), K_DOCSTRING)
            self.assertEqual(_main(arguments), 0)
            self.assertIn("Copyright (c) 2024-present ", (root / "graph.py").read_text(encoding="utf-8"))
            self.assertEqual((root / "third_party" / "vendored.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertEqual(_main([*arguments, "--report"]), 0)


    def test_writes_a_file_that_is_staged_and_not_yet_committed(self) -> None:
        """Gives a newly added file its block, dated this year."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._repository(folder)
            (root / "fresh.py").write_text(K_DOCSTRING, encoding="utf-8")
            _git(root, "add", "fresh.py")
            self.assertEqual(_main([str(root), "--project", K_PROJECT]), 0)
            written = (root / "fresh.py").read_text(encoding="utf-8")
            self.assertIn(f"Copyright (c) {date.today().year}-present ", written)

    def test_skips_a_tracked_file_deleted_from_the_working_tree(self) -> None:
        """Leaves out a tracked file the working tree no longer holds."""
        with tempfile.TemporaryDirectory() as folder:
            root = self._repository(folder)
            (root / "graph.py").unlink()
            self.assertEqual(_main([str(root), "--project", K_PROJECT, "--report"]), 0)


if __name__ == "__main__":
    unittest.main()

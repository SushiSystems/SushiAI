# test_checker.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the command line every checker shares, from tools/checking/checker.py."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Finding, run  # noqa: E402

K_CHECKER = Checker(description="Checks nothing.", collect=lambda paths: [],
                    rules={"rule_first": lambda subject: iter(()), "rule_second": lambda subject: iter(())})


K_TEST_SIZE = "small"


class CheckerTest(unittest.TestCase):
    """Runs a checker with two rules and no subjects."""

    def test_lists_its_rules_one_a_line(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, ["--list-rules"])
        self.assertEqual((code, output.getvalue()), (0, "rule_first\nrule_second\n"))


class FindingFormatTest(unittest.TestCase):
    """Formats a finding below a link as a path below the root."""

    def _link(self, link: Path, target: Path) -> None:
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, capture_output=True)

    def test_a_path_below_a_link_is_shown_below_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root, elsewhere = Path(folder) / "root", Path(folder) / "elsewhere"
            root.mkdir()
            elsewhere.mkdir()
            (elsewhere / "file.py").write_text("x = 1\n", encoding="utf-8")
            self._link(root / "third_party", elsewhere)
            shown = Finding(root / "third_party" / "file.py", 3, "rule_x", "bad").format(root)
        self.assertEqual(shown, "third_party/file.py:3: rule_x: bad")

    def test_a_path_outside_the_root_keeps_its_own_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            root.mkdir()
            outside = Path(folder) / "other" / "file.py"
            shown = Finding(outside, 1, "rule_x", "bad").format(root)
        self.assertEqual(shown, f"{outside.as_posix()}:1: rule_x: bad")


if __name__ == "__main__":
    unittest.main()

# test_check_all.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests which checkers tools/check_all.py runs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_all import K_CHECKERS, own_checkers  # noqa: E402


class OwnCheckersTest(unittest.TestCase):
    """Covers how a repository's own checkers are found beside the shared ones."""

    def test_finds_a_checker_the_shared_list_does_not_name(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tools = Path(folder)
            for name in (*K_CHECKERS, "layering/check_package_layers.py", "tests/check_fake.py",
                         "layering/helper.py"):
                (tools / name).parent.mkdir(parents=True, exist_ok=True)
                (tools / name).write_text("\n", encoding="utf-8")
            self.assertEqual(own_checkers(tools), ("layering/check_package_layers.py",))


if __name__ == "__main__":
    unittest.main()

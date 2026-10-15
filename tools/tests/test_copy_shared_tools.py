# test_copy_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the shared tools copy on an upstream and a repository built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.copy_shared_tools import copy, merged, pending  # noqa: E402

K_UPSTREAM_HEADER = "# check.py\n# Upstream - https://example.org/upstream\n"
K_LOCAL_HEADER = "# check.py\n# Local - https://example.org/local\n"
K_BODY = '"""Checks something."""\n\nK_VALUE = 2\n'


def _write(root: Path, relative: str, text: str) -> None:
    """Writes a file and its folders below root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class MergedTest(unittest.TestCase):
    """Covers how one shared file is rebuilt for a repository."""

    def test_a_python_file_keeps_the_repository_header(self) -> None:
        self.assertEqual(merged("x.py", K_UPSTREAM_HEADER + K_BODY, K_LOCAL_HEADER + "old\n"),
                         K_LOCAL_HEADER + K_BODY)

    def test_a_new_python_file_takes_the_upstream_header(self) -> None:
        self.assertEqual(merged("x.py", K_UPSTREAM_HEADER + K_BODY, None), K_UPSTREAM_HEADER + K_BODY)

    def test_another_file_is_copied_as_it_is(self) -> None:
        self.assertEqual(merged("README.md", "# tools\nupstream\n", "# tools\nlocal\n"), "# tools\nupstream\n")


class CopyTest(unittest.TestCase):
    """Covers which files a copy writes and which it leaves alone."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.upstream = Path(self._folder.name) / "upstream"
        self.root = Path(self._folder.name) / "repository"
        _write(self.upstream, "tools/layering/check.py", K_UPSTREAM_HEADER + K_BODY)
        _write(self.upstream, "tools/settings.py", "K_TIER_ORDER = ()\n")
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + '"""Checks something."""\n\nK_VALUE = 1\n')
        _write(self.root, "tools/settings.py", 'K_TIER_ORDER = ("core",)\n')
        _write(self.root, "tools/naming/check_naming.py", "local\n")
        _write(self.upstream, "tools/README.md", "# tools\nupstream\n")
        _write(self.root, "tools/README.md", "# tools\nlocal\n")
        self.shared = ["tools/README.md", "tools/layering/check.py", "tools/settings.py"]

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_a_changed_shared_file_is_pending(self) -> None:
        self.assertEqual(pending(self.upstream, self.root, self.shared), ["tools/layering/check.py"])

    def test_a_copy_writes_shared_files_and_keeps_settings_and_local_tools(self) -> None:
        copy(self.upstream, self.root, self.shared)
        self.assertEqual((self.root / "tools/layering/check.py").read_text(encoding="utf-8"), K_LOCAL_HEADER + K_BODY)
        self.assertEqual((self.root / "tools/settings.py").read_text(encoding="utf-8"), 'K_TIER_ORDER = ("core",)\n')
        self.assertEqual((self.root / "tools/naming/check_naming.py").read_text(encoding="utf-8"), "local\n")
        self.assertEqual((self.root / "tools/README.md").read_text(encoding="utf-8"), "# tools\nlocal\n")
        self.assertEqual(pending(self.upstream, self.root, self.shared), [])

    def test_missing_settings_are_written_from_upstream(self) -> None:
        (self.root / "tools/settings.py").unlink()
        self.assertEqual(pending(self.upstream, self.root, self.shared),
                         ["tools/layering/check.py", "tools/settings.py"])
        copy(self.upstream, self.root, self.shared)
        self.assertEqual((self.root / "tools/settings.py").read_text(encoding="utf-8"), "K_TIER_ORDER = ()\n")


if __name__ == "__main__":
    unittest.main()

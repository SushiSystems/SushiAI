# test_check_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the shared tools checker on repositories written to a temporary folder."""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.check_shared_tools import K_CHECKER, K_NO_LOCK, K_SOURCE_MARKER  # noqa: E402
from checking.checker import run  # noqa: E402

K_SHARED_FILE = "tools/checking/checker.py"
K_SHARED_TEXT = b"print('shared')\n"


def _digest(content: bytes) -> str:
    """Returns the SHA-256 of some bytes as hex, as the lock stores it."""
    return hashlib.sha256(content).hexdigest()


class SharedToolsTest(unittest.TestCase):
    """Runs the checker's command line over a repository built for each case."""

    K_TEST_SIZE = "small"

    def setUp(self) -> None:
        """Makes an empty repository folder that the test deletes afterwards."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _write(self, path: str, content: bytes) -> None:
        """Writes a file below the root, with its folders."""
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def _write_lock(self, files: dict[str, str]) -> None:
        """Writes the lock of the repository, with the given hash for each file."""
        rows = "".join(f'"{path}" = "{value}"\n' for path, value in files.items())
        header = 'tag = "v0.10.0"\ncommit = "0"\n\n[files]\n'
        self._write("tools/shared_tools.lock", (header + rows).encode("utf-8"))

    def _check(self) -> tuple[int, str]:
        """Runs the checker over the repository and returns its exit code and printed output."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, [str(self.root)])
        return code, output.getvalue()

    def test_missing_lock_in_a_consumer_fails(self) -> None:
        """A repository other than SushiCore that holds no lock fails with the lock message."""
        code, output = self._check()
        self.assertEqual(code, 1)
        self.assertIn("rule_lock_present", output)
        self.assertIn(K_NO_LOCK, output)

    def test_sushicore_root_without_lock_passes(self) -> None:
        """The SushiCore source holds no lock and passes, detected by its shared_tools package."""
        self._write(f"{K_SOURCE_MARKER.as_posix()}/__init__.py", b"")
        code, output = self._check()
        self.assertEqual(code, 0)
        self.assertEqual(output, "")

    def test_edited_shared_file_fails(self) -> None:
        """A locked file whose bytes differ from its hash fails and is named in the output."""
        self._write(K_SHARED_FILE, K_SHARED_TEXT)
        self._write_lock({K_SHARED_FILE: _digest(K_SHARED_TEXT)})
        self._write(K_SHARED_FILE, K_SHARED_TEXT + b"\n")
        code, output = self._check()
        self.assertEqual(code, 1)
        self.assertIn(f"{K_SHARED_FILE}:1: rule_shared_file_matches", output)

    def test_missing_shared_file_fails(self) -> None:
        """A locked file that is not on disk fails and is named in the output."""
        self._write_lock({K_SHARED_FILE: _digest(K_SHARED_TEXT)})
        code, output = self._check()
        self.assertEqual(code, 1)
        self.assertIn(f"{K_SHARED_FILE}:1: rule_shared_file_matches", output)

    def test_matching_lock_passes(self) -> None:
        """A repository whose every locked file hashes to the lock passes with no output."""
        self._write(K_SHARED_FILE, K_SHARED_TEXT)
        self._write_lock({K_SHARED_FILE: _digest(K_SHARED_TEXT)})
        code, output = self._check()
        self.assertEqual(code, 0)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()

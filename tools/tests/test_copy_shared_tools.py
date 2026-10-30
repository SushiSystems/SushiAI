# test_copy_shared_tools.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the shared tools copy on an upstream and a repository built on disk."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.copy_shared_tools import (  # noqa: E402
    LocalEditsError,
    copy,
    known_bodies,
    local_edits,
    main,
    merged,
    pending,
)

K_UPSTREAM_HEADER = "# check.py\n# Upstream - https://example.org/upstream\n"
K_LOCAL_HEADER = "# check.py\n# Local - https://example.org/local\n"
K_BODY = '"""Checks something."""\n\nK_VALUE = 2\n'
K_OLD_BODY = '"""Checks something."""\n\nK_VALUE = 1\n'
K_EDITED_BODY = '"""Checks something."""\n\nK_VALUE = 99\n'


def _write(root: Path, relative: str, text: str) -> None:
    """Writes a file and its folders below root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(root: Path, *arguments: str) -> None:
    """Runs one git command in a repository with a fixed identity."""
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "-C", str(root), *identity, *arguments], check=True, capture_output=True)


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
        _write(self.upstream, "tools/layering/check.py", K_UPSTREAM_HEADER + K_OLD_BODY)
        _git(self.upstream, "init", "-q")
        _git(self.upstream, "add", "-A")
        _git(self.upstream, "commit", "-q", "-m", "first")
        _write(self.upstream, "tools/layering/check.py", K_UPSTREAM_HEADER + K_BODY)
        _write(self.upstream, "tools/settings.py", "K_TIER_ORDER = ()\n")
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_OLD_BODY)
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


    def test_an_older_upstream_version_is_an_upgrade_not_a_local_edit(self) -> None:
        self.assertEqual(local_edits(self.upstream, self.root, self.shared), [])

    def test_a_text_no_upstream_version_held_is_a_local_edit(self) -> None:
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_EDITED_BODY)
        self.assertEqual(local_edits(self.upstream, self.root, self.shared), ["tools/layering/check.py"])

    def test_a_copy_refuses_a_local_edit_and_writes_nothing(self) -> None:
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_EDITED_BODY)
        (self.root / "tools/settings.py").unlink()
        with self.assertRaises(LocalEditsError) as refused:
            copy(self.upstream, self.root, self.shared)
        self.assertEqual(refused.exception.files, ["tools/layering/check.py"])
        self.assertEqual((self.root / "tools/layering/check.py").read_text(encoding="utf-8"),
                         K_LOCAL_HEADER + K_EDITED_BODY)
        self.assertFalse((self.root / "tools/settings.py").exists())

    def test_force_overwrites_a_local_edit(self) -> None:
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_EDITED_BODY)
        copy(self.upstream, self.root, self.shared, force=True)
        self.assertEqual((self.root / "tools/layering/check.py").read_text(encoding="utf-8"), K_LOCAL_HEADER + K_BODY)

    def test_a_file_the_repository_lacks_is_never_a_local_edit(self) -> None:
        (self.root / "tools/layering/check.py").unlink()
        self.assertEqual(local_edits(self.upstream, self.root, self.shared), [])

    def test_crlf_newlines_alone_are_not_a_local_edit(self) -> None:
        crlf = (K_LOCAL_HEADER + K_OLD_BODY).replace("\n", "\r\n")
        (self.root / "tools/layering/check.py").write_bytes(crlf.encode("utf-8"))
        self.assertEqual(local_edits(self.upstream, self.root, self.shared), [])

    def test_an_upstream_without_history_knows_only_its_current_text(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder)
            _write(plain, "tools/x.py", K_UPSTREAM_HEADER + K_BODY)
            self.assertEqual(known_bodies(plain, "tools/x.py"), {K_BODY})

    def test_the_command_reports_a_local_edit_and_refuses_to_copy_it(self) -> None:
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_EDITED_BODY)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.root), "--from", str(self.upstream), "--report"]), 1)
        self.assertIn("tools/layering/check.py: differs from upstream (local edit)", output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.root), "--from", str(self.upstream)]), 1)


    def test_a_kept_file_is_left_alone_and_the_rest_is_copied(self) -> None:
        _git(self.upstream, "add", "-A")
        _git(self.upstream, "commit", "-q", "-m", "second")
        _write(self.root, "tools/layering/check.py", K_LOCAL_HEADER + K_EDITED_BODY)
        (self.root / "tools/settings.py").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([str(self.root), "--from", str(self.upstream), "--keep", "tools/layering/check.py"])
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "tools/layering/check.py").read_text(encoding="utf-8"),
                         K_LOCAL_HEADER + K_EDITED_BODY)
        self.assertTrue((self.root / "tools/settings.py").exists())


class NewFileHeaderTest(unittest.TestCase):
    """Checks that a shared file new to a repository takes the repository's header."""

    def test_a_new_python_file_takes_the_template_header_with_its_own_name(self) -> None:
        upstream = "# check_x.py\n# SushiSkills - url\n# Licensed A.\n\"\"\"Doc.\"\"\"\nbody\n"
        template = "# check_all.py\n# AcmeEngine - url\n# All rights reserved.\n\"\"\"Other.\"\"\"\n"
        self.assertEqual(merged("tools/testing/check_x.py", upstream, None, template),
                         "# check_x.py\n# AcmeEngine - url\n# All rights reserved.\n\"\"\"Doc.\"\"\"\nbody\n")

    def test_without_a_template_the_upstream_header_stays(self) -> None:
        upstream = "# check_x.py\n# SushiSkills - url\nbody\n"
        self.assertEqual(merged("tools/testing/check_x.py", upstream, None, None), upstream)


if __name__ == "__main__":
    unittest.main()

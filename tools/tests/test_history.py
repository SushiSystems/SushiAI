# test_history.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests reading the commits of a range and the text of objects, each in one git call.

The listings git prints are parsed from values in memory; git itself runs once, in the one medium class.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from commits.history import (  # noqa: E402
    K_FIELD,
    K_RECORD,
    blob_texts,
    commits_since,
    git,
    parse_blob_stream,
    parse_log,
)
from repo_builder import Repo  # noqa: E402

K_ROOT = Path("repo")


def _record(sha: str, parent: str, message: str) -> str:
    """Returns one commit as git prints it in the log format of history.py."""
    return f"{sha}{K_FIELD}{parent}{K_FIELD}{message}{K_RECORD}"


def _object(name: str, content: bytes) -> bytes:
    """Returns one object as `git cat-file --batch` prints it."""
    return f"{name} blob {len(content)}\n".encode() + content + b"\n"


class ParseLogTest(unittest.TestCase):
    """Covers turning a git log listing into commits."""

    K_TEST_SIZE = "small"

    def test_each_record_becomes_a_commit_in_listing_order(self) -> None:
        listing = _record("aaa", "p1", "one\n\n") + "\n" + _record("bbb", "aaa", "two\n")
        found = parse_log(K_ROOT, listing)
        self.assertEqual([(c.root, c.sha, c.parent, c.message) for c in found],
                         [(K_ROOT, "aaa", "p1", "one\n"), (K_ROOT, "bbb", "aaa", "two\n")])

    def test_a_commit_without_a_parent_is_left_out(self) -> None:
        listing = _record("aaa", "", "root\n") + _record("bbb", "aaa", "next\n")
        self.assertEqual([c.sha for c in parse_log(K_ROOT, listing)], ["bbb"])

    def test_an_empty_listing_holds_no_commit(self) -> None:
        self.assertEqual(parse_log(K_ROOT, ""), [])
        self.assertEqual(parse_log(K_ROOT, "\n"), [])

    def test_the_whole_message_is_kept_with_one_closing_newline(self) -> None:
        found = parse_log(K_ROOT, _record("aaa", "p", "subject\n\nbody\n\nBacklog: none (x)\n\n\n"))
        self.assertEqual(found[0].message, "subject\n\nbody\n\nBacklog: none (x)\n")

    def test_a_message_may_hold_the_field_mark(self) -> None:
        found = parse_log(K_ROOT, _record("aaa", "p", f"a{K_FIELD}b"))
        self.assertEqual(found[0].message, f"a{K_FIELD}b\n")


class ParseBlobStreamTest(unittest.TestCase):
    """Covers turning the output of `git cat-file --batch` into texts."""

    K_TEST_SIZE = "small"

    def test_each_object_is_read_by_its_size(self) -> None:
        data = _object("h1", b"one\ntwo\n") + _object("h2", b"x")
        self.assertEqual(parse_blob_stream(data), {"h1": "one\ntwo\n", "h2": "x"})

    def test_an_object_git_does_not_hold_is_left_out_and_the_next_is_read(self) -> None:
        data = b"h1 missing\n" + _object("h2", b"kept")
        self.assertEqual(parse_blob_stream(data), {"h2": "kept"})

    def test_bytes_that_are_not_utf8_are_replaced(self) -> None:
        self.assertEqual(parse_blob_stream(_object("h1", b"a\xffb")), {"h1": "a�b"})

    def test_no_output_holds_no_text(self) -> None:
        self.assertEqual(parse_blob_stream(b""), {})


class HistoryTest(unittest.TestCase):
    """Builds a repository of three commits and reads it back through git."""

    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self.repo = Repo()
        self.addCleanup(self.repo.close)
        self.repo.write("base.md", "base"+chr(10))
        self.base = self.repo.commit("first")
        self.repo.write("a.md", "one\n")
        self.second = self.repo.commit("second" + chr(10) + chr(10) + "Backlog: none (test)")
        self.repo.write("a.md", "two\n")
        self.third = self.repo.commit("third")

    def test_the_commits_after_a_base_come_oldest_first(self) -> None:
        found = commits_since(self.repo.root, self.base.sha)
        self.assertEqual([c.sha for c in found], [self.second.sha, self.third.sha])

    def test_each_commit_gives_its_parent(self) -> None:
        found = commits_since(self.repo.root, self.base.sha)
        self.assertEqual([c.parent for c in found], [self.base.sha, self.second.sha])

    def test_each_commit_gives_its_whole_message(self) -> None:
        found = commits_since(self.repo.root, self.base.sha)
        self.assertIn("Backlog: none (test)", found[0].message)

    def test_commits_since_leaves_out_merge_commits(self) -> None:
        self.repo.git("checkout", "-q", "-b", "side", self.base.sha)
        self.repo.write("side.md", "side\n")
        side = self.repo.commit("side work")
        self.repo.git("checkout", "-q", "-")
        self.repo.git("merge", "--no-ff", "-q", "side", "-m", "Merge branch side")
        found = commits_since(self.repo.root, self.base.sha)
        self.assertEqual(sorted(c.sha for c in found), sorted([self.second.sha, self.third.sha, side.sha]))

    def test_commits_since_stops_with_a_message_when_git_cannot_read_the_range(self) -> None:
        with self.assertRaises(SystemExit) as stopped:
            commits_since(self.repo.root, "0" * 40)
        self.assertEqual(stopped.exception.code,
                         f"cannot read the history after {'0' * 40}; fetch the full history")

    def test_git_returns_none_when_a_git_command_fails(self) -> None:
        self.assertIsNone(git(self.repo.root, "rev-parse", "--verify", "no-such-ref"))

    def test_a_blob_is_read_by_its_hash(self) -> None:
        blob = self.repo.git("rev-parse", "HEAD:a.md").strip()
        self.assertEqual(blob_texts(self.repo.root, [blob]), {blob: "two\n"})

    def test_an_unknown_hash_is_left_out(self) -> None:
        self.assertEqual(blob_texts(self.repo.root, ["0" * 40]), {})


if __name__ == "__main__":
    unittest.main()

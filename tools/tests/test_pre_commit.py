# test_pre_commit.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that the pre-commit hook checks the repository in place, with a stub for check_all."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hooks import pre_commit  # noqa: E402
from repo_builder import Repo, without_commit_dates  # noqa: E402

K_STUB = """import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
seen = []
if (root / "third_party" / "sub" / "marker.txt").is_file():
    seen.append("submodule")
fields = [str(root), os.environ.get("GIT_INDEX_FILE", ""), " ".join(sys.argv[2:]), ",".join(seen)]
with open(os.environ["PRE_COMMIT_TEST_LOG"], "a", encoding="utf-8") as log:
    log.write("\\t".join(fields) + "\\n")
sys.exit(1 if (root / "bad.txt").exists() else 0)
"""


class PreCommitTest(unittest.TestCase):
    """Runs the hook with a check_all that fails when `bad.txt` exists in the folder it is given."""

    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        # The repository's own parent, so no other process writes beside it during a test.
        parent = tempfile.TemporaryDirectory()
        self.addCleanup(parent.cleanup)
        self.repo = Repo(Path(parent.name))
        self.addCleanup(self.repo.close)
        self.log = self.repo.root.parent / f"{self.repo.root.name}_checked.log"
        self.addCleanup(lambda: self.log.unlink(missing_ok=True))
        self.repo.write("tools/check_all.py", K_STUB)
        self.repo.write("src/a.py", "x = 1\n")
        self.repo.commit("first")

    def _run(self, **extra: str) -> int:
        environment = {**without_commit_dates(), "PRE_COMMIT_TEST_LOG": str(self.log), **extra}
        for name in pre_commit_git_variables():
            if name not in extra:
                environment.pop(name, None)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = pre_commit.run(self.repo.root, environment)
        self.errors = errors.getvalue()
        return code

    def _logged(self) -> list[tuple[Path, str]]:
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [(Path(line.split("\t")[0]), line.split("\t")[1]) for line in lines]

    def _logged_arguments(self) -> list[str]:
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [line.split("\t")[2] for line in lines]

    def _logged_sightings(self) -> list[str]:
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [line.split("\t")[3] for line in lines]

    def _git_dir(self) -> Path:
        return Path(self.repo.git("rev-parse", "--absolute-git-dir").strip())

    def test_a_red_file_the_commit_holds_refuses_it(self) -> None:
        self.repo.write("bad.txt", "staged\n")
        self.repo.git("add", "bad.txt")
        self.assertEqual(self._run(), 1)

    def test_a_clean_commit_passes(self) -> None:
        self.repo.write("src/a.py", "x = 2\n")
        self.repo.git("add", "src/a.py")
        self.assertEqual(self._run(), 0)

    def test_a_staged_file_changed_on_disk_refuses_and_is_named_without_running_checks(
            self) -> None:
        self.repo.write("src/a.py", "x = 2\n")
        self.repo.git("add", "src/a.py")
        self.repo.write("src/a.py", "x = 3\n")
        code = self._run()
        self.assertEqual((code, "src/a.py" in self.errors, self.log.exists()), (1, True, False))

    def test_a_staged_file_deleted_on_disk_refuses_the_commit(self) -> None:
        self.repo.write("bad.txt", "staged\n")
        self.repo.git("add", "bad.txt")
        self.repo.remove("bad.txt")
        code = self._run()
        self.assertEqual((code, "bad.txt" in self.errors), (1, True))

    def test_the_checks_run_in_the_repository_itself(self) -> None:
        self._run()
        [(folder, _)] = self._logged()
        self.assertEqual(folder, self.repo.root.resolve())

    def test_nothing_is_written_beside_the_repository(self) -> None:
        before = sorted(self.repo.root.parent.iterdir())
        self._run()
        after = [path for path in sorted(self.repo.root.parent.iterdir()) if path != self.log]
        self.assertEqual(after, before)

    def test_a_submodule_the_index_names_is_present_in_the_checked_tree(self) -> None:
        sub = Repo()
        self.addCleanup(sub.close)
        sub.write("marker.txt", "x\n")
        sub.commit("marker")
        self.repo.git("-c", "protocol.file.allow=always", "submodule", "add", "-q",
                      str(sub.root), "third_party/sub")
        self._run()
        self.assertEqual(self._logged_sightings(), ["submodule"])

    def test_the_checkers_that_read_the_history_are_left_out(self) -> None:
        self._run()
        self.assertEqual(self._logged_arguments(), ["--skip-history"])

    def test_the_repository_index_is_checked_when_git_hands_no_other(self) -> None:
        self._run()
        [(_, index)] = self._logged()
        self.assertEqual(Path(index), self._git_dir() / "index")

    def test_the_index_git_hands_the_hook_is_the_one_checked(self) -> None:
        handed = self._git_dir() / "next-index"
        shutil.copyfile(self._git_dir() / "index", handed)
        code = self._run(GIT_INDEX_FILE=str(handed))
        [(_, index)] = self._logged()
        self.assertEqual((code, Path(index)), (0, handed))

    def test_a_red_file_only_the_handed_index_holds_refuses_the_commit(self) -> None:
        self.repo.write("bad.txt", "staged\n")
        self.repo.git("add", "bad.txt")
        handed = self._git_dir() / "next-index"
        shutil.copyfile(self._git_dir() / "index", handed)
        self.repo.git("rm", "--cached", "bad.txt")
        self.assertEqual(self._run(GIT_INDEX_FILE=str(handed)), 1)

    def test_an_index_git_cannot_read_is_refused_and_named(self) -> None:
        broken = self._git_dir() / "broken-index"
        broken.write_text("not an index", encoding="utf-8")
        code = self._run(GIT_INDEX_FILE=str(broken))
        self.assertEqual((code, "cannot read the index" in self.errors), (1, True))

    def test_an_index_without_check_all_is_refused_and_named(self) -> None:
        self.repo.git("rm", "--cached", "tools/check_all.py")
        code = self._run()
        self.assertEqual((code, "holds no tools/check_all.py" in self.errors), (1, True))

    def test_a_folder_that_is_no_repository_is_refused(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = pre_commit.run(self.repo.root.parent / "no_such_repository",
                                  {"PATH": os.environ["PATH"]})
        self.assertEqual((code, "not a git repository" in errors.getvalue()), (1, True))


class CheckedEnvironmentTest(unittest.TestCase):
    """Tests the environment `check_all.py` runs in."""

    K_TEST_SIZE = "small"
    base = Path(tempfile.gettempdir())

    def test_git_points_at_the_repository_and_its_tree(self) -> None:
        git_dir, tree = self.base / ".git", self.base / "tree"
        environment = pre_commit.checked_environment(git_dir, tree, {"PATH": "p"})
        self.assertEqual(
            (environment["GIT_DIR"], environment["GIT_WORK_TREE"], environment["PATH"]),
            (str(git_dir), str(tree), "p"))

    def test_the_index_git_hands_the_hook_replaces_the_repository_index(self) -> None:
        git_dir = self.base / ".git"
        handed = str(git_dir / "next-index")
        environment = pre_commit.checked_environment(
            git_dir, self.base / "tree", {"GIT_INDEX_FILE": handed})
        self.assertEqual(environment["GIT_INDEX_FILE"], handed)

    def test_the_repository_index_is_used_when_git_hands_none(self) -> None:
        git_dir = self.base / ".git"
        environment = pre_commit.checked_environment(git_dir, self.base / "tree", {})
        self.assertEqual(environment["GIT_INDEX_FILE"], str(git_dir / "index"))


def pre_commit_git_variables() -> tuple[str, ...]:
    """Returns the variables that would point git at the repository that runs the tests."""
    return ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX")


if __name__ == "__main__":
    unittest.main()

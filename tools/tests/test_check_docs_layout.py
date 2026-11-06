# test_check_docs_layout.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests each rule of the docs layout checker on values, and its shell on trees on disk."""

from __future__ import annotations

import dataclasses
import datetime
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_docs_layout import (  # noqa: E402
    K_CHECKER,
    K_REQUIRED_ENTRIES,
    Repository,
    collect,
    link_probes,
    rule_closed_work,
    rule_design_ceiling,
    rule_docs_entries,
    rule_document_names,
    rule_forbidden_kinds,
    rule_link_anchors,
    rule_links,
    rule_module_readmes,
    rule_reachable_from_index,
    rule_required_entries,
    rule_status_lines,
    rule_work_folders,
)

K_ROOT = Path("repo")
K_TODAY = datetime.date(2026, 11, 4)
K_SCRIPT = Path(__file__).resolve().parents[1] / "documentation" / "check_docs_layout.py"


def _repository(files: dict[str, str]) -> Repository:
    """Returns the values the shell reads for a repository whose files hold these texts.

    The clock is fixed at K_TODAY, and a link probe finds a target when the files hold it.
    """
    folders = {"docs"}
    for name in files:
        parts = PurePosixPath(name).parts
        folders.update("/".join(parts[:index]) for index in range(1, len(parts)))
    documents = {
        name: text for name, text in files.items()
        if name.startswith("docs/") and name.endswith(".md")
    }
    reached, linked = link_probes(documents)
    texts = {
        **documents,
        **{name: files[name] for name in linked if name in files and name not in documents},
    }
    entries = {
        PurePosixPath(name).name for name in (*folders, *files)
        if PurePosixPath(name).parent == PurePosixPath("docs")
    }
    return Repository(
        root=K_ROOT,
        today=K_TODAY,
        folders=frozenset(folders),
        files=frozenset(files),
        texts=texts,
        present=frozenset(name for name in reached if name in files or name in folders),
        docs_entries=frozenset(entries),
    )


def _names(issues: Iterable[Issue]) -> list[str]:
    """Returns the docs-relative path of each issue."""
    return [issue.path.as_posix().split("/docs/", 1)[1] for issue in issues]


def _days_ago(days: int) -> str:
    """Returns the ISO date the given number of days before the fixed day of the tests."""
    return (K_TODAY - datetime.timedelta(days=days)).isoformat()


class RequiredEntriesTest(unittest.TestCase):
    """Checks that each required document is demanded."""
    K_TEST_SIZE = "small"

    def test_is_registered(self) -> None:
        """Runs both rules as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_required_entries"], rule_required_entries)
        self.assertIs(K_CHECKER.rules["rule_reachable_from_index"], rule_reachable_from_index)

    def test_reports_every_missing_document(self) -> None:
        """Reports all required documents for a docs/ tree that holds none."""
        repository = _repository({"docs/guides/X.md": "# X\n"})
        self.assertEqual(_names(rule_required_entries(repository)), list(K_REQUIRED_ENTRIES))

    def test_accepts_a_complete_tree(self) -> None:
        """Reports nothing when every required document exists."""
        repository = _repository({f"docs/{name}": "# T\n" for name in K_REQUIRED_ENTRIES})
        self.assertEqual(list(rule_required_entries(repository)), [])


class ReachabilityTest(unittest.TestCase):
    """Checks which documents count as reached from the index."""
    K_TEST_SIZE = "small"

    def test_follows_links_through_an_index(self) -> None:
        """Reaches a document linked from an index the root index links, anchor included."""
        files = {
            "docs/README.md": (
                "# Manual\n[design](design/README.md)\n[guide](guides/GUIDE.md#install)\n"
            ),
            "docs/design/README.md": "# Design\n[topic](TOPIC.md)\n",
            "docs/design/TOPIC.md": "# Topic\n",
            "docs/guides/GUIDE.md": "# Guide\n",
        }
        self.assertEqual(list(rule_reachable_from_index(_repository(files))), [])

    def test_follows_a_link_that_climbs_out_of_its_folder(self) -> None:
        """Reaches a document a link reaches through '..', by the path it normalises to."""
        files = {
            "docs/README.md": "# Manual\n[guide](guides/GUIDE.md)\n",
            "docs/guides/GUIDE.md": "# Guide\n[topic](../design/TOPIC.md)\n",
            "docs/design/TOPIC.md": "# Topic\n",
        }
        self.assertEqual(list(rule_reachable_from_index(_repository(files))), [])

    def test_reports_an_orphan(self) -> None:
        """Reports a document no index links."""
        files = {"docs/README.md": "# Manual\n", "docs/guides/ORPHAN.md": "# Orphan\n"}
        self.assertEqual(_names(rule_reachable_from_index(_repository(files))),
                         ["guides/ORPHAN.md"])

    def test_ignores_links_in_code(self) -> None:
        """Does not count a link inside a fence or a code span as reaching its target."""
        files = {
            "docs/README.md": "# Manual\n`[a](guides/A.md)`\n```\n[b](guides/B.md)\n```\n",
            "docs/guides/A.md": "# A\n",
            "docs/guides/B.md": "# B\n",
        }
        self.assertEqual(_names(rule_reachable_from_index(_repository(files))),
                         ["guides/A.md", "guides/B.md"])

    def test_exempts_work_folders_and_archive(self) -> None:
        """Reports nothing for unlinked documents under agent/ and archive/."""
        files = {
            "docs/README.md": "# Manual\n",
            "docs/agent/2026_10_05_WORK/SPEC.md": "# Spec\n",
            "docs/archive/OLD.md": "# Old\n",
        }
        self.assertEqual(list(rule_reachable_from_index(_repository(files))), [])

    def test_reads_titled_wrapped_bracketed_and_queried_links(self) -> None:
        """Reaches targets written with a title, a wrapped text, angle brackets or a query."""
        files = {
            "docs/README.md": (
                '# Manual\n[a](guides/A.md "the title")\n[a link whose text\nwraps](guides/B.md)\n'
                "[c](<guides/C.md>)\n[d](guides/D.md?plain=1)\n[e](guides/MY%20E.md)\n"
            ),
            "docs/guides/A.md": "# A\n",
            "docs/guides/B.md": "# B\n",
            "docs/guides/C.md": "# C\n",
            "docs/guides/D.md": "# D\n",
            "docs/guides/MY E.md": "# E\n",
        }
        repository = _repository(files)
        self.assertEqual(_names(rule_reachable_from_index(repository)), [])
        self.assertEqual(_names(rule_links(repository)), [])

    def test_reports_a_broken_titled_link_on_its_line(self) -> None:
        """Reports a titled link whose target is missing, with the line it starts on."""
        files = {"docs/README.md": '# Manual\n\n[a](guides/GONE.md "the title")\n'}
        issues = list(rule_links(_repository(files)))
        self.assertEqual([(issue.line, issue.message) for issue in issues],
                         [(3, "link target does not exist: guides/GONE.md")])

    def test_does_not_reach_through_a_file_outside_docs(self) -> None:
        """Reports a guide the index reaches only by way of the root README."""
        files = {
            "README.md": "[a](docs/guides/A.md)\n",
            "docs/README.md": "# Manual\n[front](../README.md)\n",
            "docs/guides/A.md": "# A\n",
        }
        self.assertEqual(_names(rule_reachable_from_index(_repository(files))), ["guides/A.md"])

    def test_does_not_reach_through_a_work_folder(self) -> None:
        """Reports a guide linked only from an agent work folder the index links."""
        files = {
            "docs/README.md": "# Manual\n[spec](agent/2026_10_05_WORK/SPEC.md)\n",
            "docs/agent/2026_10_05_WORK/SPEC.md": "# Spec\n[a](../../guides/A.md)\n",
            "docs/guides/A.md": "# A\n",
        }
        self.assertEqual(_names(rule_reachable_from_index(_repository(files))), ["guides/A.md"])

    def test_is_silent_without_an_index(self) -> None:
        """Reports nothing when docs/README.md is absent; the required-entry rule owns that."""
        repository = _repository({"docs/guides/X.md": "# X\n"})
        self.assertEqual(list(rule_reachable_from_index(repository)), [])


class DocsEntriesTest(unittest.TestCase):
    """Checks which entries may sit directly under docs/.

    The shell drops the entries git ignores before the rule sees them; ShellTest checks that.
    """
    K_TEST_SIZE = "small"

    def test_accepts_the_publish_list(self) -> None:
        """Accepts docs/publish.toml and still reports any other loose file."""
        files = {"docs/README.md": "# R\n", "docs/publish.toml": "", "docs/notes.txt": ""}
        self.assertEqual(_names(rule_docs_entries(_repository(files))), ["notes.txt"])

    def test_accepts_the_docs_truth_allow_list(self) -> None:
        """Accepts docs/.docs_truth_allow, the one configuration file a checker reads."""
        files = {"docs/README.md": "# R\n", "docs/.docs_truth_allow": ""}
        self.assertEqual(_names(rule_docs_entries(_repository(files))), [])


class LinkAnchorsTest(unittest.TestCase):
    """Covers the rule that a link's anchor names a heading of the document it reaches."""
    K_TEST_SIZE = "small"

    def test_accepts_a_heading_slug_and_a_label(self) -> None:
        """Checks that a slugged heading, a repeated heading and a {#label} all resolve."""
        files = {
            "docs/README.md": "[a](guides/A.md#the-first-step) [b](guides/A.md#the-first-step-1) "
                              "[c](guides/A.md#module-a) [d](#index)\n\n## Index\n",
            "docs/guides/A.md": "# A {#module-a}\n\n## The `first` step\n\n## The first step\n",
        }
        self.assertEqual(list(rule_link_anchors(_repository(files))), [])

    def test_reports_an_anchor_no_heading_has(self) -> None:
        """Checks that an anchor of a renamed section is a finding on the line of the link."""
        files = {
            "docs/README.md": "# Index\n\n[a](guides/A.md#the-old-name)\n",
            "docs/guides/A.md": "# A\n\n## The new name\n",
        }
        issues = list(rule_link_anchors(_repository(files)))
        self.assertEqual([issue.line for issue in issues], [3])

    def test_ignores_a_heading_inside_a_fence(self) -> None:
        """Checks that a heading line inside a fenced block gives no anchor."""
        files = {
            "docs/README.md": "# Index\n\n[a](guides/A.md#fenced)\n",
            "docs/guides/A.md": "# A\n\n```\n## Fenced\n```\n",
        }
        self.assertEqual(len(list(rule_link_anchors(_repository(files)))), 1)


class StatusLinesTest(unittest.TestCase):
    """Covers the four status forms in design documents and work folders."""
    K_TEST_SIZE = "small"

    def test_accepts_the_four_forms(self) -> None:
        """Reports nothing for each form, with and without text after the date."""
        files = {
            "docs/design/A.md": "# A\n\n**Status:** Open — phase 2 of 4\n",
            "docs/design/B.md": "# B\n\n**Status:** Shipped — 2026-10-01\n",
            "docs/design/C.md": "# C\n\n**Status:** Shipped — 2026-10-01. Measured on one card.\n",
            "docs/design/D.md": "# D\n\n**Status:** Superseded — 2026-10-01 by [E](E.md)\n",
            "docs/design/E.md": (
                "# E\n\n**Status:** Withdrawn — 2026-10-01, the owner chose another engine\n"
            ),
            "docs/agent/2026_10_01_WORK/SPEC.md": "# W\n\n**Status:** Open — task 1 of 3\n",
        }
        self.assertEqual(list(rule_status_lines(_repository(files))), [])

    def test_reports_prose_and_an_undated_closed_status(self) -> None:
        """Reports a prose status, a Shipped line with no date and an impossible date."""
        files = {
            "docs/design/A.md": "# A\n\n**Status:** in progress; phases 1 to 3 landed\n",
            "docs/design/B.md": "# B\n\n**Status:** Shipped\n",
            "docs/design/C.md": "# C\n\n**Status:** Shipped — 2026-13-40\n",
        }
        self.assertEqual(_names(rule_status_lines(_repository(files))),
                         ["design/A.md", "design/B.md", "design/C.md"])

    def test_reads_the_first_file_of_a_work_folder(self) -> None:
        """Reads SPEC.md first, and a REPORT.md only when the folder holds nothing else."""
        files = {
            "docs/agent/2026_10_01_A/SPEC.md": "# A\n\n**Status:** Open — writing\n",
            "docs/agent/2026_10_01_A/REPORT.md": "# A report\n\nNo status here.\n",
            "docs/agent/2026_10_01_B/REPORT.md": "# B report\n\nNo status here.\n",
        }
        self.assertEqual(_names(rule_status_lines(_repository(files))),
                         ["agent/2026_10_01_B/REPORT.md"])


class ClosedWorkTest(unittest.TestCase):
    """Covers the clock on closed work, which reads the date in the status line."""
    K_TEST_SIZE = "small"

    def test_counts_from_the_status_date(self) -> None:
        """Lists a withdrawn document and a folder dated 91 days ago, not ones dated 89 days ago."""
        files = {
            "docs/design/OLD.md": f"# O\n\n**Status:** Withdrawn — {_days_ago(91)}, dropped\n",
            "docs/design/NEW.md": f"# N\n\n**Status:** Withdrawn — {_days_ago(89)}, dropped\n",
            "docs/design/OPEN.md": "# P\n\n**Status:** Open — phase 1 of 2\n",
            "docs/agent/2026_01_01_OLD/SPEC.md": (
                f"# W\n\n**Status:** Superseded — {_days_ago(91)} by X\n"
            ),
        }
        self.assertEqual(sorted(_names(rule_closed_work(_repository(files)))),
                         ["agent/2026_01_01_OLD", "design/OLD.md"])

    def test_a_shipped_design_is_kept(self) -> None:
        """Lists nothing for a shipped design, the record of its decisions, however long ago it shipped."""
        files = {"docs/design/SHIPPED.md": f"# S\n\n**Status:** Shipped — {_days_ago(400)}\n"}
        self.assertEqual(_names(rule_closed_work(_repository(files))), [])

    def test_a_shipped_work_folder_is_listed(self) -> None:
        """Lists a work folder shipped beyond its grace."""
        files = {"docs/agent/2026_01_01_DONE/SPEC.md": f"# W\n\n**Status:** Shipped — {_days_ago(91)}\n"}
        self.assertEqual(_names(rule_closed_work(_repository(files))), ["agent/2026_01_01_DONE"])

    def test_a_grace_of_zero_lists_work_closed_today(self) -> None:
        """Lists a work folder closed today when the repository deletes at close."""
        files = {
            "docs/agent/2026_10_21_NOW/SPEC.md": f"# W\n\n**Status:** Shipped — {_days_ago(0)}\n",
        }
        repository = dataclasses.replace(_repository(files), closed_grace_days=0)
        self.assertEqual(_names(rule_closed_work(repository)), ["agent/2026_10_21_NOW"])
        self.assertEqual(
            _names(rule_closed_work(dataclasses.replace(repository, closed_grace_days=90))), [],
        )


class ForbiddenKindsTest(unittest.TestCase):
    """Covers the files that repeat the backlog or the changelog."""
    K_TEST_SIZE = "small"

    def test_reports_each_kind_outside_agent_and_archive(self) -> None:
        """Reports roadmap, journal, handover, milestone, priority and to-do files."""
        names = ["design/ROADMAP.md", "design/X/ROADMAP_JOURNAL.md", "design/X/HANDOVER.md",
                 "design/MILESTONES.md", "design/PRIORITY_ORDER.md", "guides/TODO.md"]
        files = {f"docs/{name}": "# T\n" for name in names}
        files.update({
            "docs/agent/2026_10_01_A/REPORT.md": "# R\n",
            "docs/archive/design/ROADMAP.md": "# R\n",
        })
        self.assertEqual(sorted(_names(rule_forbidden_kinds(_repository(files)))),
                         sorted(names))

    def test_accepts_a_design_document(self) -> None:
        """Reports nothing for a design document and the backlog."""
        files = {"docs/design/RENDERER.md": "# R\n", "docs/design/REMAINING_WORK.md": "# W\n"}
        self.assertEqual(list(rule_forbidden_kinds(_repository(files))), [])


class DocumentNamesTest(unittest.TestCase):
    """Checks the names of live documents."""
    K_TEST_SIZE = "small"

    def test_a_lower_case_document_is_reported(self) -> None:
        self.assertEqual(_names(rule_document_names(_repository({"docs/guides/setup.md": "# S\n"}))),
                         ["guides/setup.md"])

    def test_an_upper_snake_case_document_is_clean(self) -> None:
        self.assertEqual(list(rule_document_names(_repository({"docs/guides/SET_UP.md": "# S\n"}))), [])


class WorkFoldersTest(unittest.TestCase):
    """Checks the folders under docs/agent/ and the files inside them."""
    K_TEST_SIZE = "small"

    def test_a_folder_without_a_date_is_reported(self) -> None:
        self.assertEqual(_names(rule_work_folders(_repository({"docs/agent/REVIEW/PLAN.md": "# P\n"}))),
                         ["agent/REVIEW"])

    def test_a_foreign_file_in_a_work_folder_is_reported(self) -> None:
        repository = _repository({"docs/agent/2026_11_05_REVIEW/NOTES.md": "# N\n"})
        self.assertEqual(_names(rule_work_folders(repository)), ["agent/2026_11_05_REVIEW/NOTES.md"])

    def test_a_dated_folder_with_its_plan_is_clean(self) -> None:
        self.assertEqual(list(rule_work_folders(_repository({"docs/agent/2026_11_05_REVIEW/PLAN.md": "# P\n"}))), [])


class DesignCeilingTest(unittest.TestCase):
    """Checks the length of a design document."""
    K_TEST_SIZE = "small"

    def test_a_design_document_over_the_ceiling_is_reported(self) -> None:
        repository = _repository({"docs/design/TOPIC/PHASE_1.md": "line\n" * 401})
        self.assertEqual(_names(rule_design_ceiling(repository)), ["design/TOPIC/PHASE_1.md"])

    def test_a_design_document_at_the_ceiling_is_clean(self) -> None:
        repository = _repository({"docs/design/TOPIC/PHASE_1.md": "line\n" * 400})
        self.assertEqual(list(rule_design_ceiling(repository)), [])


class ModuleReadmesTest(unittest.TestCase):
    """Checks that each module folder holds a README.md."""
    K_TEST_SIZE = "small"

    def test_a_module_without_a_readme_is_reported(self) -> None:
        repository = _repository({"modules/core/maths/vector.hpp": ""})
        self.assertEqual([issue.message for issue in rule_module_readmes(repository)], ["module lacks README.md"])

    def test_a_module_with_a_readme_is_clean(self) -> None:
        repository = _repository({"modules/core/maths/README.md": "# Maths\n"})
        self.assertEqual(list(rule_module_readmes(repository)), [])


def _write_tree(folder: str, files: dict[str, str]) -> Path:
    """Writes the given root-relative files below a folder, and returns the folder as a path."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _docs_entry_names(root: Path) -> list[str]:
    """Returns the docs-relative name of each docs entry the rule reports on a tree on disk."""
    return _names(issue for repository in collect([root])
                  for issue in rule_docs_entries(repository))


class ShellTest(unittest.TestCase):
    """Runs the shell on temporary trees: git's ignores, the clock across a commit, the CLI."""
    K_TEST_SIZE = "medium"

    def test_leaves_out_a_folder_git_ignores(self) -> None:
        """Reports a loose docs/ entry until git ignores it, then reports nothing for it."""
        with tempfile.TemporaryDirectory() as folder:
            root = _write_tree(folder, {"docs/README.md": "# R\n", "docs/api-site/index.html": ""})
            subprocess.run(["git", "init", "-q", folder], check=True)
            self.assertEqual(_docs_entry_names(root), ["api-site"])
            (root / ".gitignore").write_text("docs/api-site/\n", encoding="utf-8")
            self.assertEqual(_docs_entry_names(root), [])

    def test_a_fresh_commit_does_not_reset_the_clock(self) -> None:
        """Lists a document withdrawn 91 days ago that a commit touched today."""
        withdrawn = (datetime.date.today() - datetime.timedelta(days=91)).isoformat()
        with tempfile.TemporaryDirectory() as folder:
            root = _write_tree(folder, {
                "docs/design/OLD.md": f"# O\n\n**Status:** Withdrawn — {withdrawn}, dropped\n",
            })
            for command in (["init", "-q"], ["add", "."],
                            ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"]):
                subprocess.run(["git", "-C", folder, *command], check=True, capture_output=True)
            issues = (issue for repository in collect([root])
                      for issue in rule_closed_work(repository))
            self.assertEqual(_names(issues), ["design/OLD.md"])

    def test_command_line_names_each_missing_document(self) -> None:
        """Runs the script on one document and reports each required document it lacks."""
        with tempfile.TemporaryDirectory() as folder:
            root = _write_tree(folder, {"docs/guides/X.md": "# X\n"})
            completed = subprocess.run(
                [sys.executable, str(K_SCRIPT), str(root), "--rule", "rule_required_entries"],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(len(completed.stdout.splitlines()), len(K_REQUIRED_ENTRIES))
        self.assertIn("docs/CONTRIBUTING.md:1: rule_required_entries: required document is missing",
                      completed.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()

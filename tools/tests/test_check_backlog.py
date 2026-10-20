# test_check_backlog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of the backlog checker."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_backlog import (  # noqa: E402
    K_CHECKER,
    Repository,
    main,
    rule_design_coverage,
    rule_entry_size,
    rule_ids,
    rule_paths,
    rule_references,
    rule_scope,
    rule_shape,
)

K_BACKLOG = """# Remaining work

Next id: RW-5

## Next

- RW-2 The kernels wait on it

## Decisions waiting for the owner

### RW-1 Choose a licence for the samples
- State: decision
- Done when: the owner has named the licence

## Kernels

### RW-2 Run the benchmarks
- State: unmeasured
- Done when: each program has a recorded run
- Source: `docs/design/KERNELS.md` §4

Prose may follow the fields.

### RW-3 Bind a late operand address
- State: blocked
- Blocked by: RW-2, external: a vendor licence
- Done when: the recorder accepts an address
"""

K_ISSUES = """# Known issues

Next id: KI-3

## Storage

### KI-1 A save truncates a file above 4 GiB
- Origin: own
- Where: `src/save.py`
- Cause: a 32-bit length
- Recognise: the file ends at exactly 4 GiB
- Workaround: split the save into files below 4 GiB
- Fix: RW-2

### KI-2 The linker drops a symbol
- Origin: toolchain
- Cause: unknown
- Recognise: an undefined symbol at link time
  that only the release build shows
- Workaround: build the debug configuration
"""

K_DESIGN = "# Kernels\n\n**Status:** Open — phase 1 of 2\n"


def _write(root: Path, files: dict[str, str]) -> None:
    """Writes the given files under a root."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _repository(folder: str, files: dict[str, str], name: str = "alpha") -> Repository:
    """Returns a repository named after its folder, holding the given files beside a valid pair."""
    root = Path(folder) / name
    base = {
        "docs/design/REMAINING_WORK.md": K_BACKLOG,
        "docs/reference/KNOWN_ISSUES.md": K_ISSUES,
        "docs/design/KERNELS.md": K_DESIGN,
        "src/save.py": "",
    }
    _write(root, {**base, **files})
    return Repository(root)


def _messages(issues: Iterable[Issue]) -> list[str]:
    """Returns the message of each issue."""
    return [issue.message for issue in issues]


def _lines(issues: Iterable[Issue]) -> list[int]:
    """Returns the line of each issue."""
    return [issue.line for issue in issues]


class RegistrationTest(unittest.TestCase):
    """Checks the rule table."""

    def test_names_every_rule(self) -> None:
        """Holds the seven rules under their names."""
        self.assertEqual(
            sorted(K_CHECKER.rules),
            [
                "rule_design_coverage", "rule_entry_size", "rule_ids", "rule_paths",
                "rule_references", "rule_scope", "rule_shape",
            ],
        )


class ValidPairTest(unittest.TestCase):
    """Checks that a pair written to the format passes every rule."""

    def test_reports_nothing(self) -> None:
        """Runs every rule over the valid pair."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {})
            for rule in K_CHECKER.rules.values():
                self.assertEqual(list(rule(repository)), [], rule.__name__)


class ShapeTest(unittest.TestCase):
    """Covers titles, counters, sections, headings and fields."""

    def _shape(self, backlog: str = K_BACKLOG, issues: str = K_ISSUES) -> list[str]:
        """Returns the shape messages for the given pair."""
        with tempfile.TemporaryDirectory() as folder:
            files = {"docs/design/REMAINING_WORK.md": backlog, "docs/reference/KNOWN_ISSUES.md": issues}
            return _messages(rule_shape(_repository(folder, files)))

    def test_reports_a_missing_counter(self) -> None:
        """Reports a backlog without its Next id line."""
        self.assertEqual(len(self._shape(backlog=K_BACKLOG.replace("Next id: RW-5\n", ""))), 1)

    def test_reports_a_dated_section_and_a_finished_section(self) -> None:
        """Reports a section named after a date and one holding finished work."""
        backlog = K_BACKLOG + "\n## Opened 2026-10-13\n\n## What a build has confirmed\n"
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_a_misshapen_heading_and_a_sub_heading(self) -> None:
        """Reports an entry heading without an ID and a heading below an entry."""
        backlog = K_BACKLOG + "\n### Run the rest\n- State: open\n- Done when: x\n\n#### Detail\n"
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_an_unknown_field_and_finished_work(self) -> None:
        """Reports a Done field in the field block and a done bullet in the body."""
        backlog = K_BACKLOG.replace("- Done when: the recorder", "- Done: the old part\n- Done when: the recorder")
        backlog += "\n  - Done: the first half\n"
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_missing_fields_and_a_bad_state(self) -> None:
        """Reports an entry with no Done when and a state outside the four."""
        backlog = K_BACKLOG + "\n### RW-4 Tidy the kernels\n- State: later\n"
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_states_in_the_wrong_section(self) -> None:
        """Reports a decision among the areas and an open entry among the decisions."""
        backlog = K_BACKLOG.replace("- State: decision", "- State: open").replace(
            "- State: unmeasured", "- State: decision"
        )
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_blocked_by_on_an_open_entry_and_its_absence_on_a_blocked_one(self) -> None:
        """Ties Blocked by to the blocked state both ways."""
        backlog = K_BACKLOG.replace("- State: blocked", "- State: open").replace(
            "- State: unmeasured", "- State: blocked"
        )
        self.assertEqual(len(self._shape(backlog=backlog)), 2)

    def test_reports_next_after_another_section_and_in_the_issues(self) -> None:
        """Keeps Next first in the backlog and out of the known issues."""
        backlog = K_BACKLOG.replace("## Next\n\n- RW-2 The kernels wait on it\n\n", "") + "\n## Next\n"
        issues = K_ISSUES.replace("## Storage", "## Next\n\n## Storage")
        self.assertEqual(len(self._shape(backlog=backlog, issues=issues)), 2)

    def test_reports_the_known_issue_fields_each_origin_needs(self) -> None:
        """Requires Where, Cause and Fix of an own defect and Upstream of a sibling's."""
        issues = K_ISSUES.replace("- Where: `src/save.py`\n", "").replace("- Fix: RW-2\n", "")
        issues = issues.replace("- Origin: toolchain\n- Cause: unknown\n", "- Origin: beta\n")
        self.assertEqual(len(self._shape(issues=issues)), 3)

    def test_accepts_where_on_a_backlog_entry(self) -> None:
        """Reads Where as a field of a backlog entry."""
        backlog = K_BACKLOG.replace("- Done when: each program", "- Where: `src/save.py`\n- Done when: each program")
        self.assertEqual(self._shape(backlog=backlog), [])

    def test_accepts_continuation_lines(self) -> None:
        """Reads an indented line as the rest of the field above it."""
        self.assertEqual(self._shape(), [])


class ScopeTest(unittest.TestCase):
    """Covers the scope of the known issues."""

    def _scope(self, issues: str) -> list[str]:
        """Returns the scope messages for the given known issues."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"docs/reference/KNOWN_ISSUES.md": issues})
            return _messages(rule_scope(repository))

    def test_reports_an_own_defect_with_a_fix_and_no_workaround(self) -> None:
        """Reports work that sits in the known issues: a scheduled fix and nothing a reader can do."""
        self.assertEqual(len(self._scope(K_ISSUES.replace("split the save into files below 4 GiB", "none"))), 1)

    def test_accepts_a_limitation_chosen_on_purpose_with_no_workaround(self) -> None:
        """Accepts Workaround: none beside Fix: none with its reason."""
        issues = K_ISSUES.replace("split the save into files below 4 GiB", "none").replace(
            "- Fix: RW-2", "- Fix: none, the format caps a file at 4 GiB"
        )
        self.assertEqual(self._scope(issues), [])

    def test_accepts_a_toolchain_issue_with_no_workaround(self) -> None:
        """Leaves an issue the repository cannot fix to the known issues."""
        self.assertEqual(self._scope(K_ISSUES.replace("build the debug configuration", "none")), [])


class IdsTest(unittest.TestCase):
    """Covers duplicate IDs, the counter and the prefix of each file."""

    def test_reports_a_duplicate_an_id_past_the_counter_and_a_foreign_prefix(self) -> None:
        """Reports three faults on their heading lines."""
        backlog = K_BACKLOG.replace("### RW-3", "### RW-2").replace(
            "### RW-1 Choose", "### KI-1 Choose"
        ) + "\n### RW-9 Late\n- State: open\n- Done when: x\n"
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_ids(_repository(folder, {"docs/design/REMAINING_WORK.md": backlog})))
        self.assertEqual(len(issues), 3)


class PathsTest(unittest.TestCase):
    """Covers the paths in Where and Source."""

    def test_reports_a_missing_and_a_shortened_path(self) -> None:
        """Reports a path that does not exist and one written below the module."""
        issues_text = K_ISSUES.replace("`src/save.py`", "`src/save.py`, `save.py`, `src/gone.py`")
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_paths(_repository(folder, {"docs/reference/KNOWN_ISSUES.md": issues_text})))
        self.assertEqual(len(issues), 2)


class ReferencesTest(unittest.TestCase):
    """Covers local IDs and the IDs of sibling repositories."""

    def test_reports_local_ids_with_no_entry(self) -> None:
        """Reports an ID in Next, in Blocked by and in Fix that names nothing."""
        backlog = K_BACKLOG.replace("- RW-2 The kernels", "- RW-4 The kernels").replace(
            "Blocked by: RW-2", "Blocked by: KI-7"
        )
        issues = K_ISSUES.replace("- Fix: RW-2", "- Fix: RW-4")
        files = {"docs/design/REMAINING_WORK.md": backlog, "docs/reference/KNOWN_ISSUES.md": issues}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(list(rule_references(_repository(folder, files)))), 3)

    def test_checks_a_sibling_that_is_checked_out(self) -> None:
        """Resolves a sibling's IDs, reports a missing one and counts an absent sibling."""
        backlog = K_BACKLOG.replace("Blocked by: RW-2", "Blocked by: beta RW-2, beta RW-8, gamma RW-1")
        issues = K_ISSUES.replace("- Origin: toolchain\n- Cause: unknown\n", "- Origin: beta\n- Upstream: beta KI-1\n")
        files = {"docs/design/REMAINING_WORK.md": backlog, "docs/reference/KNOWN_ISSUES.md": issues}
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {}, name="beta")
            repository = _repository(folder, files)
            self.assertEqual(_lines(rule_references(repository)), [26])
            self.assertEqual(repository.unchecked_references(), 1)


class EntrySizeTest(unittest.TestCase):
    """Covers the entry ceiling."""

    def test_reports_an_entry_above_the_ceiling(self) -> None:
        """Reports an entry of 26 lines and not one of 25."""
        body = "\n".join(f"Line {index}." for index in range(21))
        backlog = K_BACKLOG.replace("Prose may follow the fields.", body)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(list(rule_entry_size(_repository(folder, {"docs/design/REMAINING_WORK.md": backlog})))), 1)
        backlog = K_BACKLOG.replace("Prose may follow the fields.", "\n".join(body.splitlines()[:20]))
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_entry_size(_repository(folder, {"docs/design/REMAINING_WORK.md": backlog}))), [])


class DesignCoverageTest(unittest.TestCase):
    """Covers the link from every open design document to the backlog."""

    def test_reports_an_open_document_no_entry_cites(self) -> None:
        """Reports an open document and a split topic left uncited, not a shipped one."""
        files = {
            "docs/design/AUDIO.md": "# Audio\n\n**Status:** Open — phase 1 of 3\n",
            "docs/design/OLD.md": "# Old\n\n**Status:** Shipped — 2026-01-01\n",
            "docs/design/PHYSICS/README.md": "# Physics\n\n**Status:** Open — phase 2 of 9\n",
            "docs/design/RENDER/README.md": "# Render\n\n**Status:** Open — phase 1 of 2\n",
            "docs/design/RENDER/PHASE_1.md": "# Phase 1\n",
            "docs/design/REMAINING_WORK.md": K_BACKLOG.replace(
                "- Done when: the recorder", "- Source: `docs/design/RENDER/PHASE_1.md`\n- Done when: the recorder"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            names = [issue.path.name for issue in rule_design_coverage(_repository(folder, files))]
        self.assertEqual(names, ["AUDIO.md", "README.md"])


class ReportTest(unittest.TestCase):
    """Covers what --report adds."""

    def test_lists_an_entry_naming_a_sibling(self) -> None:
        """Lists an entry whose title names a sibling, not one naming it only in Blocked by."""
        backlog = K_BACKLOG.replace("Bind a late operand address", "Ask Beta for a late operand address").replace(
            "Blocked by: RW-2", "Blocked by: beta RW-2"
        )
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {}, name="beta")
            repository = _repository(folder, {"docs/design/REMAINING_WORK.md": backlog})
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([str(repository.root), "--report"])
        self.assertEqual(code, 0)
        self.assertIn("entries naming a sibling: 1", output.getvalue())
        self.assertIn("RW-3 names beta", output.getvalue())


if __name__ == "__main__":
    unittest.main()

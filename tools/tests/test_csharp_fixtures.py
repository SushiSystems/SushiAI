# test_csharp_fixtures.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests tools/testing/csharp_fixtures.py on C# source text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.csharp_fixtures import fixtures_of  # noqa: E402

K_TEST_SIZE = "small"

K_TABLE_STUB = (
    "namespace Game.Tests {\n"
    "[TestFixture, Category(\"small\")]\n"
    "public sealed class TrafficTests {\n"
    "  private sealed class Table : IMoverRank {\n"
    "    public int Rank(int id) => id;\n"
    "  }\n"
    "  [Test] public void RanksByIdentity() { Assert.AreEqual(1, new Table().Rank(1)); }\n"
    "}\n"
    "}\n"
)


class FixturesTest(unittest.TestCase):
    """Finds the fixtures and their test methods."""

    def test_a_nested_stub_class_is_not_a_fixture_and_its_host_keeps_its_test(self) -> None:
        fixtures = fixtures_of(K_TABLE_STUB)
        self.assertEqual([(fixture.name, fixture.categories, [method.name for method in fixture.methods])
                          for fixture in fixtures],
                         [("TrafficTests", frozenset({"small"}), ["RanksByIdentity"])])

    def test_a_nested_class_with_a_test_attribute_is_still_no_fixture(self) -> None:
        text = "public class Host {\n  public class Inner {\n    [Test] public void A() {}\n  }\n}\n"
        self.assertEqual([fixture.name for fixture in fixtures_of(text)], ["Host"])

    def test_an_abstract_class_and_a_class_without_a_test_are_no_fixtures(self) -> None:
        text = ("public abstract class Base {\n  [Test] public void A() {}\n}\n"
                "public class Helper {\n  public void B() {}\n}\n")
        self.assertEqual(fixtures_of(text), [])

    def test_a_category_in_a_comment_is_not_read_and_a_brace_in_a_string_does_not_end_the_body(self) -> None:
        text = "// [Category(\"small\")]\npublic class A {\n  [Test] public void T() { var s = \"}\"; }\n}\n"
        self.assertEqual([(fixture.categories, len(fixture.methods)) for fixture in fixtures_of(text)],
                         [(frozenset(), 1)])


if __name__ == "__main__":
    unittest.main()

# test_naming.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rename map builder and the tool that applies a rename map."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naming.apply_rename import (  # noqa: E402
    K_EXIT_CLEAN,
    K_EXIT_REFUSED,
    K_EXIT_UNSAFE,
    K_RENAME_MAP,
    RenameEntry,
    RenameError,
    RenameNames,
    RenameRefusedError,
    Rewrite,
    apply_rename,
    code_only,
    main,
    read_rename_map,
    rewrite,
    tokenize,
    uncommitted_paths,
    write_rename_map,
)
from naming.build_rename_map import (  # noqa: E402
    build_rename_map,
    is_pascal_case,
    list_unresolved,
    pascal_case,
    upper_snake,
)

K_HEADER = """\
namespace Demo
{
    using DType = SushiBLAS::Core::DataType;
    inline constexpr int MAX_RANK = 8;
    inline constexpr int K_LIMIT = 4;
    inline constexpr int IM2COL = 1;
    inline constexpr int MAXPOOL2D_BACKWARD = 2;
    constexpr std::string_view LINEAR_KEYS[] = {"units", "use_bias"};
    constexpr int twice(int value) { return 2 * value; }
    constexpr bool operator==(const Box& a, const Box& b) { return a.x == b.x; }
    template<typename T>
    inline constexpr bool Module = Detail::IsModule<T>::value;
    static const double tolerance = 1e-6;
    constexpr int N = 3;
    enum class Layout : std::uint8_t
    {
        NHWC, /**< Channels last. */
        NCHW,
        Packed,
    };
    enum class Kind { NUL, GEMM_BIAS_RELU };
    enum Scalar : std::size_t { BETA1, SCALAR_COUNT };
    class IShape { };
    class ILayer;
    class ILayer { };
    struct Layer { };
    class IForward;
    class Image { };
}
#define SA_CHECK(x) (x)
#define SUSHIAI_VERSION 1
#define OLD_CHECK(x) (x)
"""
K_SOURCE = """\
#include "demo/MAX_RANK.hpp"
// MAX_RANK is the limit; see @ref MAX_RANK and @c Layout::NHWC.
int f()
{
    const char* text = "MAX_RANK and Layout::NHWC";
    auto wide = SushiBLAS::Core::DataType::FLOAT64;
    if (Layout::NHWC == layout && Kind::NUL != kind) return MAX_RANK;
    return static_cast<int>(DType::FLOAT32) + SushiBLAS::Core::MAX_RANK;
}
"""
K_DOCUMENT = "`MAX_RANK` bounds a shape. `Layout::NHWC` is the default; MAX_RANK again.\n"
K_MODEL = '{"layout": "nhwc", "units": 4}\n'
K_COMMAND = 'parser.add_argument("--max-rank", type=int)\n'


def _tree(folder: str, files: dict[str, str]) -> Path:
    """Returns the root of a fixture tree holding the given files."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    return root


def _fixture(folder: str) -> Path:
    """Returns the root of the fixture every map test reads."""
    return _tree(folder, {
        "include/demo/demo.hpp": K_HEADER,
        "src/demo.cpp": K_SOURCE,
        "docs/guides/DEMO.md": K_DOCUMENT,
        "examples/model.json": K_MODEL,
        "cli/commands/run.py": K_COMMAND,
    })


def _entry(kind: str, old: str, new: str, status: str = "rename") -> RenameEntry:
    """Returns a map row with empty counts."""
    return RenameEntry(kind, old, new, "include/demo/demo.hpp", 0, 0, status, "")


def _names(*entries: RenameEntry) -> RenameNames:
    """Returns the lookups of a handful of rows."""
    return RenameNames.from_entries(entries)


K_ENTRIES = (
    _entry("constant", "MAX_RANK", "K_MAX_RANK"),
    _entry("enumerator", "Layout::NHWC", "Layout::Nhwc"),
    _entry("enumerator", "Kind::NUL", "Kind::Null"),
    _entry("sibling_enumerator", "DType::MAX_RANK", "DType::MaxRank", "sibling"),
)


class MappingRuleTest(unittest.TestCase):
    """Checks the spelling rules of SPEC 2.13."""

    def test_writes_an_enumerator_in_pascal_case(self) -> None:
        """Capitalises each underscore-separated part and lower-cases the rest."""
        self.assertEqual(pascal_case("WEIGHT"), "Weight")
        self.assertEqual(pascal_case("GEMM_BIAS_RELU"), "GemmBiasRelu")
        self.assertEqual(pascal_case("NHWC"), "Nhwc")
        self.assertEqual(pascal_case("MAXPOOL2D"), "Maxpool2d")

    def test_keeps_the_exception_recorded_by_hand(self) -> None:
        """Maps NUL to Null."""
        self.assertEqual(pascal_case("NUL"), "Null")

    def test_tells_pascal_case_from_upper_case(self) -> None:
        """Accepts Packed and X, refuses NHWC and Gemm_Bias."""
        self.assertTrue(is_pascal_case("Packed"))
        self.assertTrue(is_pascal_case("X"))
        self.assertFalse(is_pascal_case("NHWC"))
        self.assertFalse(is_pascal_case("Gemm_Bias"))

    def test_writes_a_lower_case_constant_in_upper_snake(self) -> None:
        """Splits camel case and drops a leading k."""
        self.assertEqual(upper_snake("tolerance"), "TOLERANCE")
        self.assertEqual(upper_snake("maxRank"), "MAX_RANK")
        self.assertEqual(upper_snake("kClasses"), "CLASSES")

    def test_keeps_an_upper_case_name_with_a_digit(self) -> None:
        """Adds no underscore after a digit."""
        self.assertEqual(upper_snake("IM2COL"), "IM2COL")
        self.assertEqual(upper_snake("MAXPOOL2D_BACKWARD"), "MAXPOOL2D_BACKWARD")
        self.assertEqual(upper_snake("conv2dStride"), "CONV2D_STRIDE")


class BuildRenameMapTest(unittest.TestCase):
    """Checks which names the builder lists and how it classes them."""

    @classmethod
    def setUpClass(cls) -> None:
        """Builds the map of the fixture tree once."""
        with tempfile.TemporaryDirectory() as folder:
            entries = build_rename_map(_fixture(folder))
        cls.rows = {(entry.kind, entry.old_name): entry for entry in entries}

    def _row(self, kind: str, old: str) -> RenameEntry:
        """Returns the row for a name, failing the test when it is absent."""
        self.assertIn((kind, old), self.rows)
        return self.rows[(kind, old)]

    def test_renames_an_upper_case_constant(self) -> None:
        """Prefixes K_; counts the declaration, the use and the @ref, and the documents apart."""
        row = self._row("constant", "MAX_RANK")
        self.assertEqual((row.new_name, row.status), ("K_MAX_RANK", "rename"))
        self.assertEqual(row.declaring_file, "include/demo/demo.hpp")
        self.assertEqual((row.occurrences, row.document_occurrences), (3, 2))

    def test_prefixes_a_constant_whose_name_holds_a_digit(self) -> None:
        """Writes K_ and the present spelling, as SPEC 2.13 words the rule."""
        self.assertEqual(self._row("constant", "IM2COL").new_name, "K_IM2COL")
        self.assertEqual(
            self._row("constant", "MAXPOOL2D_BACKWARD").new_name, "K_MAXPOOL2D_BACKWARD"
        )

    def test_reads_an_array_constant(self) -> None:
        """Takes the name before the array brackets."""
        self.assertEqual(self._row("constant", "LINEAR_KEYS").new_name, "K_LINEAR_KEYS")

    def test_skips_what_is_already_right_or_is_a_function(self) -> None:
        """Lists no K_ constant, no constexpr function and no constexpr operator."""
        for name in ("K_LIMIT", "twice", "operator", "value"):
            self.assertNotIn(("constant", name), self.rows)

    def test_excludes_a_trait_variable_template(self) -> None:
        """Keeps the D12 exemption as an excluded row with its reason."""
        row = self._row("constant", "Module")
        self.assertEqual((row.new_name, row.status), ("Module", "excluded"))
        self.assertIn("D12", row.reason)

    def test_leaves_a_lower_case_or_short_constant_to_a_person(self) -> None:
        """Marks a constant whose whole-word replacement is unsafe as by_hand."""
        self.assertEqual(self._row("constant", "tolerance").status, "by_hand")
        self.assertEqual(self._row("constant", "tolerance").new_name, "K_TOLERANCE")
        self.assertEqual(self._row("constant", "N").status, "by_hand")

    def test_renames_enumerators_in_their_qualified_form(self) -> None:
        """Lists Enum::NAME to Enum::Name and skips an enumerator already in PascalCase."""
        row = self._row("enumerator", "Layout::NHWC")
        self.assertEqual((row.new_name, row.status), ("Layout::Nhwc", "rename"))
        self.assertEqual((row.occurrences, row.document_occurrences), (3, 1))
        self.assertEqual(self._row("enumerator", "Kind::NUL").new_name, "Kind::Null")
        self.assertEqual(
            self._row("enumerator", "Kind::GEMM_BIAS_RELU").new_name, "Kind::GemmBiasRelu"
        )
        self.assertNotIn(("enumerator", "Layout::Packed"), self.rows)

    def test_lists_an_enum_without_a_sized_underlying_type(self) -> None:
        """Lists the unsized enum for a hand edit and leaves the sized one out."""
        row = self._row("enum_underlying", "Kind")
        self.assertEqual((row.new_name, row.status), ("Kind : std::uint8_t", "by_hand"))
        self.assertNotIn(("enum_underlying", "Layout"), self.rows)

    def test_lists_an_unscoped_enum_for_a_hand_edit(self) -> None:
        """Names the enumerators of the unscoped enum in the reason."""
        row = self._row("unscoped_enum", "Scalar")
        self.assertEqual(row.status, "by_hand")
        self.assertIn("BETA1, SCALAR_COUNT", row.reason)
        self.assertNotIn(("enumerator", "Scalar::BETA1"), self.rows)

    def test_drops_the_interface_prefix(self) -> None:
        """Renames IShape and leaves ILayer to a person because Layer exists."""
        self.assertEqual(self._row("interface", "IShape").new_name, "Shape")
        self.assertEqual(self._row("interface", "IShape").status, "rename")
        row = self._row("interface", "ILayer")
        self.assertEqual((row.new_name, row.status), ("Layer", "by_hand"))
        self.assertIn("already a token", row.reason)

    def test_lists_no_forward_declaration_and_no_word_that_starts_with_i(self) -> None:
        """Leaves out a class that is only declared ahead, and a class named Image."""
        self.assertNotIn(("interface", "IForward"), self.rows)
        self.assertNotIn(("interface", "Image"), self.rows)

    def test_excludes_a_prefixed_macro(self) -> None:
        """Keeps SA_CHECK and gives OLD_CHECK the prefix."""
        self.assertEqual(self._row("macro", "SA_CHECK").status, "excluded")
        self.assertEqual(self._row("macro", "SUSHIAI_VERSION").status, "excluded")
        self.assertEqual(self._row("macro", "OLD_CHECK").new_name, "SA_OLD_CHECK")

    def test_lists_sibling_names_apart(self) -> None:
        """Marks an enumerator reached through a SushiBLAS alias, and a sibling constant."""
        row = self._row("sibling_enumerator", "DType::FLOAT32")
        self.assertEqual(
            (row.new_name, row.status, row.occurrences), ("DType::Float32", "sibling", 1)
        )
        self.assertEqual(self._row("sibling_name", "MAX_RANK").status, "sibling")

    def test_counts_a_sibling_enumerator_under_its_namespace(self) -> None:
        """Counts the use that the rename pass skips because a sibling's namespace roots it."""
        row = self._row("sibling_enumerator", "DataType::FLOAT64")
        self.assertEqual((row.status, row.occurrences), ("sibling", 1))

    def test_excludes_the_string_that_spells_a_renamed_name(self) -> None:
        """Lists the JSON spelling of an enumerator as a string that stays."""
        row = self._row("string_spelling", "nhwc")
        self.assertEqual(
            (row.new_name, row.status, row.declaring_file),
            ("nhwc", "excluded", "examples/model.json"),
        )

    def test_excludes_the_flag_that_spells_a_renamed_name(self) -> None:
        """Lists a command line flag of the CLI as a string that stays."""
        row = self._row("string_spelling", "max_rank")
        self.assertEqual((row.status, row.declaring_file), ("excluded", "cli/commands/run.py"))

    def test_lists_an_unscoped_enum_once_per_file(self) -> None:
        """Keeps the enumerators of two enums of one name apart, each in its file's row."""
        files = {
            "src/a.cpp": "enum Scalar : std::size_t { BETA1, SCALAR_COUNT };\n",
            "src/b.cpp": "enum Scalar : std::size_t { MOMENTUM, SCALAR_COUNT };\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                (entry.declaring_file, entry.reason.rpartition(": ")[2])
                for entry in build_rename_map(_tree(folder, files))
                if entry.kind == "unscoped_enum"
            ]
        self.assertEqual(
            rows, [("src/a.cpp", "BETA1, SCALAR_COUNT"), ("src/b.cpp", "MOMENTUM, SCALAR_COUNT")]
        )

    def test_lists_a_type_that_shares_an_enumerator_name(self) -> None:
        """Excludes the class SGD, so its tokens are not unresolved uses of Kind::SGD."""
        files = {"src/a.cpp": "enum class Kind : int { SGD };\nclass SGD { };\nSGD make();\n"}
        with tempfile.TemporaryDirectory() as folder:
            entries = build_rename_map(_tree(folder, files))
            unresolved = list_unresolved(Path(folder), entries)
        rows = {(entry.kind, entry.old_name): entry for entry in entries}
        self.assertEqual(rows[("type_name", "SGD")].status, "excluded")
        self.assertEqual(rows[("enumerator", "Kind::SGD")].new_name, "Kind::Sgd")
        self.assertEqual(unresolved, [])

    def test_lists_the_tokens_no_qualifier_resolves(self) -> None:
        """Names the file and the line of a bare token that spells a renamed enumerator."""
        files = {"src/a.cpp": "enum class Kind : int { SGD };\nconst int SGD = 1;\n"}
        with tempfile.TemporaryDirectory() as folder:
            entries = build_rename_map(_tree(folder, files))
            self.assertEqual(list_unresolved(Path(folder), entries), ["src/a.cpp:2: SGD"])

    def test_round_trips_through_the_file(self) -> None:
        """Reads back the rows it wrote."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.tsv"
            entries = sorted(self.rows.values(), key=lambda entry: (entry.kind, entry.old_name))
            write_rename_map(path, entries)
            self.assertEqual(read_rename_map(path), entries)

    def test_refuses_a_map_with_another_header(self) -> None:
        """Raises for a file whose first line is not the column row."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.tsv"
            path.write_text("old\tnew\n", encoding="utf-8")
            with self.assertRaises(RenameRefusedError):
                read_rename_map(path)


class RewriteTest(unittest.TestCase):
    """Checks which tokens one pass over a text renames."""

    def test_gives_the_text_back_from_its_tokens(self) -> None:
        """Loses no character when a text is lexed."""
        self.assertEqual(
            "".join(token.text for token in tokenize(K_HEADER + K_SOURCE)), K_HEADER + K_SOURCE
        )

    def test_blanks_comments_and_literals_in_place(self) -> None:
        """Keeps the length and the lines of a text whose comments and literals are blanked."""
        blanked = code_only(K_SOURCE)
        self.assertEqual((len(blanked), blanked.count("\n")), (len(K_SOURCE), K_SOURCE.count("\n")))
        self.assertNotIn("limit", blanked)

    def test_renames_code_tokens_only(self) -> None:
        """Renames the constant and the qualified enumerators; keeps the literal and the include."""
        outcome = rewrite(K_SOURCE, _names(*K_ENTRIES))
        self.assertIn(
            "if (Layout::Nhwc == layout && Kind::Null != kind) return K_MAX_RANK;", outcome.text
        )
        self.assertIn('"MAX_RANK and Layout::NHWC"', outcome.text)
        self.assertIn('#include "demo/MAX_RANK.hpp"', outcome.text)

    def test_renames_in_a_comment_only_after_a_reference_tag(self) -> None:
        """Rewrites the names after @ref and @c and counts the plain mention it left."""
        outcome = rewrite(K_SOURCE, _names(*K_ENTRIES))
        self.assertIn(
            "// MAX_RANK is the limit; see @ref K_MAX_RANK and @c Layout::Nhwc.", outcome.text
        )
        self.assertEqual(outcome.comment_mentions, 1)
        self.assertEqual(outcome.counts, {"MAX_RANK": 2, "Layout::NHWC": 2, "Kind::NUL": 1})

    def test_renames_after_the_see_tag(self) -> None:
        """Rewrites the name that follows @see."""
        outcome = rewrite("/** @see MAX_RANK */\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "/** @see K_MAX_RANK */\n")

    def test_leaves_a_sibling_name_qualified_from_the_global_scope(self) -> None:
        """Keeps a name whose chain opens with `::` after an operator or a keyword."""
        text = (
            "auto r = ::SushiBLAS::Core::MAX_RANK;\n"
            "int f() { return ::SushiBLAS::Core::MAX_RANK; }\n"
        )
        self.assertEqual(rewrite(text, _names(*K_ENTRIES)).text, text)

    def test_renames_an_enumerator_an_initialiser_uses(self) -> None:
        """Resolves a bare name anywhere in an enum body against that enum's own enumerators."""
        names = _names(*K_ENTRIES, _entry("enumerator", "Kind::INT", "Kind::Int"))
        outcome = rewrite("enum class Kind { NUL, INT = NUL + 1 };\n", names)
        self.assertEqual(outcome.text, "enum class Kind { Null, Int = Null + 1 };\n")
        self.assertEqual(outcome.unresolved, [])

    def test_renames_a_constant_an_initialiser_uses_and_no_declared_enumerator(self) -> None:
        """Renames a constant in an initialiser and keeps an enumerator of the same spelling."""
        text = "enum class Other { FIRST = MAX_RANK };\nenum Scalar { MAX_RANK, LAST };\n"
        outcome = rewrite(text, _names(*K_ENTRIES))
        self.assertEqual(
            outcome.text,
            "enum class Other { FIRST = K_MAX_RANK };\nenum Scalar { MAX_RANK, LAST };\n",
        )

    def test_lists_an_enumerator_reached_through_an_alias(self) -> None:
        """Leaves K::NUL and decltype(k)::NUL in place and reports each with its line."""
        text = "using K = Kind;\nauto a = K::NUL;\nauto b = decltype(a)::NUL;\n"
        outcome = rewrite(text, _names(*K_ENTRIES))
        self.assertEqual(outcome.text, text)
        self.assertEqual(outcome.unresolved, ["2: K::NUL", "3: NUL"])

    def test_does_not_list_a_name_the_map_explains(self) -> None:
        """Reports nothing for another enum's enumerator or for an old name of another row."""
        names = _names(*K_ENTRIES, _entry("type_name", "NUL", "NUL", status="excluded"))
        text = "enum class Other { NHWC };\nauto a = DType::NHWC;\nNUL b;\n"
        self.assertEqual(rewrite(text, names).unresolved, [])

    def test_applies_the_kinds_asked_for_and_protects_with_the_rest(self) -> None:
        """Renames the enumerators alone and still keeps a name under an enum of the map."""
        names = RenameNames.from_entries(K_ENTRIES, kinds=("enumerator",))
        text = "int a = MAX_RANK + int(Layout::NHWC) + int(DType::MAX_RANK);\n"
        self.assertEqual(
            rewrite(text, names).text,
            "int a = MAX_RANK + int(Layout::Nhwc) + int(DType::MAX_RANK);\n",
        )

    def test_leaves_a_name_a_sibling_qualifies(self) -> None:
        """Keeps a name under a sibling namespace and under an enum the map does not rename."""
        text = (
            "int a = SushiBLAS::Core::MAX_RANK + sb::MAX_RANK"
            " + static_cast<int>(DType::MAX_RANK);\n"
        )
        self.assertEqual(rewrite(text, _names(*K_ENTRIES)).text, text)

    def test_renames_a_name_under_a_namespace_of_its_own(self) -> None:
        """Renames a constant qualified by a namespace that is no enum."""
        outcome = rewrite("int a = Demo::MAX_RANK;\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "int a = Demo::K_MAX_RANK;\n")

    def test_renames_the_declaration_inside_the_enum_body(self) -> None:
        """Renames the bare enumerator where its enum declares it, and no other enum's."""
        text = (
            "enum class Layout : std::uint8_t\n{\n    NHWC = 1, /**< NHWC. */\n    NCHW,\n};\n"
            "enum class Other { NHWC };\n"
        )
        outcome = rewrite(text, _names(*K_ENTRIES))
        self.assertIn("    Nhwc = 1, /**< NHWC. */\n    NCHW,", outcome.text)
        self.assertIn("enum class Other { NHWC };", outcome.text)

    def test_renames_inside_a_macro_body(self) -> None:
        """Renames a constant a #define uses."""
        outcome = rewrite("#define SA_RANK_OK(r) ((r) <= MAX_RANK)\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "#define SA_RANK_OK(r) ((r) <= K_MAX_RANK)\n")

    def test_applies_only_rows_marked_rename(self) -> None:
        """Leaves a by_hand row's name in place."""
        names = _names(_entry("constant", "MAX_RANK", "K_MAX_RANK", status="by_hand"))
        self.assertEqual(rewrite("int a = MAX_RANK;\n", names).text, "int a = MAX_RANK;\n")


class ApplyRenameTest(unittest.TestCase):
    """Checks the tree-wide run, its dry run and its refusals."""

    def test_dry_run_counts_and_writes_nothing(self) -> None:
        """Returns the per-file counts and leaves every file as it was."""
        with tempfile.TemporaryDirectory() as folder:
            root = _fixture(folder)
            changed = apply_rename(
                root, list(K_ENTRIES), dry_run=True, list_uncommitted=lambda _: ["src/x.cpp"]
            )
            self.assertEqual({name: sum(o.counts.values()) for name, o in changed.items()},
                             {"include/demo/demo.hpp": 3, "src/demo.cpp": 5})
            self.assertEqual((root / "src/demo.cpp").read_text(encoding="utf-8"), K_SOURCE)

    def test_writes_the_tree_and_keeps_line_endings(self) -> None:
        """Renames on disk and leaves CRLF line endings as they were."""
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": "int a = MAX_RANK;\r\nint b = 0;\r\n"})
            apply_rename(root, list(K_ENTRIES), dry_run=False, list_uncommitted=lambda _: [])
            self.assertEqual(
                (root / "src/a.cpp").read_bytes(), b"int a = K_MAX_RANK;\r\nint b = 0;\r\n"
            )

    def test_rewrites_the_apps_and_the_examples(self) -> None:
        """Renames under apps/ and examples/ as under src/."""
        files = {"apps/a.cpp": "int a = MAX_RANK;\n", "examples/b.cpp": "int b = MAX_RANK;\n"}
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, files)
            changed = apply_rename(
                root, list(K_ENTRIES), dry_run=False, list_uncommitted=lambda _: []
            )
            self.assertEqual(sorted(changed), ["apps/a.cpp", "examples/b.cpp"])
            self.assertEqual(
                (root / "examples/b.cpp").read_text(encoding="utf-8"), "int b = K_MAX_RANK;\n"
            )

    def test_applies_one_kind_of_the_map(self) -> None:
        """Renames the constants and leaves the enumerators for a run of their own."""
        text = "int a = MAX_RANK + int(Layout::NHWC);\n"
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": text})
            apply_rename(
                root,
                list(K_ENTRIES),
                dry_run=False,
                list_uncommitted=lambda _: [],
                kinds=("constant",),
            )
            self.assertEqual(
                (root / "src/a.cpp").read_text(encoding="utf-8"),
                "int a = K_MAX_RANK + int(Layout::NHWC);\n",
            )

    def test_refuses_a_rewrite_that_changes_a_literal(self) -> None:
        """Raises when the literals of a file differ after the rewrite."""

        def careless(text: str, _: RenameNames) -> Rewrite:
            return Rewrite(text.replace('"MAX_RANK"', '"K_MAX_RANK"'), Counter({"MAX_RANK": 1}))

        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": 'const char* name = "MAX_RANK";\n'})
            with mock.patch("naming.apply_rename.rewrite", careless):
                with self.assertRaises(RenameError) as raised:
                    apply_rename(root, list(K_ENTRIES), dry_run=True)
            self.assertIn("src/a.cpp: a string literal would change", str(raised.exception))

    def test_refuses_an_old_enumerator_that_survives(self) -> None:
        """Raises, names the token and writes nothing when no qualifier resolves an old name."""
        text = "using K = Kind;\nauto a = K::NUL + int(Kind::NUL);\n"
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": text})
            with self.assertRaises(RenameError) as raised:
                apply_rename(root, list(K_ENTRIES), dry_run=False, list_uncommitted=lambda _: [])
            self.assertNotIsInstance(raised.exception, RenameRefusedError)
            self.assertIn("src/a.cpp:2: K::NUL", str(raised.exception))
            self.assertEqual((root / "src/a.cpp").read_text(encoding="utf-8"), text)

    def test_refuses_a_tree_with_uncommitted_work(self) -> None:
        """Raises before writing when the freeze does not hold."""
        with tempfile.TemporaryDirectory() as folder:
            root = _fixture(folder)
            with self.assertRaises(RenameRefusedError):
                apply_rename(
                    root,
                    list(K_ENTRIES),
                    dry_run=False,
                    list_uncommitted=lambda _: ["src/demo.cpp"],
                )
            self.assertEqual((root / "src/demo.cpp").read_text(encoding="utf-8"), K_SOURCE)

    def test_refuses_a_new_name_that_exists(self) -> None:
        """Raises when the tree already holds the new spelling as a token."""
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": "int K_MAX_RANK = 1;\nint a = MAX_RANK;\n"})
            with self.assertRaises(RenameError) as raised:
                apply_rename(root, list(K_ENTRIES), dry_run=True)
            self.assertNotIsInstance(raised.exception, RenameRefusedError)
            self.assertIn("K_MAX_RANK exists", str(raised.exception))

    def test_command_line_reports_a_dry_run(self) -> None:
        """Prints one line per file and the total, and returns 0."""
        with tempfile.TemporaryDirectory() as folder:
            root = _fixture(folder)
            (root / K_RENAME_MAP).parent.mkdir(parents=True)
            write_rename_map(root / K_RENAME_MAP, list(K_ENTRIES))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main([folder, "--dry-run"])
            self.assertEqual(code, K_EXIT_CLEAN)
            self.assertIn("src/demo.cpp: 5, 1 comment mentions left", printed.getvalue())
            self.assertIn("would rename 8 tokens in 2 files", printed.getvalue())

    def test_command_line_applies_one_kind(self) -> None:
        """Counts the enumerator tokens alone under --kind enumerator."""
        with tempfile.TemporaryDirectory() as folder:
            root = _fixture(folder)
            (root / K_RENAME_MAP).parent.mkdir(parents=True)
            write_rename_map(root / K_RENAME_MAP, list(K_ENTRIES))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main([folder, "--dry-run", "--kind", "enumerator"])
            self.assertEqual(code, K_EXIT_CLEAN)
            self.assertIn("would rename 5 tokens in 2 files", printed.getvalue())

    def test_command_line_exit_codes_of_a_refusal(self) -> None:
        """Returns 2 without a map and 1 for a name that is taken."""
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": "int K_MAX_RANK = MAX_RANK;\n"})
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([folder, "--dry-run"]), K_EXIT_REFUSED)
                (root / K_RENAME_MAP).parent.mkdir(parents=True)
                write_rename_map(root / K_RENAME_MAP, list(K_ENTRIES))
                self.assertEqual(main([folder, "--dry-run"]), K_EXIT_UNSAFE)

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_reads_the_freeze_from_git(self) -> None:
        """Lists an untracked file under a code root by its path, whatever the user's git shows."""
        with tempfile.TemporaryDirectory() as folder:
            root = _tree(folder, {"src/a.cpp": "int a;\n", "docs/NOTE.md": "note\n"})
            subprocess.run(
                ["git", "-C", folder, "init", "--quiet"], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", folder, "config", "status.showUntrackedFiles", "no"],
                check=True,
                capture_output=True,
            )
            self.assertEqual(uncommitted_paths(root), ["src/a.cpp"])


if __name__ == "__main__":
    unittest.main()

# test_naming.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rename map builder and the tool that applies a rename map.

The rules run on repositories held in memory; the disk reader, the writer and git each keep a
medium case.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naming.apply_rename import (  # noqa: E402
    K_EXIT_CLEAN,
    K_EXIT_REFUSED,
    K_EXIT_UNSAFE,
    K_RENAME_MAP,
    DiskRepository,
    MemoryRepository,
    RenameEntry,
    RenameError,
    RenameNames,
    RenameRefusedError,
    apply_rename,
    check_literals_unchanged,
    code_only,
    format_rename_map,
    load_rename_map,
    main,
    parse_rename_map,
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
K_FIXTURE_FILES = {
    "include/demo/demo.hpp": K_HEADER,
    "src/demo.cpp": K_SOURCE,
    "docs/guides/DEMO.md": K_DOCUMENT,
    "examples/model.json": K_MODEL,
    "cli/commands/run.py": K_COMMAND,
}
K_ROOT = Path("repo")


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


def _repository(files: dict[str, str]) -> MemoryRepository:
    """Returns a repository held in memory."""
    return MemoryRepository(files)


def _with_map(files: dict[str, str]) -> MemoryRepository:
    """Returns a repository held in memory that carries the rename map of K_ENTRIES."""
    return MemoryRepository({**files, K_RENAME_MAP.as_posix(): format_rename_map(K_ENTRIES)})


def _apply(files: dict[str, str], **keywords):
    """Returns the repository and the result of a dry run or a real run of K_ENTRIES over files."""
    repository = _repository(files)
    changed = apply_rename(
        K_ROOT,
        list(K_ENTRIES),
        dry_run=keywords.pop("dry_run", False),
        list_uncommitted=lambda _: [],
        files=repository,
        **keywords,
    )
    return repository, changed


def _run_main(repository: MemoryRepository, *arguments: str) -> tuple[int, str]:
    """Returns the exit code and the printed text of one command line over a repository in memory."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
        code = main([str(K_ROOT), *arguments], files=repository, list_uncommitted=lambda _: [])
    return code, printed.getvalue()


def _disk_tree(folder: str, files: dict[str, str]) -> Path:
    """Writes the files below a folder on disk and returns the folder."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    return root


class MappingRuleTest(unittest.TestCase):
    """Checks the spelling rules of SPEC 2.13."""

    K_TEST_SIZE = "small"

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

    K_TEST_SIZE = "small"

    @classmethod
    def setUpClass(cls) -> None:
        """Builds the map of the fixture repository once."""
        entries = build_rename_map(_repository(K_FIXTURE_FILES))
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

    def test_prefixes_a_constant_with_k_and_its_present_spelling(self) -> None:
        """Writes K_ and the present spelling, as SPEC 2.13 words the rule, for a name with a digit and an array."""
        rows = (
            ("IM2COL", "K_IM2COL"),
            ("MAXPOOL2D_BACKWARD", "K_MAXPOOL2D_BACKWARD"),
            ("LINEAR_KEYS", "K_LINEAR_KEYS"),
        )
        for old, new in rows:
            with self.subTest(constant=old):
                self.assertEqual(self._row("constant", old).new_name, new)

    def test_skips_what_is_already_right_or_is_a_function(self) -> None:
        """Lists no K_ constant, no constexpr function and no constexpr operator."""
        for name in ("K_LIMIT", "twice", "operator", "value"):
            with self.subTest(name=name):
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
        """Renames IShape to Shape."""
        self.assertEqual(self._row("interface", "IShape").new_name, "Shape")
        self.assertEqual(self._row("interface", "IShape").status, "rename")

    def test_leaves_an_interface_to_a_person_when_its_new_name_exists(self) -> None:
        """Marks ILayer by_hand because Layer exists."""
        row = self._row("interface", "ILayer")
        self.assertEqual((row.new_name, row.status), ("Layer", "by_hand"))
        self.assertIn("already a token", row.reason)

    def test_lists_no_forward_declaration(self) -> None:
        """Leaves out a class that is only declared ahead."""
        self.assertNotIn(("interface", "IForward"), self.rows)

    def test_lists_no_word_that_starts_with_i(self) -> None:
        """Leaves out a class named Image."""
        self.assertNotIn(("interface", "Image"), self.rows)

    def test_excludes_a_prefixed_macro(self) -> None:
        """Keeps SA_ and SUSHIAI_ macros as they are."""
        self.assertEqual(self._row("macro", "SA_CHECK").status, "excluded")
        self.assertEqual(self._row("macro", "SUSHIAI_VERSION").status, "excluded")

    def test_gives_an_unprefixed_macro_the_prefix(self) -> None:
        """Gives OLD_CHECK the SA_ prefix."""
        self.assertEqual(self._row("macro", "OLD_CHECK").new_name, "SA_OLD_CHECK")

    def test_lists_a_sibling_enumerator_apart(self) -> None:
        """Marks an enumerator reached through a SushiBLAS alias."""
        row = self._row("sibling_enumerator", "DType::FLOAT32")
        self.assertEqual(
            (row.new_name, row.status, row.occurrences), ("DType::Float32", "sibling", 1)
        )

    def test_lists_a_sibling_constant_apart(self) -> None:
        """Marks a constant a sibling's namespace declares."""
        self.assertEqual(self._row("sibling_name", "MAX_RANK").status, "sibling")

    def test_counts_a_sibling_enumerator_under_its_namespace(self) -> None:
        """Counts the use that the rename pass skips because a sibling's namespace roots it."""
        row = self._row("sibling_enumerator", "DataType::FLOAT64")
        self.assertEqual((row.status, row.occurrences), ("sibling", 1))

    def test_excludes_a_string_or_a_flag_that_spells_a_renamed_name(self) -> None:
        """Lists the JSON spelling of an enumerator and a command line flag of the CLI as strings that stay."""
        rows = (
            ("nhwc", "examples/model.json"),
            ("max_rank", "cli/commands/run.py"),
        )
        for key, file in rows:
            with self.subTest(spelling=key):
                row = self._row("string_spelling", key)
                self.assertEqual((row.new_name, row.status, row.declaring_file), (key, "excluded", file))

    def test_lists_an_unscoped_enum_once_per_file(self) -> None:
        """Keeps the enumerators of two enums of one name apart, each in its file's row."""
        files = {
            "src/a.cpp": "enum Scalar : std::size_t { BETA1, SCALAR_COUNT };\n",
            "src/b.cpp": "enum Scalar : std::size_t { MOMENTUM, SCALAR_COUNT };\n",
        }
        rows = [
            (entry.declaring_file, entry.reason.rpartition(": ")[2])
            for entry in build_rename_map(_repository(files))
            if entry.kind == "unscoped_enum"
        ]
        self.assertEqual(
            rows, [("src/a.cpp", "BETA1, SCALAR_COUNT"), ("src/b.cpp", "MOMENTUM, SCALAR_COUNT")]
        )

    def test_lists_a_type_that_shares_an_enumerator_name(self) -> None:
        """Excludes the class SGD, so its tokens are not unresolved uses of Kind::SGD."""
        repository = _repository(
            {"src/a.cpp": "enum class Kind : int { SGD };\nclass SGD { };\nSGD make();\n"}
        )
        entries = build_rename_map(repository)
        rows = {(entry.kind, entry.old_name): entry for entry in entries}
        self.assertEqual(rows[("type_name", "SGD")].status, "excluded")
        self.assertEqual(rows[("enumerator", "Kind::SGD")].new_name, "Kind::Sgd")
        self.assertEqual(list_unresolved(repository, entries), [])

    def test_lists_the_tokens_no_qualifier_resolves(self) -> None:
        """Names the file and the line of a bare token that spells a renamed enumerator."""
        repository = _repository({"src/a.cpp": "enum class Kind : int { SGD };\nconst int SGD = 1;\n"})
        entries = build_rename_map(repository)
        self.assertEqual(list_unresolved(repository, entries), ["src/a.cpp:2: SGD"])

    def test_round_trips_through_its_text(self) -> None:
        """Reads back the rows it formatted."""
        entries = sorted(self.rows.values(), key=lambda entry: (entry.kind, entry.old_name))
        self.assertEqual(parse_rename_map(format_rename_map(entries), "map.tsv"), entries)

    def test_refuses_a_map_with_another_header(self) -> None:
        """Raises for a text whose first line is not the column row."""
        with self.assertRaises(RenameRefusedError):
            parse_rename_map("old\tnew\n", "map.tsv")

    def test_refuses_an_absent_map(self) -> None:
        """Raises, naming the map, for a repository that holds none."""
        with self.assertRaises(RenameRefusedError) as raised:
            load_rename_map(_repository({}), K_RENAME_MAP)
        self.assertIn("no rename map at", str(raised.exception))


class RewriteTest(unittest.TestCase):
    """Checks which tokens one pass over a text renames."""

    K_TEST_SIZE = "small"

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
        """Rewrites the names after @ref, @c and @see and counts the plain mention it left."""
        outcome = rewrite(K_SOURCE, _names(*K_ENTRIES))
        self.assertIn(
            "// MAX_RANK is the limit; see @ref K_MAX_RANK and @c Layout::Nhwc.", outcome.text
        )
        self.assertEqual(outcome.comment_mentions, 1)
        self.assertEqual(outcome.counts, {"MAX_RANK": 2, "Layout::NHWC": 2, "Kind::NUL": 1})
        self.assertEqual(
            rewrite("/** @see MAX_RANK */\n", _names(*K_ENTRIES)).text, "/** @see K_MAX_RANK */\n"
        )

    def test_leaves_a_name_a_sibling_qualifies(self) -> None:
        """Keeps a name under a sibling namespace, one that opens at the global scope, and one under an enum the map does not rename."""
        rows = (
            "auto r = ::SushiBLAS::Core::MAX_RANK;\nint f() { return ::SushiBLAS::Core::MAX_RANK; }\n",
            "int a = SushiBLAS::Core::MAX_RANK + sb::MAX_RANK + static_cast<int>(DType::MAX_RANK);\n",
        )
        for text in rows:
            with self.subTest(text=text):
                self.assertEqual(rewrite(text, _names(*K_ENTRIES)).text, text)

    def test_renames_an_enumerator_an_initialiser_uses(self) -> None:
        """Resolves a bare name anywhere in an enum body against that enum's own enumerators."""
        names = _names(*K_ENTRIES, _entry("enumerator", "Kind::INT", "Kind::Int"))
        outcome = rewrite("enum class Kind { NUL, INT = NUL + 1 };\n", names)
        self.assertEqual(outcome.text, "enum class Kind { Null, Int = Null + 1 };\n")
        self.assertEqual(outcome.unresolved, [])

    def test_renames_a_constant_an_initialiser_uses(self) -> None:
        """Renames a constant in an initialiser."""
        outcome = rewrite("enum class Other { FIRST = MAX_RANK };\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "enum class Other { FIRST = K_MAX_RANK };\n")

    def test_keeps_an_enumerator_that_spells_a_constant(self) -> None:
        """Keeps an enumerator of the same spelling where its own enum declares it."""
        text = "enum Scalar { MAX_RANK, LAST };\n"
        self.assertEqual(rewrite(text, _names(*K_ENTRIES)).text, text)

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

    def test_applies_the_kinds_asked_for(self) -> None:
        """Renames the enumerators alone and leaves the constant."""
        names = RenameNames.from_entries(K_ENTRIES, kinds=("enumerator",))
        text = "int a = MAX_RANK + int(Layout::NHWC);\n"
        self.assertEqual(rewrite(text, names).text, "int a = MAX_RANK + int(Layout::Nhwc);\n")

    def test_protects_a_name_under_an_enum_of_the_map_when_one_kind_runs(self) -> None:
        """Keeps a name under an enum of the map although only the enumerators are renamed."""
        names = RenameNames.from_entries(K_ENTRIES, kinds=("enumerator",))
        text = "int a = int(DType::MAX_RANK);\n"
        self.assertEqual(rewrite(text, names).text, text)

    def test_renames_a_name_under_a_namespace_of_its_own(self) -> None:
        """Renames a constant qualified by a namespace that is no enum."""
        outcome = rewrite("int a = Demo::MAX_RANK;\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "int a = Demo::K_MAX_RANK;\n")

    def test_renames_the_declaration_inside_its_own_enum_body(self) -> None:
        """Renames the bare enumerator where its enum declares it."""
        text = "enum class Layout : std::uint8_t\n{\n    NHWC = 1, /**< NHWC. */\n    NCHW,\n};\n"
        outcome = rewrite(text, _names(*K_ENTRIES))
        self.assertIn("    Nhwc = 1, /**< NHWC. */\n    NCHW,", outcome.text)

    def test_keeps_the_declaration_inside_another_enum_body(self) -> None:
        """Keeps another enum's enumerator of the same spelling."""
        text = "enum class Other { NHWC };\n"
        self.assertEqual(rewrite(text, _names(*K_ENTRIES)).text, text)

    def test_renames_inside_a_macro_body(self) -> None:
        """Renames a constant a #define uses."""
        outcome = rewrite("#define SA_RANK_OK(r) ((r) <= MAX_RANK)\n", _names(*K_ENTRIES))
        self.assertEqual(outcome.text, "#define SA_RANK_OK(r) ((r) <= K_MAX_RANK)\n")

    def test_applies_only_rows_marked_rename(self) -> None:
        """Leaves a by_hand row's name in place."""
        names = _names(_entry("constant", "MAX_RANK", "K_MAX_RANK", status="by_hand"))
        self.assertEqual(rewrite("int a = MAX_RANK;\n", names).text, "int a = MAX_RANK;\n")


class ApplyRenameTest(unittest.TestCase):
    """Checks the tree-wide run, its dry run and its refusals, on a repository held in memory."""

    K_TEST_SIZE = "small"

    def test_dry_run_counts_and_writes_nothing(self) -> None:
        """Returns the per-file counts and leaves every file as it was."""
        repository, changed = _apply(K_FIXTURE_FILES, dry_run=True)
        self.assertEqual(
            {name: sum(outcome.counts.values()) for name, outcome in changed.items()},
            {"include/demo/demo.hpp": 3, "src/demo.cpp": 5},
        )
        self.assertEqual(repository.files, K_FIXTURE_FILES)

    def test_writes_the_renamed_text(self) -> None:
        """Replaces the text of a file that holds an old name."""
        repository, _ = _apply({"src/a.cpp": "int a = MAX_RANK;\n"})
        self.assertEqual(repository.files["src/a.cpp"], "int a = K_MAX_RANK;\n")

    def test_rewrites_the_apps_and_the_examples(self) -> None:
        """Renames under apps/ and examples/ as under src/."""
        files = {"apps/a.cpp": "int a = MAX_RANK;\n", "examples/b.cpp": "int b = MAX_RANK;\n"}
        repository, changed = _apply(files)
        self.assertEqual(sorted(changed), ["apps/a.cpp", "examples/b.cpp"])
        self.assertEqual(repository.files["examples/b.cpp"], "int b = K_MAX_RANK;\n")

    def test_applies_one_kind_of_the_map(self) -> None:
        """Renames the constants and leaves the enumerators for a run of their own."""
        text = "int a = MAX_RANK + int(Layout::NHWC);\n"
        repository, _ = _apply({"src/a.cpp": text}, kinds=("constant",))
        self.assertEqual(repository.files["src/a.cpp"], "int a = K_MAX_RANK + int(Layout::NHWC);\n")

    def test_refuses_a_rewrite_that_changes_a_literal(self) -> None:
        """Raises, naming the file, when the literals of a file differ after the rewrite."""
        with self.assertRaises(RenameError) as raised:
            check_literals_unchanged(
                "src/a.cpp", 'const char* name = "MAX_RANK";\n', 'const char* name = "K_MAX_RANK";\n'
            )
        self.assertIn("src/a.cpp: a string literal would change", str(raised.exception))

    def test_accepts_a_rewrite_that_keeps_its_literals(self) -> None:
        """Raises nothing when only code tokens differ."""
        check_literals_unchanged(
            "src/a.cpp", 'int a = MAX_RANK; const char* n = "x";\n', 'int a = K_MAX_RANK; const char* n = "x";\n'
        )

    def test_refuses_an_old_enumerator_that_survives(self) -> None:
        """Raises, names the token and writes nothing when no qualifier resolves an old name."""
        text = "using K = Kind;\nauto a = K::NUL + int(Kind::NUL);\n"
        repository = _repository({"src/a.cpp": text})
        with self.assertRaises(RenameError) as raised:
            apply_rename(
                K_ROOT, list(K_ENTRIES), dry_run=False, list_uncommitted=lambda _: [], files=repository
            )
        self.assertNotIsInstance(raised.exception, RenameRefusedError)
        self.assertIn("src/a.cpp:2: K::NUL", str(raised.exception))
        self.assertEqual(repository.files["src/a.cpp"], text)

    def test_refuses_a_tree_with_uncommitted_work(self) -> None:
        """Raises before writing when the freeze does not hold."""
        repository = _repository(K_FIXTURE_FILES)
        with self.assertRaises(RenameRefusedError):
            apply_rename(
                K_ROOT,
                list(K_ENTRIES),
                dry_run=False,
                list_uncommitted=lambda _: ["src/demo.cpp"],
                files=repository,
            )
        self.assertEqual(repository.files, K_FIXTURE_FILES)

    def test_refuses_a_new_name_that_exists(self) -> None:
        """Raises when the tree already holds the new spelling as a token."""
        repository = _repository({"src/a.cpp": "int K_MAX_RANK = 1;\nint a = MAX_RANK;\n"})
        with self.assertRaises(RenameError) as raised:
            apply_rename(K_ROOT, list(K_ENTRIES), dry_run=True, files=repository)
        self.assertNotIsInstance(raised.exception, RenameRefusedError)
        self.assertIn("K_MAX_RANK exists", str(raised.exception))


class CommandLineTest(unittest.TestCase):
    """Checks what the command line prints and returns, on a repository held in memory."""

    K_TEST_SIZE = "small"

    def test_command_line_reports_a_dry_run(self) -> None:
        """Prints one line per file and the total, and returns 0."""
        code, printed = _run_main(_with_map(K_FIXTURE_FILES), "--dry-run")
        self.assertEqual(code, K_EXIT_CLEAN)
        self.assertIn("src/demo.cpp: 5, 1 comment mentions left", printed)
        self.assertIn("would rename 8 tokens in 2 files", printed)

    def test_command_line_applies_one_kind(self) -> None:
        """Counts the enumerator tokens alone under --kind enumerator."""
        code, printed = _run_main(_with_map(K_FIXTURE_FILES), "--dry-run", "--kind", "enumerator")
        self.assertEqual(code, K_EXIT_CLEAN)
        self.assertIn("would rename 5 tokens in 2 files", printed)

    def test_command_line_refuses_a_repository_without_a_map(self) -> None:
        """Returns 2 when the repository holds no rename map."""
        code, _ = _run_main(_repository({"src/a.cpp": "int K_MAX_RANK = MAX_RANK;\n"}), "--dry-run")
        self.assertEqual(code, K_EXIT_REFUSED)

    def test_command_line_refuses_a_name_that_is_taken(self) -> None:
        """Returns 1 for a new name the tree already uses."""
        repository = _with_map({"src/a.cpp": "int K_MAX_RANK = MAX_RANK;\n"})
        code, _ = _run_main(repository, "--dry-run")
        self.assertEqual(code, K_EXIT_UNSAFE)


class DiskRepositoryTest(unittest.TestCase):
    """Checks the tools over files on disk: the walk, the line endings, the map file and git."""

    K_TEST_SIZE = "medium"

    def test_lists_the_files_as_the_repository_in_memory_does(self) -> None:
        """Gives the same names and texts for the code roots, the Markdown and the JSON."""
        files = {**K_FIXTURE_FILES, "build/skip.cpp": "int x;\n", "src/note.txt": "text\n"}
        with tempfile.TemporaryDirectory() as folder:
            disk = DiskRepository(_disk_tree(folder, files))
            memory = MemoryRepository(files)
            for roots, suffixes in ((("include", "src"), (".hpp", ".cpp")), (("docs",), (".md",)), (("examples",), (".json",))):
                with self.subTest(roots=roots):
                    self.assertEqual(disk.texts(roots, suffixes), memory.texts(roots, suffixes))

    def test_writes_the_tree_and_keeps_line_endings(self) -> None:
        """Renames on disk and leaves CRLF line endings as they were."""
        with tempfile.TemporaryDirectory() as folder:
            root = _disk_tree(folder, {"src/a.cpp": "int a = MAX_RANK;\r\nint b = 0;\r\n"})
            apply_rename(root, list(K_ENTRIES), dry_run=False, list_uncommitted=lambda _: [])
            self.assertEqual(
                (root / "src/a.cpp").read_bytes(), b"int a = K_MAX_RANK;\r\nint b = 0;\r\n"
            )

    def test_command_line_writes_the_renamed_files(self) -> None:
        """Reads the map from the repository folder and renames the files below it."""
        with tempfile.TemporaryDirectory() as folder:
            root = _disk_tree(folder, {"src/a.cpp": "int a = MAX_RANK;\n"})
            (root / K_RENAME_MAP).parent.mkdir(parents=True)
            write_rename_map(root / K_RENAME_MAP, list(K_ENTRIES))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([folder], list_uncommitted=lambda _: [])
            self.assertEqual(code, K_EXIT_CLEAN)
            self.assertEqual((root / "src/a.cpp").read_text(encoding="utf-8"), "int a = K_MAX_RANK;\n")

    def test_round_trips_through_the_file(self) -> None:
        """Reads back from disk the rows it wrote."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.tsv"
            write_rename_map(path, list(K_ENTRIES))
            self.assertEqual(load_rename_map(DiskRepository(Path(folder)), "map.tsv"), list(K_ENTRIES))

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_reads_the_freeze_from_git(self) -> None:
        """Lists an untracked file under a code root by its path, whatever the user's git shows."""
        with tempfile.TemporaryDirectory() as folder:
            root = _disk_tree(folder, {"src/a.cpp": "int a;\n", "docs/NOTE.md": "note\n"})
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

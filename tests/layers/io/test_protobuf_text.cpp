/**************************************************************************/
/* test_protobuf_text.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_protobuf_text.cpp
 * @brief Tests the protobuf text parser: the tree of a prototxt, its scalars and its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/protobuf_text.hpp>

using namespace SushiAI;
using namespace std::string_view_literals;

namespace
{
    /** @brief Names the document in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.prototxt";

    /** @brief Holds a LeNet-style network: input, convolution, pooling, ReLU and softmax. */
    constexpr std::string_view K_LENET = R"(name: "LeNet"
# The input layer.
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 28 dim: 28 } }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler { type: "xavier" }
    bias_filler { type: "constant" }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param { pool: MAX kernel_size: 2 stride: 2 }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
  include { phase: TEST }
}
)"sv;

    /**
     * @brief Appends the tree under @p field to @p out as `name=value` or `name{...}`, with
     *        the fields of a message joined by `;`; a string is written in double quotes.
     */
    void dump(const IO::TextField& field, std::string& out)
    {
        bool first = true;
        for (const IO::TextField& child : field.fields)
        {
            if (!first)
                out += ';';
            first = false;
            out += child.name;
            if (child.kind == IO::TextKind::Message)
            {
                out += '{';
                dump(child, out);
                out += '}';
            }
            else if (child.kind == IO::TextKind::String)
            {
                out += "=\"" + child.text + "\"";
            }
            else
            {
                out += "=" + child.text;
            }
        }
    }

    /** @brief Returns the tree of @p root written by dump. */
    std::string dumped(const IO::TextField& root)
    {
        std::string out;
        dump(root, out);
        return out;
    }

    /** @brief Returns the depth of the chain of first fields under @p field. */
    std::size_t chain_depth(const IO::TextField& field)
    {
        std::size_t depth = 0;
        const IO::TextField* at = &field;
        while (!at->fields.empty())
        {
            at = &at->fields.front();
            ++depth;
        }
        return depth;
    }

    /** @brief Returns @p count copies of "a{", the opening of that many nested messages. */
    std::string openings(std::size_t count)
    {
        std::string out;
        for (std::size_t i = 0; i < count; ++i)
            out += "a{";
        return out;
    }

    /** @brief Returns @p count closing braces. */
    std::string closings(std::size_t count)
    {
        return std::string(count, '}');
    }

    /** @brief Names one value and the kind the parser must give it. */
    struct ScalarRow
    {
        std::string_view name;  /**< The row's name. */
        std::string_view token; /**< The value as written after `a: `. */
        IO::TextKind kind;      /**< The kind it parses to. */
    };

    /** @brief Lists numbers in every notation, signed specials and the words that are not. */
    const ScalarRow K_SCALAR_ROWS[] = {
        {"Integer", "12", IO::TextKind::Number},
        {"Negative", "-3", IO::TextKind::Number},
        {"Plus", "+7", IO::TextKind::Number},
        {"Hexadecimal", "0x1F", IO::TextKind::Number},
        {"Decimal", "1.5", IO::TextKind::Number},
        {"NoLeadingDigit", "-.5", IO::TextKind::Number},
        {"Exponent", "1e-5", IO::TextKind::Number},
        {"CapitalExponent", "2.5E+3", IO::TextKind::Number},
        {"FloatSuffix", "1f", IO::TextKind::Number},
        {"TrailingPoint", "1.", IO::TextKind::Number},
        {"SignedInfinity", "-inf", IO::TextKind::Number},
        {"BareNan", "nan", IO::TextKind::Identifier},
        {"EnumValue", "MAX", IO::TextKind::Identifier},
        {"Boolean", "true", IO::TextKind::Identifier},
    };

    /** @brief Runs one row of K_SCALAR_ROWS. */
    class Small_ProtobufTextScalar : public ::testing::TestWithParam<ScalarRow>
    {

    };

    /** @brief Returns the name of the scalar row @p info runs. */
    std::string scalar_name(const ::testing::TestParamInfo<ScalarRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Names one string as written and the bytes it decodes to. */
    struct StringRow
    {
        std::string_view name;     /**< The row's name. */
        std::string_view source;   /**< The value as written after `a: `. */
        std::string_view expected; /**< The decoded bytes. */
    };

    /** @brief Lists every escape the parser decodes, both quotes and the joining of strings. */
    const StringRow K_STRING_ROWS[] = {
        {"Newline", R"("a\nb")"sv, "a\nb"sv},
        {"SingleQuotes", R"('x"y')"sv, "x\"y"sv},
        {"EscapedQuote", R"("q\"x")"sv, "q\"x"sv},
        {"Backslash", R"("\\")"sv, "\\"sv},
        {"HexAndOctal", R"("\x41\102")"sv, "AB"sv},
        {"OneDigitHex", R"("\x4")"sv, "\x04"sv},
        {"TwoByteCodePoint", R"("\u00e9")"sv, "\xc3\xa9"sv},
        {"FourByteCodePoint", R"("\U0001F600")"sv, "\xf0\x9f\x98\x80"sv},
        {"Joined", R"("ab" 'cd')"sv, "abcd"sv},
        {"JoinedAcrossAComment", "\"ab\" # note\n \"cd\""sv, "abcd"sv},
        {"Empty", R"("")"sv, ""sv},
    };

    /** @brief Runs one row of K_STRING_ROWS. */
    class Small_ProtobufTextString : public ::testing::TestWithParam<StringRow>
    {

    };

    /** @brief Returns the name of the string row @p info runs. */
    std::string string_name(const ::testing::TestParamInfo<StringRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Names one document the parser must refuse and where and why. */
    struct RefusalRow
    {
        std::string_view name;     /**< The row's name. */
        std::string_view document; /**< The text. */
        std::string_view location; /**< The line and column the refusal must name. */
        std::string_view what;     /**< What the refusal must say. */
    };

    /** @brief Lists a refusal for each kind of stray brace, bad token and bad scalar. */
    const RefusalRow K_REFUSAL_ROWS[] = {
        {"StrayClosingBrace", "a: 1\n}\n"sv, "line 2, column 1", "stray '}'"},
        {"UnclosedBrace", "a {\n  b: 1\n"sv, "line 1, column 3", "never closed"},
        {"UnclosedInnerBrace", "a {\n b {\n  c: 1\n"sv, "line 2, column 4", "never closed"},
        {"BadValueToken", "a: 1\nb: @"sv, "line 2, column 4", "expected a value, found '@'"},
        {"BraceWhereAValueIs", "a { b: }"sv, "line 1, column 8", "expected a value, found '}'"},
        {"ListSyntax", "a: [1, 2]"sv, "line 1, column 4", "expected a value, found '['"},
        {"MissingColon", "a 1"sv, "line 1, column 3", "expected ':' or '{' after the field 'a'"},
        {"NumberAsAFieldName", "1: 2"sv, "line 1, column 1", "expected a field name, found '1'"},
        {"ControlByte", "a: 1\n\x01"sv, "line 2, column 1", "found byte 0x01"},
        {"MalformedNumber", "a: 12abc"sv, "line 1, column 4", "'12abc' is not a number"},
        {"LoneSign", "a: -"sv, "line 1, column 4", "'-' is not a number"},
        {"JunkAfterAValue", "a: 1\"x\""sv, "line 1, column 5",
         "unexpected '\"' after the value of 'a'"},
        {"UnterminatedString", "a: \"abc\n"sv, "line 1, column 4", "never closed"},
        {"StringCutByTheEnd", "a: \"abc"sv, "line 1, column 4", "never closed"},
        {"UnknownEscape", "a: \"x\\q\""sv, "line 1, column 7", "unknown escape \\q"},
        {"OctalAboveAByte", "a: \"\\777\""sv, "line 1, column 6", "above \\377"},
        {"ShortUnicodeEscape", "a: \"\\u12\""sv, "line 1, column 6",
         "needs 4 hexadecimal digits"},
        {"EscapeCutByTheEnd", "a: \"x\\"sv, "line 1, column 7", "ends inside an escape"},
        {"EndAfterTheColon", "a:"sv, "line 1, column 3", "ends after the field 'a'"},
        {"EndAfterTheName", "a"sv, "line 1, column 2", "ends after the field 'a'"},
    };

    /** @brief Runs one row of K_REFUSAL_ROWS. */
    class Small_ProtobufTextRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };

    /** @brief Returns the name of the refusal row @p info runs. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return std::string(info.param.name);
    }
} // namespace

/** @brief Tests that a LeNet-style prototxt parses to the tree its braces and fields describe. */
TEST(Small_ProtobufText, ParsesALeNetPrototxtToItsTree)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text(K_LENET, K_ORIGIN));

    const std::string expected =
        "name=\"LeNet\";"
        "layer{name=\"data\";type=\"Input\";top=\"data\";"
        "input_param{shape{dim=1;dim=1;dim=28;dim=28}}};"
        "layer{name=\"conv1\";type=\"Convolution\";bottom=\"data\";top=\"conv1\";"
        "param{lr_mult=1};param{lr_mult=2};"
        "convolution_param{num_output=20;kernel_size=5;stride=1;"
        "weight_filler{type=\"xavier\"};bias_filler{type=\"constant\"}}};"
        "layer{name=\"pool1\";type=\"Pooling\";bottom=\"conv1\";top=\"pool1\";"
        "pooling_param{pool=MAX;kernel_size=2;stride=2}};"
        "layer{name=\"relu1\";type=\"ReLU\";bottom=\"ip1\";top=\"ip1\"};"
        "layer{name=\"prob\";type=\"Softmax\";bottom=\"ip2\";top=\"prob\";include{phase=TEST}}";
    EXPECT_EQ(dumped(root), expected);
    EXPECT_EQ(root.kind, IO::TextKind::Message);
}

/** @brief Tests that fields found by name keep their repeats in document order. */
TEST(Small_ProtobufText, FindsRepeatedFieldsInDocumentOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text(K_LENET, K_ORIGIN));

    const std::vector<const IO::TextField*> layers = IO::find_fields(root, "layer");
    ASSERT_EQ(layers.size(), 5u);
    const IO::TextField* conv_params = IO::find_field(*layers[1], "convolution_param");
    ASSERT_NE(conv_params, nullptr);
    const IO::TextField* kernel = IO::find_field(*conv_params, "kernel_size");
    ASSERT_NE(kernel, nullptr);

    EXPECT_EQ(IO::find_field(*layers[0], "name")->text, "data");
    EXPECT_EQ(IO::find_field(*layers[4], "name")->text, "prob");
    EXPECT_EQ(IO::find_fields(*layers[1], "param").size(), 2u);
    EXPECT_EQ(kernel->kind, IO::TextKind::Number);
    EXPECT_EQ(kernel->text, "5");
    EXPECT_EQ(IO::find_field(root, "absent"), nullptr);
}

/** @brief Tests that each field records the line and column where its name starts. */
TEST(Small_ProtobufText, RecordsWhereEachFieldStarts)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text("a: 1\n  bb { c: 2 }"sv, K_ORIGIN));

    const IO::TextField* inner = IO::find_field(*IO::find_field(root, "bb"), "c");
    EXPECT_EQ(root.fields[0].line, 1u);
    EXPECT_EQ(root.fields[0].column, 1u);
    EXPECT_EQ(root.fields[1].line, 2u);
    EXPECT_EQ(root.fields[1].column, 3u);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->line, 2u);
    EXPECT_EQ(inner->column, 8u);
}

/** @brief Tests that a number, an identifier or a signed special is the kind its form gives. */
TEST_P(Small_ProtobufTextScalar, TakesTheKindItsFormGives)
{
    const std::string document = "a: " + std::string(GetParam().token);

    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text(document, K_ORIGIN));

    ASSERT_EQ(root.fields.size(), 1u);
    EXPECT_EQ(root.fields[0].kind, GetParam().kind);
    EXPECT_EQ(root.fields[0].text, GetParam().token);
}

INSTANTIATE_TEST_SUITE_P(Forms, Small_ProtobufTextScalar, ::testing::ValuesIn(K_SCALAR_ROWS),
                         scalar_name);

/** @brief Tests that a quoted string decodes its escapes and joins with the strings after it. */
TEST_P(Small_ProtobufTextString, DecodesToTheBytesItWrites)
{
    const std::string document = "a: " + std::string(GetParam().source);

    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text(document, K_ORIGIN));

    ASSERT_EQ(root.fields.size(), 1u);
    EXPECT_EQ(root.fields[0].kind, IO::TextKind::String);
    EXPECT_EQ(root.fields[0].text, std::string(GetParam().expected));
}

INSTANTIATE_TEST_SUITE_P(Escapes, Small_ProtobufTextString, ::testing::ValuesIn(K_STRING_ROWS),
                         string_name);

/** @brief Tests that comments, semicolons and commas between fields change nothing. */
TEST(Small_ProtobufText, IgnoresCommentsAndFieldSeparators)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::TextField root,
        IO::parse_protobuf_text("# head\na: 1; b: 2, c { } # tail\nd: x # last"sv, K_ORIGIN));

    EXPECT_EQ(dumped(root), "a=1;b=2;c{};d=x");
}

/** @brief Tests that a colon before an opening brace is optional. */
TEST(Small_ProtobufText, ReadsAMessageWithAndWithoutAColon)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField with_colon,
                            IO::parse_protobuf_text("a: { b: 1 }"sv, K_ORIGIN));
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField without,
                            IO::parse_protobuf_text("a { b: 1 }"sv, K_ORIGIN));

    EXPECT_EQ(dumped(with_colon), "a{b=1}");
    EXPECT_EQ(dumped(without), "a{b=1}");
}

/** @brief Tests that an empty document and one of only comments parse to an empty root. */
TEST(Small_ProtobufText, ParsesAnEmptyDocumentToAnEmptyRoot)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField empty, IO::parse_protobuf_text(""sv, K_ORIGIN));
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField comment,
                            IO::parse_protobuf_text("  # only a comment\n"sv, K_ORIGIN));

    EXPECT_TRUE(empty.fields.empty());
    EXPECT_TRUE(comment.fields.empty());
}

/** @brief Tests that messages nested to the limit of 100 braces parse. */
TEST(Small_ProtobufText, ParsesMessagesNestedToTheLimit)
{
    const std::string document = openings(100) + closings(100);

    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField root,
                            IO::parse_protobuf_text(document, K_ORIGIN));

    EXPECT_EQ(chain_depth(root), 100u);
}

/** @brief Tests that a message nested one brace past the limit is refused at that brace. */
TEST(Small_ProtobufText, RefusesMessagesNestedPastTheLimit)
{
    // Each "a{" takes two columns, so the 101st brace is the byte at column 2 * 100 + 2.
    const std::string document = openings(101) + closings(101);

    SA_EXPECT_FAILURE(IO::parse_protobuf_text(document, K_ORIGIN), StatusCode::InvalidFile,
                      "line 1, column 202", "deeper than 100");
}

/** @brief Tests that each malformed document is refused with the line and column it is at. */
TEST_P(Small_ProtobufTextRefusal, NamesTheLineAndColumnItFailsAt)
{
    SA_EXPECT_FAILURE(IO::parse_protobuf_text(GetParam().document, K_ORIGIN),
                      StatusCode::InvalidFile, "'fixture.prototxt'", GetParam().location,
                      GetParam().what);
}

INSTANTIATE_TEST_SUITE_P(Documents, Small_ProtobufTextRefusal, ::testing::ValuesIn(K_REFUSAL_ROWS),
                         refusal_name);

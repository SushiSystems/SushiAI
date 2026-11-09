/**************************************************************************/
/* test_darknet_config.cpp                                                */
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
 * @file test_darknet_config.cpp
 * @brief Tests the .cfg reader: each lexical form of the grammar, its refusals and a missing file.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Vision::Importers::DarknetConfig;
using Vision::Importers::DarknetOption;
using Vision::Importers::load_darknet_config;
using Vision::Importers::parse_darknet_config;

namespace
{
    /**
     * @brief Expects @p option to spell @p key and @p value on line @p line.
     * @param option The option under test.
     * @param key The key the file spells.
     * @param value The value the file spells, with its whitespace removed.
     * @param line The 1-based line of the file that holds the key.
     */
    void expect_option(const DarknetOption& option, std::string_view key, std::string_view value,
                       std::int64_t line)
    {
        EXPECT_EQ(option.key, key);
        EXPECT_EQ(option.value, value);
        EXPECT_EQ(option.line, line);
    }

    /** @brief Holds a section line the parser refuses and a text its diagnostic holds. */
    struct MalformedSection
    {
        const char* name;   /**< Names the row. */
        const char* text;   /**< Holds the text to parse. */
        const char* detail; /**< Holds a text the diagnostic names besides the line; may be empty. */
    };

    /** @brief Runs one malformed section line per row. */
    class Small_DarknetConfigMalformedSection : public ::testing::TestWithParam<MalformedSection>
    {
    };
} // namespace

/** @brief Tests that comment lines and blank lines are skipped and the lines are counted. */
TEST(Small_DarknetConfig, SkipsCommentsAndBlankLinesAndCountsThem)
{
    const std::string text =
        "# the first line is a comment\n"
        "; so is a line that starts with a semicolon\n"
        "\n"
        "[network]\n"
        "width=416\n"
        "# a comment between keys\n"
        "height=608\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));

    EXPECT_EQ(config.net.type, "network");
    EXPECT_EQ(config.net.line, 4);
    ASSERT_EQ(config.net.options.size(), 2u);
    expect_option(config.net.options[0], "width", "416", 5);
    expect_option(config.net.options[1], "height", "608", 7);
}

/** @brief Tests that whitespace around a line, a key, the sign and a value is dropped. */
TEST(Small_DarknetConfig, DropsTheWhitespaceAroundTheSignAndTheLine)
{
    const std::string text =
        "[network]\n"
        "  width = 416\n"
        "height\t=\t416\n"
        "  [convolutional]\n"
        "   size=3   \n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));

    ASSERT_EQ(config.net.options.size(), 2u);
    expect_option(config.net.options[0], "width", "416", 2);
    expect_option(config.net.options[1], "height", "416", 3);
    ASSERT_EQ(config.layers.size(), 1u);
    EXPECT_EQ(config.layers[0].type, "convolutional");
    EXPECT_EQ(config.layers[0].line, 4);
    ASSERT_EQ(config.layers[0].options.size(), 1u);
    expect_option(config.layers[0].options[0], "size", "3", 5);
}

/** @brief Tests that a list value loses the spaces between its entries. */
TEST(Small_DarknetConfig, AListValueLosesItsInnerSpaces)
{
    const std::string text =
        "[network]\n"
        "anchors = 10, 14,  23 ,27\n"
        "[route]\n"
        "layers = -1 , 61\n"
        "[yolo]\n"
        "mask=0,1 , 2\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));

    ASSERT_EQ(config.net.options.size(), 1u);
    expect_option(config.net.options[0], "anchors", "10,14,23,27", 2);
    ASSERT_EQ(config.layers.size(), 2u);
    ASSERT_EQ(config.layers[0].options.size(), 1u);
    expect_option(config.layers[0].options[0], "layers", "-1,61", 4);
    ASSERT_EQ(config.layers[1].options.size(), 1u);
    expect_option(config.layers[1].options[0], "mask", "0,1,2", 6);
}

/** @brief Tests that a key with nothing after the sign keeps an empty value. */
TEST(Small_DarknetConfig, AnEmptyValueStaysEmpty)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config,
                            parse_darknet_config("[net]\n[convolutional]\nlabel=\n", "test.cfg"));

    ASSERT_EQ(config.layers.size(), 1u);
    ASSERT_EQ(config.layers[0].options.size(), 1u);
    expect_option(config.layers[0].options[0], "label", "", 3);
}

/** @brief Tests that a value keeps a second sign. */
TEST(Small_DarknetConfig, ASecondEqualsSignStaysInTheValue)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config,
                            parse_darknet_config("[net]\nexpr=a=b\n", "test.cfg"));

    ASSERT_EQ(config.net.options.size(), 1u);
    expect_option(config.net.options[0], "expr", "a=b", 2);
}

/** @brief Tests that a CRLF text parses to the same keys, values and lines. */
TEST(Small_DarknetConfig, IgnoresCarriageReturnsAtLineEnds)
{
    const std::string text = "[net]\r\nwidth=416\r\n[convolutional]\r\nfilters=16\r\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "crlf.cfg"));

    EXPECT_EQ(config.net.type, "net");
    ASSERT_EQ(config.net.options.size(), 1u);
    expect_option(config.net.options[0], "width", "416", 2);

    ASSERT_EQ(config.layers.size(), 1u);
    EXPECT_EQ(config.layers[0].line, 3);
    ASSERT_EQ(config.layers[0].options.size(), 1u);
    expect_option(config.layers[0].options[0], "filters", "16", 4);
}

/** @brief Tests that a key may appear once in each of two sections. */
TEST(Small_DarknetConfig, AcceptsOneKeyInTwoSections)
{
    const std::string text = "[net]\nwidth=416\n[convolutional]\nwidth=8\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));

    ASSERT_EQ(config.net.options.size(), 1u);
    ASSERT_EQ(config.layers.size(), 1u);
    ASSERT_EQ(config.layers[0].options.size(), 1u);
    expect_option(config.layers[0].options[0], "width", "8", 4);
}

/** @brief Tests that a key before any section fails naming origin, line and key. */
TEST(Small_DarknetConfig, RefusesAKeyBeforeAnySection)
{
    const Result<DarknetConfig> outcome = parse_darknet_config("width=416\n[net]\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:1", "key 'width'");
}

/** @brief Tests that a line without '=' fails naming origin and line. */
TEST(Small_DarknetConfig, RefusesALineThatIsNeitherSectionNorOption)
{
    const Result<DarknetConfig> outcome = parse_darknet_config("[net]\nwidth 416\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:2");
}

/** @brief Tests that a line that starts with '=' fails naming origin and line. */
TEST(Small_DarknetConfig, RefusesAnEmptyKey)
{
    const Result<DarknetConfig> outcome = parse_darknet_config("[net]\n=416\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:2");
}

/** @brief Tests that a repeated key fails naming both of its lines. */
TEST(Small_DarknetConfig, RefusesAKeyRepeatedWithinASection)
{
    const Result<DarknetConfig> outcome =
        parse_darknet_config("[net]\nwidth=416\nwidth=608\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:3", "key 'width'",
                      "on line 2");
}

/** @brief Tests that a first section that is not [net] or [network] fails naming it. */
TEST(Small_DarknetConfig, RefusesAFirstSectionThatIsNotNetOrNetwork)
{
    const Result<DarknetConfig> outcome =
        parse_darknet_config("[convolutional]\nfilters=16\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:1", "[convolutional]");
}

/** @brief Tests that an unclosed section, an empty name and text after ']' each fail. */
TEST_P(Small_DarknetConfigMalformedSection, RefusesAMalformedSectionLine)
{
    const Result<DarknetConfig> outcome = parse_darknet_config(GetParam().text, "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg:1", GetParam().detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetConfigMalformedSection,
    ::testing::Values(MalformedSection{"AnUnclosedSection", "[net\n", "']'"},
                      MalformedSection{"AnEmptyName", "[]\n", ""},
                      MalformedSection{"TextAfterTheBracket", "[net]x\n", ""}),
    [](const ::testing::TestParamInfo<MalformedSection>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that a text of comments alone fails naming the origin. */
TEST(Small_DarknetConfig, RefusesATextWithoutASection)
{
    const Result<DarknetConfig> outcome = parse_darknet_config("# only a comment\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg", "[net]");
}

/** @brief Tests that a path that cannot be opened gives FileNotFound naming the path. */
TEST(Medium_DarknetConfig, MissingFileGivesFileNotFound)
{
    const Result<DarknetConfig> outcome = load_darknet_config("absent_darknet_model.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::FileNotFound, "absent_darknet_model.cfg");
}

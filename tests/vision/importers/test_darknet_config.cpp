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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
} // namespace

/** @brief Tests that comments, blank lines and inner whitespace parse to the stated lines. */
TEST(Small_DarknetConfig, ParsesEveryLexicalFormOfTheGrammar)
{
    const std::string text =
        "# the first line is a comment\n"
        "; so is a line that starts with a semicolon\n"
        "\n"
        "[network]\n"
        "  width = 416\n"
        "height\t=\t416\n"
        "channels=3\n"
        "# a comment between keys\n"
        "anchors = 10, 14,  23 ,27\n"
        "  [convolutional]\n"
        "filters=16\n"
        "   size=3   \n"
        "label=\n"
        "[route]\n"
        "layers = -1 , 61\n"
        "[yolo]\n"
        "mask=0,1 , 2\n"
        "expr=a=b\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));

    EXPECT_EQ(config.net.type, "network");
    EXPECT_EQ(config.net.line, 4);
    ASSERT_EQ(config.net.options.size(), 4u);
    expect_option(config.net.options[0], "width", "416", 5);
    expect_option(config.net.options[1], "height", "416", 6);
    expect_option(config.net.options[2], "channels", "3", 7);
    expect_option(config.net.options[3], "anchors", "10,14,23,27", 9);

    ASSERT_EQ(config.layers.size(), 3u);

    EXPECT_EQ(config.layers[0].type, "convolutional");
    EXPECT_EQ(config.layers[0].line, 10);
    ASSERT_EQ(config.layers[0].options.size(), 3u);
    expect_option(config.layers[0].options[0], "filters", "16", 11);
    expect_option(config.layers[0].options[1], "size", "3", 12);
    expect_option(config.layers[0].options[2], "label", "", 13);

    EXPECT_EQ(config.layers[1].type, "route");
    EXPECT_EQ(config.layers[1].line, 14);
    ASSERT_EQ(config.layers[1].options.size(), 1u);
    expect_option(config.layers[1].options[0], "layers", "-1,61", 15);

    EXPECT_EQ(config.layers[2].type, "yolo");
    EXPECT_EQ(config.layers[2].line, 16);
    ASSERT_EQ(config.layers[2].options.size(), 2u);
    expect_option(config.layers[2].options[0], "mask", "0,1,2", 17);
    expect_option(config.layers[2].options[1], "expr", "a=b", 18);
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
TEST(Small_DarknetConfig, RefusesAMalformedSectionLine)
{
    const Result<DarknetConfig> unclosed = parse_darknet_config("[net\n", "test.cfg");
    SA_EXPECT_FAILURE(unclosed, StatusCode::InvalidDescription, "test.cfg:1", "']'");

    const Result<DarknetConfig> empty_name = parse_darknet_config("[]\n", "test.cfg");
    SA_EXPECT_FAILURE(empty_name, StatusCode::InvalidDescription, "test.cfg:1");

    const Result<DarknetConfig> trailing = parse_darknet_config("[net]x\n", "test.cfg");
    SA_EXPECT_FAILURE(trailing, StatusCode::InvalidDescription, "test.cfg:1");
}

/** @brief Tests that a text of comments alone fails naming the origin. */
TEST(Small_DarknetConfig, RefusesATextWithoutASection)
{
    const Result<DarknetConfig> outcome = parse_darknet_config("# only a comment\n", "test.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "test.cfg", "[net]");
}

/** @brief Tests that a path that cannot be opened gives FileNotFound naming the path. */
TEST(Small_DarknetConfig, MissingFileGivesFileNotFound)
{
    const Result<DarknetConfig> outcome = load_darknet_config("absent_darknet_model.cfg");

    SA_EXPECT_FAILURE(outcome, StatusCode::FileNotFound, "absent_darknet_model.cfg");
}

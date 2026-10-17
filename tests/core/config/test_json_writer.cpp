/**************************************************************************/
/* test_json_writer.cpp                                                   */
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
 * @file test_json_writer.cpp
 * @brief Tests the text write_json produces and that parse_json reads it back unchanged.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::StatusCode;
using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;
using SushiAI::Config::write_json;

namespace
{
    /** @brief Holds the depth that writes a whole document on one line. */
    constexpr std::size_t K_ONE_LINE = 0;

    /**
     * @brief Reports whether @p left and @p right are the same double, bit for bit.
     *
     * The tolerance of every real compared in this file is zero, absolute and relative.
     */
    bool same_bits(double left, double right)
    {
        std::uint64_t left_bits = 0;
        std::uint64_t right_bits = 0;
        std::memcpy(&left_bits, &left, sizeof(left_bits));
        std::memcpy(&right_bits, &right, sizeof(right_bits));
        return left_bits == right_bits;
    }

    /** @brief Writes @p value on one line and parses the text back into @p read. */
    void read_back(const Value& value, Value& read)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(value, K_ONE_LINE));
        read = parse_json(text, "<written>");
    }
}

TEST(Small_JsonWriter, WritesEveryKindOnOneLineInTheOrderOfTheKeys)
{
    const Value root = parse_json(
        R"({"zebra": 1, "apple": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2],
            "g": {}, "h": [], "i": {"nested": [false]}})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(root, K_ONE_LINE));
    EXPECT_EQ(text,
              "{ \"zebra\": 1, \"apple\": 2.5, \"c\": \"text\", \"d\": true, \"e\": null, "
              "\"f\": [1, 2], \"g\": {}, \"h\": [], \"i\": { \"nested\": [false] } }\n");
}

TEST(Small_JsonWriter, ExpandsTheFirstTwoLevelsAndKeepsDeeperOnesOnOneLine)
{
    const Value root = parse_json(
        R"({"format_version": 2, "input": {"shape": [28, 28, 1]},
            "layers": [{"type": "relu"}, {"type": "concat", "from": [-1, 0]}],
            "empty": []})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(root));
    EXPECT_EQ(text,
              "{\n"
              "    \"format_version\": 2,\n"
              "    \"input\": {\n"
              "        \"shape\": [28, 28, 1]\n"
              "    },\n"
              "    \"layers\": [\n"
              "        { \"type\": \"relu\" },\n"
              "        { \"type\": \"concat\", \"from\": [-1, 0] }\n"
              "    ],\n"
              "    \"empty\": []\n"
              "}\n");
}

TEST(Small_JsonWriter, EndsEveryLineWithALineFeedAndTheTextWithOne)
{
    const Value root = parse_json(R"({"a": {"b": [1, 2]}, "c": "text"})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(root));
    ASSERT_FALSE(text.empty());
    EXPECT_EQ(text.back(), '\n');
    EXPECT_NE(text[text.size() - 2], '\n');
    EXPECT_EQ(text.find('\r'), std::string::npos);
}

TEST(Small_JsonWriter, WritesARootScalarAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(Value::integer(-3)));
    EXPECT_EQ(text, "-3\n");
}

TEST(Small_JsonWriter, SpellsNumbersInTheShortestFormThatReadsBack)
{
    const Value root = parse_json("[7, -3, 0, 0.001, 1e-3, 2.5, 1.0, 1e300]", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(root, K_ONE_LINE));
    EXPECT_EQ(text, "[7, -3, 0, 0.001, 0.001, 2.5, 1.0, 1e+300]\n");
}

TEST(Small_JsonWriter, AnIntegerReadsBackAsTheSameInteger)
{
    const std::vector<std::int64_t> integers = {
        7, -3, 0, 1, -1, 9007199254740993,
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::min(),
    };

    std::vector<Value> entries;
    for (const std::int64_t integer : integers)
        entries.push_back(Value::integer(integer));

    Value read;
    ASSERT_NO_FATAL_FAILURE(read_back(Value::array(entries), read));

    const std::vector<Value>& found = read.as_array("<written>");
    ASSERT_EQ(found.size(), integers.size());
    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        ASSERT_EQ(found[i].kind(), Kind::Int) << "entry " << i;
        EXPECT_EQ(found[i].as_int("<written>"), integers[i]) << "entry " << i;
    }
}

TEST(Small_JsonWriter, ARealReadsBackAsTheSameBits)
{
    // Each real is compared bit for bit: the tolerance is zero, absolute and relative.
    const std::vector<double> reals = {
        0.001, 1e-3, 1.0, -1.0, 0.0, -0.0, 100.0, 2.5, -2.5, 0.1 + 0.2, 1.0 / 3.0,
        1e22, 1e300, 123456789012345680000.0,
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::epsilon(),
    };

    std::vector<Value> entries;
    for (const double real : reals)
        entries.push_back(Value::real(real));

    Value read;
    ASSERT_NO_FATAL_FAILURE(read_back(Value::array(entries), read));

    const std::vector<Value>& found = read.as_array("<written>");
    ASSERT_EQ(found.size(), reals.size());
    for (std::size_t i = 0; i < reals.size(); ++i)
    {
        ASSERT_EQ(found[i].kind(), Kind::Real) << "entry " << i;
        EXPECT_TRUE(same_bits(found[i].as_real("<written>"), reals[i]))
            << "entry " << i << " was written from " << reals[i] << " and read back as "
            << found[i].as_real("<written>");
    }
}

TEST(Small_JsonWriter, EscapesQuotesBackslashesAndControlCharacters)
{
    const std::string raw = std::string("q\" b\\ s/ \b\f\n\r\t \x01\x1f \x7f") + '\0' + "end";

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(Value::string(raw)));
    EXPECT_EQ(text, "\"q\\\" b\\\\ s/ \\b\\f\\n\\r\\t \\u0001\\u001f \x7f\\u0000end\"\n");

    Value read;
    ASSERT_NO_FATAL_FAILURE(read_back(Value::string(raw), read));
    EXPECT_EQ(read.as_string("<written>"), raw);
}

TEST(Small_JsonWriter, WritesUTF8AsItsOwnBytes)
{
    // U+011F, U+65E5 and U+1F363, as two, three and four bytes.
    const std::string raw = std::string("\xC4\x9F") + " " + "\xE6\x97\xA5" + " "
                          + "\xF0\x9F\x8D\xA3";

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(Value::string(raw)));
    EXPECT_EQ(text, "\"" + raw + "\"\n");

    Value read;
    ASSERT_NO_FATAL_FAILURE(read_back(Value::string(raw), read));
    EXPECT_EQ(read.as_string("<written>"), raw);
}

TEST(Small_JsonWriter, EscapesAKeyAsItEscapesAString)
{
    const Value root = Value::object({"a \"quoted\"\tkey"}, {Value::integer(1)});

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_json(root, K_ONE_LINE));
    EXPECT_EQ(text, "{ \"a \\\"quoted\\\"\\tkey\": 1 }\n");

    Value read;
    ASSERT_NO_FATAL_FAILURE(read_back(root, read));
    EXPECT_EQ(read.keys(), root.keys());
}

TEST(Small_JsonWriter, RefusesANumberThatIsNotFinite)
{
    SA_EXPECT_FAILURE(write_json(Value::real(std::numeric_limits<double>::quiet_NaN())),
                      StatusCode::InvalidDescription, "not a number");
    SA_EXPECT_FAILURE(write_json(Value::real(std::numeric_limits<double>::infinity())),
                      StatusCode::InvalidDescription, "infinite");
    SA_EXPECT_FAILURE(write_json(Value::real(-std::numeric_limits<double>::infinity())),
                      StatusCode::InvalidDescription, "infinite");
}

TEST(Small_JsonWriter, RefusesAStringThatIsNotUTF8)
{
    const std::vector<std::pair<std::string_view, std::string>> malformed = {
        {"a byte that leads no form", "\xFF"},
        {"a continuation byte alone", "\x80"},
        {"an overlong solidus", "\xC0\xAF"},
        {"a three-byte form cut short", "\xE2\x82"},
        {"the surrogate U+D800", "\xED\xA0\x80"},
        {"U+110000", "\xF4\x90\x80\x80"},
    };

    for (const std::pair<std::string_view, std::string>& item : malformed)
    {
        SCOPED_TRACE(std::string(item.first));
        SA_EXPECT_FAILURE(write_json(Value::string(item.second)),
                          StatusCode::InvalidDescription, "UTF-8");
        SA_EXPECT_FAILURE(write_json(Value::object({item.second}, {Value::integer(1)})),
                          StatusCode::InvalidDescription, "UTF-8", "key 0");
    }
}

TEST(Small_JsonWriter, RefusesAnObjectThatStatesAKeyTwice)
{
    const Value root = Value::object({"rate", "units", "rate"},
                                     {Value::integer(1), Value::integer(2), Value::integer(3)});

    SA_EXPECT_FAILURE(write_json(root), StatusCode::InvalidDescription, "'rate'", "twice");
}

TEST(Small_JsonWriter, ARefusalNamesTheWayToTheValue)
{
    const Value row = Value::object(
        {"rate"}, {Value::real(std::numeric_limits<double>::quiet_NaN())});
    const Value root = Value::object(
        {"layers"}, {Value::array({Value::integer(1), row})});

    SA_EXPECT_FAILURE(write_json(root), StatusCode::InvalidDescription,
                      "key 'layers'", "entry 1", "key 'rate'", "not a number");
}

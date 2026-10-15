/**************************************************************************/
/* test_value.cpp                                                         */
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
 * @file test_value.cpp
 * @brief Tests how a configuration Value reads its JSON kinds and keys, and the errors it names.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/model_text/model_text.hpp>

using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    /** @brief A span over a literal allowed-key list, for reject_unknown_keys. */
    template<std::size_t N>
    SushiRuntime::span<const std::string_view> allowed(const std::string_view (&keys)[N])
    {
        return SushiRuntime::span<const std::string_view>(keys, N);
    }
} // namespace

TEST(Small_Value, ParsesEveryScalarKind)
{
    const Value root = parse_json(
        R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})", "<test>");

    ASSERT_EQ(root.kind(), Kind::Object);
    EXPECT_EQ(root.at("a", "<test>").as_int("a"), 1);
    EXPECT_DOUBLE_EQ(root.at("b", "<test>").as_real("b"), 2.5);
    EXPECT_EQ(root.at("c", "<test>").as_string("c"), "text");
    EXPECT_TRUE(root.at("d", "<test>").as_bool("d"));
    EXPECT_TRUE(root.at("e", "<test>").is_null());
    EXPECT_EQ(root.at("f", "<test>").as_array("f").size(), 2u);
}

TEST(Small_Value, KeepsTheKeyOrderTheFileSpelled)
{
    // Key order follows the file; see tests/core/config/README.md.
    const Value root = parse_json(R"({"zebra": 1, "apple": 2, "mango": 3})", "<test>");
    EXPECT_EQ(root.keys(), (std::vector<std::string>{"zebra", "apple", "mango"}));
}

TEST(Small_Value, AnIntegerReadsAsARealButARealIsNotAWholeNumber)
{
    // An integer widens to a real, never the reverse; see tests/core/config/README.md.
    const Value root = parse_json(R"({"lr": 3, "units": 2.5})", "<test>");

    EXPECT_DOUBLE_EQ(root.at("lr", "<test>").as_real("lr"), 3.0);
    EXPECT_EQ(root.at("lr", "<test>").as_int("lr"), 3);
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("'units'"),
                    "'units'", "whole number");
}

TEST(Small_Value, AWrongKindNamesBothTheContextAndTheKindFound)
{
    const Value root = parse_json(R"({"units": "many"})", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("layer 0 'units'"),
                    "layer 0 'units'", "a string");
}

TEST(Small_Value, AMissingKeyNamesTheKeyAndListsWhatIsPresent)
{
    const Value root = parse_json(R"({"alpha": 1, "beta": 2})", "model.json");
    SA_EXPECT_ERROR((void)root.at("gamma", "model.json"),
                    "model.json", "'gamma'", "'alpha'", "'beta'");
}

TEST(Small_Value, ReadingAKeyOffANonObjectSaysWhatItActuallyIs)
{
    const Value root = parse_json(R"([1, 2, 3])", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "the model"), "the model", "an array");
}

TEST(Small_Value, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    const Value root = parse_json(R"({"type": "linear", "unitss": 256})", "model.json");
    const std::string_view keys[] = {"type", "units"};

    SA_EXPECT_ERROR(root.reject_unknown_keys("layer 0 'linear'", allowed(keys)),
                    "layer 0 'linear'", "'unitss'", "'units'", "'type'");
}

TEST(Small_Value, AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything)
{
    const Value with_keys = parse_json(R"({"type": "linear", "units": 256})", "<test>");
    const std::string_view keys[] = {"type", "units"};
    EXPECT_NO_THROW(with_keys.reject_unknown_keys("layer 0", allowed(keys)));

    // An empty allowed set refuses every key; see tests/core/config/README.md.
    const std::string_view none[] = {"type"};
    const Value stray = parse_json(R"({"type": "gelu", "units": 8})", "<test>");
    SA_EXPECT_ERROR(stray.reject_unknown_keys("layer 1 'gelu'", allowed(none)),
                    "layer 1 'gelu'", "'units'");
}

TEST(Small_Value, WithoutRemovesTheDiscriminatorAndLeavesTheRest)
{
    const Value layer = parse_json(R"({"type": "linear", "units": 256, "use_bias": false})",
                                   "<test>");
    const Value attributes = layer.without("type");

    EXPECT_FALSE(attributes.has("type"));
    EXPECT_TRUE(attributes.has("units"));
    EXPECT_EQ(attributes.keys(), (std::vector<std::string>{"units", "use_bias"}));

    // Removing an absent key is not an error; see tests/core/config/README.md.
    EXPECT_EQ(attributes.without("type").keys().size(), 2u);
}

TEST(Small_Value, AtOrFallsBackWithoutThrowing)
{
    const Value root = parse_json(R"({"units": 256})", "<test>");
    const Value fallback = Value::boolean(true);

    EXPECT_EQ(root.at_or("units", fallback).as_int("units"), 256);
    EXPECT_TRUE(root.at_or("use_bias", fallback).as_bool("use_bias"));

    // A non-object falls back instead of throwing; see tests/core/config/README.md.
    const Value not_an_object = Value::integer(1);
    EXPECT_TRUE(not_an_object.at_or("use_bias", fallback).as_bool("use_bias"));
}

TEST(Small_Value, MalformedJSONNamesTheOriginAndTheComplaint)
{
    SA_EXPECT_ERROR((void)parse_json("{ not json", "train.json"),
                    "train.json", "not valid JSON");
}

TEST(Small_Value, AnObjectBuiltFromMismatchedVectorsIsRefused)
{
    std::vector<std::string> keys{"a", "b"};
    std::vector<Value> values{Value::integer(1)};
    SA_EXPECT_ERROR((void)Value::object(std::move(keys), std::move(values)), "2 keys", "1 value");
}

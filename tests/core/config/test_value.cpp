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
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::StatusCode;
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
    SA_ASSERT_OK_AND_ASSIGN(
        const Value root,
        parse_json(R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})",
                   "<test>"));

    ASSERT_EQ(root.kind(), Kind::Object);

    SA_ASSERT_OK_AND_ASSIGN(const Value& a, root.at("a", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t whole, a.as_int("a"));
    EXPECT_EQ(whole, 1);

    SA_ASSERT_OK_AND_ASSIGN(const Value& b, root.at("b", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const double real, b.as_real("b"));
    EXPECT_DOUBLE_EQ(real, 2.5);

    SA_ASSERT_OK_AND_ASSIGN(const Value& c, root.at("c", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const std::string& text, c.as_string("c"));
    EXPECT_EQ(text, "text");

    SA_ASSERT_OK_AND_ASSIGN(const Value& d, root.at("d", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const bool flag, d.as_bool("d"));
    EXPECT_TRUE(flag);

    SA_ASSERT_OK_AND_ASSIGN(const Value& e, root.at("e", "<test>"));
    EXPECT_TRUE(e.is_null());

    SA_ASSERT_OK_AND_ASSIGN(const Value& f, root.at("f", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Value>& entries, f.as_array("f"));
    EXPECT_EQ(entries.size(), 2u);
}

TEST(Small_Value, KeepsTheKeyOrderTheFileSpelled)
{
    // Key order follows the file; see tests/core/config/README.md.
    SA_ASSERT_OK_AND_ASSIGN(const Value root,
                            parse_json(R"({"zebra": 1, "apple": 2, "mango": 3})", "<test>"));
    EXPECT_EQ(root.keys(), (std::vector<std::string>{"zebra", "apple", "mango"}));
    EXPECT_EQ(root.values().size(), 3u);
}

TEST(Small_Value, AnIntegerReadsAsARealButARealIsNotAWholeNumber)
{
    // An integer widens to a real, never the reverse; see tests/core/config/README.md.
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({"lr": 3, "units": 2.5})", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const Value& lr, root.at("lr", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const Value& units, root.at("units", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const double widened, lr.as_real("lr"));
    EXPECT_DOUBLE_EQ(widened, 3.0);
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t whole, lr.as_int("lr"));
    EXPECT_EQ(whole, 3);
    SA_EXPECT_FAILURE(units.as_int("'units'"), StatusCode::InvalidDescription, "'units'",
                      "whole number");
}

TEST(Small_Value, AWrongKindNamesBothTheContextAndTheKindFound)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({"units": "many"})", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(const Value& units, root.at("units", "<test>"));

    SA_EXPECT_FAILURE(units.as_int("layer 0 'units'"), StatusCode::InvalidDescription,
                      "layer 0 'units'", "a string");
    SA_EXPECT_FAILURE(units.as_bool("layer 0 'units'"), StatusCode::InvalidDescription,
                      "must be a boolean", "a string");
    SA_EXPECT_FAILURE(units.as_real("layer 0 'units'"), StatusCode::InvalidDescription,
                      "must be a number", "a string");
    SA_EXPECT_FAILURE(units.as_array("layer 0 'units'"), StatusCode::InvalidDescription,
                      "must be an array", "a string");
    SA_EXPECT_FAILURE(Value::integer(1).as_string("layer 0 'units'"),
                      StatusCode::InvalidDescription, "must be a string", "a whole number");
}

TEST(Small_Value, AMissingKeyNamesTheKeyAndListsWhatIsPresent)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root,
                            parse_json(R"({"alpha": 1, "beta": 2})", "model.json"));
    SA_EXPECT_FAILURE(root.at("gamma", "model.json"), StatusCode::InvalidDescription,
                      "model.json", "'gamma'", "'alpha'", "'beta'");
}

TEST(Small_Value, ReadingAKeyOffANonObjectSaysWhatItActuallyIs)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"([1, 2, 3])", "<test>"));
    SA_EXPECT_FAILURE(root.at("units", "the model"), StatusCode::InvalidDescription,
                      "the model", "an array");
}

TEST(Small_Value, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root,
                            parse_json(R"({"type": "linear", "unitss": 256})", "model.json"));
    const std::string_view keys[] = {"type", "units"};

    SA_EXPECT_FAILURE(root.reject_unknown_keys("layer 0 'linear'", allowed(keys)),
                      StatusCode::InvalidDescription, "layer 0 'linear'", "'unitss'", "'units'",
                      "'type'");
}

TEST(Small_Value, AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value with_keys,
                            parse_json(R"({"type": "linear", "units": 256})", "<test>"));
    const std::string_view keys[] = {"type", "units"};
    SA_EXPECT_OK(with_keys.reject_unknown_keys("layer 0", allowed(keys)));

    // An empty allowed set refuses every key; see tests/core/config/README.md.
    const std::string_view none[] = {"type"};
    SA_ASSERT_OK_AND_ASSIGN(const Value stray,
                            parse_json(R"({"type": "gelu", "units": 8})", "<test>"));
    SA_EXPECT_FAILURE(stray.reject_unknown_keys("layer 1 'gelu'", allowed(none)),
                      StatusCode::InvalidDescription, "layer 1 'gelu'", "'units'");
}

TEST(Small_Value, WithoutRemovesTheDiscriminatorAndLeavesTheRest)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Value layer,
        parse_json(R"({"type": "linear", "units": 256, "use_bias": false})", "<test>"));
    const Value attributes = layer.without("type");

    EXPECT_FALSE(attributes.has("type"));
    EXPECT_TRUE(attributes.has("units"));
    EXPECT_EQ(attributes.keys(), (std::vector<std::string>{"units", "use_bias"}));

    // Removing an absent key is not an error; see tests/core/config/README.md.
    EXPECT_EQ(attributes.without("type").keys().size(), 2u);
}

TEST(Small_Value, AtOrFallsBackWithoutAFailure)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({"units": 256})", "<test>"));
    const Value fallback = Value::boolean(true);

    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t units,
                            root.at_or("units", fallback).as_int("units"));
    EXPECT_EQ(units, 256);
    SA_ASSERT_OK_AND_ASSIGN(const bool use_bias,
                            root.at_or("use_bias", fallback).as_bool("use_bias"));
    EXPECT_TRUE(use_bias);

    // A non-object falls back instead of failing; see tests/core/config/README.md.
    const Value not_an_object = Value::integer(1);
    SA_ASSERT_OK_AND_ASSIGN(const bool still_bias,
                            not_an_object.at_or("use_bias", fallback).as_bool("use_bias"));
    EXPECT_TRUE(still_bias);
}

TEST(Small_Value, MalformedJSONNamesTheOriginAndTheComplaint)
{
    SA_EXPECT_FAILURE(parse_json("{ not json", "train.json"), StatusCode::InvalidFile,
                      "train.json", "not valid JSON");
}

TEST(Small_Value, AnObjectBuiltFromMismatchedVectorsIsFatal)
{
    std::vector<std::string> keys{"a", "b"};
    std::vector<Value> values{Value::integer(1)};
    SA_EXPECT_FATAL((void)Value::object(std::move(keys), std::move(values)), "2 keys", "1 values");
}

TEST(Small_Value, RequireVersionAcceptsTheExpectedVersion)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root,
                            parse_json(R"({"sushiai_train": 4, "epochs": 2})", "train.json"));
    SA_EXPECT_OK(SushiAI::Config::require_version(root, "sushiai_train", 4, "train.json"));
}

TEST(Small_Value, RequireVersionRefusesAWrongOrAbsentVersionAsAnInvalidFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({"sushiai_train": 3})", "train.json"));
    SA_EXPECT_FAILURE(SushiAI::Config::require_version(root, "sushiai_train", 4, "train.json"),
                      StatusCode::InvalidFile, "train.json", "version 3", "reads version 4");
    SA_EXPECT_FAILURE(SushiAI::Config::require_version(root, "sushiai_data", 1, "train.json"),
                      StatusCode::InvalidFile, "has no 'sushiai_data'");

    SA_ASSERT_OK_AND_ASSIGN(const Value list, parse_json("[1]", "train.json"));
    SA_EXPECT_FAILURE(SushiAI::Config::require_version(list, "sushiai_train", 4, "train.json"),
                      StatusCode::InvalidFile, "must be an object", "an array");
}

TEST(Small_Value, ParseJSONFileNamesAMissingFile)
{
    const std::filesystem::path absent =
        std::filesystem::temp_directory_path() / "sushiai_test_value_absent.json";
    std::error_code ignored;
    std::filesystem::remove(absent, ignored);

    SA_EXPECT_FAILURE(SushiAI::Config::parse_json_file(absent), StatusCode::FileNotFound,
                      "could not open", "sushiai_test_value_absent.json");
}

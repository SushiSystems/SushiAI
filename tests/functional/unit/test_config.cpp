/**************************************************************************/
/* test_config.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the config layer is host-lane by construction - it turns text
// into a typed tree and never touches an Engine.
//
// Most of what is asserted here is a DIAGNOSTIC rather than a value. That is
// deliberate: a config layer's job is to refuse a wrong file loudly, and a
// test that only checked "it threw" would keep passing after a refactor
// collapsed five specific messages into one generic complaint. See
// tests/common/test_throws.hpp for the argument in full.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include "../../common/test_throws.hpp"

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
}

TEST(Unit_Config, ParsesEveryScalarKind)
{
    const Value root = parse_json(
        R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})", "<test>");

    ASSERT_EQ(root.kind(), Kind::OBJECT);
    EXPECT_EQ(root.at("a", "<test>").as_int("a"), 1);
    EXPECT_DOUBLE_EQ(root.at("b", "<test>").as_real("b"), 2.5);
    EXPECT_EQ(root.at("c", "<test>").as_string("c"), "text");
    EXPECT_TRUE(root.at("d", "<test>").as_bool("d"));
    EXPECT_TRUE(root.at("e", "<test>").is_null());
    EXPECT_EQ(root.at("f", "<test>").as_array("f").size(), 2u);
}

TEST(Unit_Config, KeepsTheKeyOrderTheFileSpelled)
{
    // The order is what lets a "this object carries ..." diagnostic read the
    // way the file does, so it is a property rather than an accident.
    const Value root = parse_json(R"({"zebra": 1, "apple": 2, "mango": 3})", "<test>");
    EXPECT_EQ(root.keys(), (std::vector<std::string>{"zebra", "apple", "mango"}));
}

TEST(Unit_Config, AnIntegerReadsAsARealButARealIsNotAWholeNumber)
{
    // JSON's grammar distinguishes 3 from 3.0 and this layer keeps that, so
    // "units": 2.5 is refused rather than truncated. The widening goes one
    // way only.
    const Value root = parse_json(R"({"lr": 3, "units": 2.5})", "<test>");

    EXPECT_DOUBLE_EQ(root.at("lr", "<test>").as_real("lr"), 3.0);
    EXPECT_EQ(root.at("lr", "<test>").as_int("lr"), 3);
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("'units'"),
                    "'units'", "whole number");
}

TEST(Unit_Config, AWrongKindNamesBothTheContextAndTheKindFound)
{
    const Value root = parse_json(R"({"units": "many"})", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("layer 0 'units'"),
                    "layer 0 'units'", "a string");
}

TEST(Unit_Config, AMissingKeyNamesTheKeyAndListsWhatIsPresent)
{
    const Value root = parse_json(R"({"alpha": 1, "beta": 2})", "model.json");
    SA_EXPECT_ERROR((void)root.at("gamma", "model.json"),
                    "model.json", "'gamma'", "'alpha'", "'beta'");
}

TEST(Unit_Config, ReadingAKeyOffANonObjectSaysWhatItActuallyIs)
{
    const Value root = parse_json(R"([1, 2, 3])", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "the model"), "the model", "an array");
}

TEST(Unit_Config, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    const Value root = parse_json(R"({"type": "linear", "unitss": 256})", "model.json");
    const std::string_view keys[] = {"type", "units"};

    SA_EXPECT_ERROR(root.reject_unknown_keys("layer 0 'linear'", allowed(keys)),
                    "layer 0 'linear'", "'unitss'", "'units'", "'type'");
}

TEST(Unit_Config, AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything)
{
    const Value with_keys = parse_json(R"({"type": "linear", "units": 256})", "<test>");
    const std::string_view keys[] = {"type", "units"};
    EXPECT_NO_THROW(with_keys.reject_unknown_keys("layer 0", allowed(keys)));

    // A layer taking no attributes at all - gelu - passes an empty list, so
    // any key whatsoever is an error rather than something quietly dropped.
    const std::string_view none[] = {"type"};
    const Value stray = parse_json(R"({"type": "gelu", "units": 8})", "<test>");
    SA_EXPECT_ERROR(stray.reject_unknown_keys("layer 1 'gelu'", allowed(none)),
                    "layer 1 'gelu'", "'units'");
}

TEST(Unit_Config, WithoutRemovesTheDiscriminatorAndLeavesTheRest)
{
    const Value layer = parse_json(R"({"type": "linear", "units": 256, "use_bias": false})",
                                   "<test>");
    const Value attributes = layer.without("type");

    EXPECT_FALSE(attributes.has("type"));
    EXPECT_TRUE(attributes.has("units"));
    EXPECT_EQ(attributes.keys(), (std::vector<std::string>{"units", "use_bias"}));

    // Removing a key that is not there is not an error: a layer spelled with
    // no attributes at all still goes through the same call.
    EXPECT_EQ(attributes.without("type").keys().size(), 2u);
}

TEST(Unit_Config, AtOrFallsBackWithoutThrowing)
{
    const Value root = parse_json(R"({"units": 256})", "<test>");
    const Value fallback = Value::boolean(true);

    EXPECT_EQ(root.at_or("units", fallback).as_int("units"), 256);
    EXPECT_TRUE(root.at_or("use_bias", fallback).as_bool("use_bias"));

    // A non-object falls back rather than throwing, so a layer spelled
    // {"type": "gelu"} - whose attributes are an empty object - reads its
    // optional keys through the same call as any other.
    const Value not_an_object = Value::integer(1);
    EXPECT_TRUE(not_an_object.at_or("use_bias", fallback).as_bool("use_bias"));
}

TEST(Unit_Config, MalformedJsonNamesTheOriginAndTheComplaint)
{
    SA_EXPECT_ERROR((void)parse_json("{ not json", "train.json"),
                    "train.json", "not valid JSON");
}

TEST(Unit_Config, AnObjectBuiltFromMismatchedVectorsIsRefused)
{
    std::vector<std::string> keys{"a", "b"};
    std::vector<Value> values{Value::integer(1)};
    SA_EXPECT_ERROR((void)Value::object(std::move(keys), std::move(values)), "2 keys", "1 value");
}

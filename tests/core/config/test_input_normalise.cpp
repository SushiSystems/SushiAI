/**************************************************************************/
/* test_input_normalise.cpp                                               */
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
 * @file test_input_normalise.cpp
 * @brief Tests how the "normalise" object of a model's input parses, which objects it refuses.
 * @author Mustafa Garip
 */

#include <string_view>

#include <gtest/gtest.h>
#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Config;

namespace
{
    /** @brief Names the object in a diagnostic of these tests. */
    constexpr std::string_view K_CONTEXT = "model.json's 'input' 'normalise'";

    /** @brief Parses @p text as a "normalise" object, which must be valid JSON. */
    Result<InputNormalise> parsed(std::string_view text)
    {
        SA_TRY_ASSIGN(const Value object, parse_json(text, "normalise.json"));
        return parse_input_normalise(object, K_CONTEXT);
    }

    /** @brief Holds one object a normalisation refuses and the words its message holds. */
    struct Refusal
    {
        std::string_view text;   /**< The object as the file spells it. */
        std::string_view needle; /**< A phrase the message must contain. */
    };

    /** @brief Provides one refusal per case to the parametrised test. */
    class Small_InputNormaliseRefusal : public ::testing::TestWithParam<Refusal>
    {
    };
} // namespace

/** @brief Tests that an object with no key is the default conversion. */
TEST(Small_InputNormalise, AnEmptyObjectIsTheDefault)
{
    SA_ASSERT_OK_AND_ASSIGN(const InputNormalise found, parsed("{}"));

    EXPECT_TRUE(found == InputNormalise());
    EXPECT_FALSE(found.colours_differ());
    EXPECT_EQ(found.scale, K_DEFAULT_NORMALISE_SCALE);
}

/** @brief Tests that each stated key is read into its own field and the rest stay default. */
TEST(Small_InputNormalise, EachStatedKeyIsReadIntoItsField)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const InputNormalise found,
        parsed(R"({"channels": "bgr", "scale": 0.5, "mean": [0.25, 0.5, 0.75],
                   "std": [2, 4, 8]})"));

    EXPECT_EQ(found.channels, ChannelOrder::Bgr);
    EXPECT_EQ(found.scale, 0.5);
    EXPECT_EQ(found.mean[0], 0.25);
    EXPECT_EQ(found.mean[1], 0.5);
    EXPECT_EQ(found.mean[2], 0.75);
    EXPECT_EQ(found.deviation[0], 2.0);
    EXPECT_EQ(found.deviation[1], 4.0);
    EXPECT_EQ(found.deviation[2], 8.0);
    EXPECT_TRUE(found.colours_differ());
}

/** @brief Tests that a key left out keeps its default while another is stated. */
TEST(Small_InputNormalise, AKeyLeftOutKeepsItsDefault)
{
    SA_ASSERT_OK_AND_ASSIGN(const InputNormalise found, parsed(R"({"scale": 1.0})"));

    EXPECT_EQ(found.channels, ChannelOrder::Rgb);
    EXPECT_EQ(found.scale, 1.0);
    EXPECT_TRUE(found.mean == InputNormalise().mean);
    EXPECT_TRUE(found.deviation == InputNormalise().deviation);
    EXPECT_FALSE(found.colours_differ());
}

/** @brief Tests that the values input_normalise_value builds read back to an equal value. */
TEST(Small_InputNormalise, TheValueOfANormalisationReadsBackEqual)
{
    InputNormalise written;
    written.channels = ChannelOrder::Bgr;
    written.scale = 0.125;
    written.mean = {1.0, 0.5, 0.25};
    written.deviation = {2.0, 0.5, 4.0};

    SA_ASSERT_OK_AND_ASSIGN(const InputNormalise read,
                            parse_input_normalise(input_normalise_value(written), K_CONTEXT));

    EXPECT_TRUE(read == written);
}

/** @brief Tests that a model file states the object under its input and that the key is kept. */
TEST(Small_InputNormalise, AModelInputStatesTheObjectUnderNormalise)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({
        "format_version": 2,
        "input": {"shape": [4, 4, 3], "normalise": {"channels": "bgr", "std": [1, 2, 4]}},
        "layers": [{"type": "relu"}]
    })", "model.json"));
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, parse_model(root, "model.json"));

    ASSERT_TRUE(model.input_normalise.has_value());
    EXPECT_EQ(model.input_normalise->channels, ChannelOrder::Bgr);
    EXPECT_EQ(model.input_normalise->deviation[2], 4.0);
}

/** @brief Tests that a model input without the key states no normalisation. */
TEST(Small_InputNormalise, AModelInputWithoutTheKeyStatesNone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({
        "format_version": 2, "input": {"shape": [4, 4, 3]}, "layers": [{"type": "relu"}]
    })", "model.json"));
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, parse_model(root, "model.json"));

    EXPECT_FALSE(model.input_normalise.has_value());
}

/** @brief Tests that a model input refuses an unknown key of its normalisation by name. */
TEST(Small_InputNormalise, AModelInputRefusesAnUnknownKeyOfTheObject)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({
        "format_version": 2,
        "input": {"shape": [4, 4, 3], "normalise": {"gamma": 2.2}},
        "layers": [{"type": "relu"}]
    })", "model.json"));

    SA_EXPECT_FAILURE(parse_model(root, "model.json"), StatusCode::InvalidDescription,
                      "'model.json'", "'input' 'normalise'", "does not accept 'gamma'");
}

/** @brief Tests that an object the reader refuses is refused with the phrase that names why. */
TEST_P(Small_InputNormaliseRefusal, TheObjectIsRefusedNamingWhy)
{
    const Refusal& row = GetParam();

    SA_EXPECT_FAILURE(parsed(row.text), StatusCode::InvalidDescription, K_CONTEXT, row.needle);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InputNormaliseRefusal,
    ::testing::Values(
        Refusal{R"({"gamma": 2.2})", "does not accept 'gamma'"},
        Refusal{R"({"channels": "grb"})", "'grb', which is no channel order"},
        Refusal{R"({"channels": 3})", "'channels'"},
        Refusal{R"({"scale": 0})", "above 0"},
        Refusal{R"({"scale": -0.5})", "above 0"},
        Refusal{R"({"scale": "big"})", "'scale'"},
        Refusal{R"({"mean": [0.5, 0.5]})", "holds 2 entries"},
        Refusal{R"({"mean": [0, 0, 0, 0]})", "holds 4 entries"},
        Refusal{R"({"mean": 0.5})", "'mean'"},
        Refusal{R"({"mean": [0, "a", 0]})", "'mean' entry 1"},
        Refusal{R"({"std": [1, 1]})", "'std' holds 2 entries"},
        Refusal{R"({"std": [1, 0, 1]})", "'std' entry 1 is"},
        Refusal{R"({"std": [1, 1, -2]})", "'std' entry 2 is"}));

/** @brief Tests that a "normalise" that is no object is refused. */
TEST(Small_InputNormalise, ANormaliseThatIsNoObjectIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value array, parse_json("[1, 2, 3]", "normalise.json"));

    SA_EXPECT_FAILURE(parse_input_normalise(array, K_CONTEXT), StatusCode::InvalidDescription,
                      K_CONTEXT, "must be an object");
}

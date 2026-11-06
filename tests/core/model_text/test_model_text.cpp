/**************************************************************************/
/* test_model_text.cpp                                                    */
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
 * @file test_model_text.cpp
 * @brief Tests the parsers that turn the JSON text of a test into config values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>

namespace
{
    using SushiAI::Config::Kind;
    using SushiAI::Config::ModelSpec;
    using SushiAI::Config::Value;
    using SushiAI::Test::attributes_from;
    using SushiAI::Test::model_from;
    using SushiAI::Test::try_attributes_from;
    using SushiAI::Test::try_model_from;
} // namespace

/** @brief Tests that attributes_from parses an object and keeps its keys. */
TEST(Small_ModelText, AttributesFromParsesAnObject)
{
    const Value attributes = attributes_from(R"({"units": 4, "bias": true})");

    ASSERT_EQ(attributes.kind(), Kind::Object);
    SA_ASSERT_OK_AND_ASSIGN(const Value& units, attributes.at("units", "case"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t count, units.as_int("case"));
    EXPECT_EQ(count, 4);
    EXPECT_TRUE(attributes.has("bias"));
}

/** @brief Tests that a parse failure names the origin "<test>". */
TEST(Small_ModelText, AParseFailureNamesTheTestOrigin)
{
    SA_EXPECT_FAILURE(try_attributes_from(R"({"units": )"), SushiAI::StatusCode::InvalidFile,
                      "'<test>'", "not valid JSON");
}

/** @brief Tests that attributes_from stops the process on text that is no JSON. */
TEST(Medium_ModelText, AttributesFromStopsTheProcessOnInvalidJSON)
{
    SA_EXPECT_FATAL((void)attributes_from(R"({"units": )"), "'<test>'", "not valid JSON");
}

/** @brief Tests that model_from stops the process on a model file it refuses. */
TEST(Medium_ModelText, ModelFromStopsTheProcessOnARefusedModelFile)
{
    SA_EXPECT_FATAL((void)model_from(R"({"input": 1})"), "'<test>'");
}

/** @brief Tests that try_model_from returns the refusal of a model file as a status. */
TEST(Small_ModelText, TryModelFromReturnsTheRefusalOfAModelFile)
{
    SA_EXPECT_FAILURE(try_model_from(R"({"input": {"features": 6}, "layers": []})"),
                      SushiAI::StatusCode::InvalidFile, "'<test>'", "sushiai_model");
}

/** @brief Tests that model_from parses a version 2 model into its input and its layers. */
TEST(Small_ModelText, ModelFromParsesAModel)
{
    const ModelSpec model = model_from(R"({"format_version": 2, "input": {"shape": [6]},
        "layers": [{"type": "linear", "units": 3}, {"type": "gelu"}]})");

    EXPECT_EQ(model.input_shape.num_elements(), 6);
    ASSERT_EQ(model.layers.size(), 2u);
    EXPECT_EQ(model.layers[0].type, "linear");
    EXPECT_EQ(model.layers[1].type, "gelu");
}

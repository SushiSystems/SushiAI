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
 * @file test_model_text.cpp
 * @brief Tests the parsers that turn the JSON text of a test into config values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>

namespace
{
    using SushiAI::Config::Kind;
    using SushiAI::Config::ModelSpec;
    using SushiAI::Config::Value;
    using SushiAI::Test::attributes_from;
    using SushiAI::Test::model_from;
} // namespace

/** @brief Tests that attributes_from parses an object and keeps its keys. */
TEST(Small_ModelText, AttributesFromParsesAnObject)
{
    const Value attributes = attributes_from(R"({"units": 4, "bias": true})");

    ASSERT_EQ(attributes.kind(), Kind::Object);
    EXPECT_EQ(attributes.at("units", "case").as_int("case"), 4);
    EXPECT_TRUE(attributes.has("bias"));
}

/** @brief Tests that a parse failure names the origin "<test>". */
TEST(Small_ModelText, AParseFailureNamesTheTestOrigin)
{
    SA_EXPECT_ERROR((void)attributes_from(R"({"units": )"), "'<test>'", "not valid JSON");
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

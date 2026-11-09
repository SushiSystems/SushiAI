/**************************************************************************/
/* test_minimal_case.cpp                                                  */
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
 * @file test_minimal_case.cpp
 * @brief Tests that make_minimal() builds the node a MinimalCase describes, or names its row.
 * @author Mustafa Garip
 */

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;
using SaTest::MinimalCase;

namespace
{
    /** @brief Names a minimal case that builds, with the case itself. */
    struct BuildRow
    {
        std::string name;        /**< The row's name. */
        MinimalCase minimal;     /**< The case the registry builds. */
    };

    /** @brief Runs the build of a minimal case over a layer row and a node row. */
    class Small_LayerRowsMinimalCaseBuild : public ::testing::TestWithParam<BuildRow>
    {
    };
} // namespace

/** @brief Tests that a row builds a node with the stated input count and output shape. */
TEST_P(Small_LayerRowsMinimalCaseBuild, AMinimalCaseBuildsItsStatedOutput)
{
    const MinimalCase& minimal = GetParam().minimal;
    const NN::NodeRegistry registry = SaTest::make_layer_registry();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            SaTest::make_minimal(registry, minimal, minimal.attributes));

    EXPECT_EQ(node->input_count(), minimal.inputs.size());
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, minimal.output);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRowsMinimalCaseBuild,
    ::testing::Values(
        BuildRow{"ALayerRow", {"linear", R"({"units": 5})", {Shape{6}}, Shape{5}}},
        BuildRow{"ANodeRowOfTwoInputs",
                 {"add", "{}", {Shape{2, 2, 3}, Shape{2, 2, 3}}, Shape{2, 2, 3}}}),
    [](const ::testing::TestParamInfo<BuildRow>& info) { return info.param.name; });

/** @brief Tests that a row the factory refuses is named as layer 0 and its type. */
TEST(Small_LayerRowsMinimalCase, ARefusedRowIsNamedAsLayerZero)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();
    const MinimalCase image{"conv2d", R"({"filters": 5, "kernel": 3})", {Shape{784}},
                            Shape{6, 6, 5}};

    SA_EXPECT_FAILURE(SaTest::make_minimal(registry, image, image.attributes),
                      StatusCode::InvalidDescription, "layer 0 'conv2d'", "[784]");
}

/**************************************************************************/
/* test_expert_trace.cpp                                                  */
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
 * @file test_expert_trace.cpp
 * @brief Tests that the ports of a block with experts carry distinct names.
 * @author Mustafa Garip
 */

#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;

/** @brief Tests that the three ports of a block with experts carry three different names. */
TEST(Small_BlockExpertTrace, ThePortsOfABlockWithExpertsHaveDistinctNames)
{
    const Shape input{5, 6};
    const Config::Value row = SushiAI::Test::require_value(
        Config::parse_json(R"({"experts": 4, "hidden": 8, "top_k": 2})", "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_moe_feed_forward_node(NN::ShapeSpan(&input, 1), row, "layer 0"));

    ASSERT_EQ(node->ports().size(), 3u);
    EXPECT_NE(node->ports()[0].name, node->ports()[1].name);
    EXPECT_NE(node->ports()[0].name, node->ports()[2].name);
    EXPECT_NE(node->ports()[1].name, node->ports()[2].name);
}

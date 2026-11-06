/**************************************************************************/
/* test_node_range.cpp                                                    */
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
 * @file test_node_range.cpp
 * @brief Tests that a default node range names every node of a graph.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the Engine the cases compile on. */
    class Medium_NodeRange : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that a step compiled over the default range records what the whole range does. */
TEST_F(Medium_NodeRange, TheDefaultRangeIsTheWholeGraph)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID doubled, builder.scale(leaf, 2.0));
    static_cast<void>(builder.scale(doubled, 3.0));
    Graph::TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    const Graph::NodeID count = static_cast<Graph::NodeID>(graph.node_count());

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Graph::CompiledStep> by_default,
                            Graph::CompiledStep::compile(*engine, graph, pool));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Graph::CompiledStep> whole,
        Graph::CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                                     Graph::NodeRange{0, count}));

    EXPECT_GE(by_default->operation_count(), 1u);
    EXPECT_EQ(by_default->operation_count(), whole->operation_count());
}

/**************************************************************************/
/* test_norm.cpp                                                          */
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
 * @file test_norm.cpp
 * @brief Tests the normalization lowering table and the scratch buffers of batch normalization.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/norm.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::ScratchRequest;
using Graph::ScratchRule;
using Graph::ValueID;

namespace
{
    /** @brief Pairs an operation with the row and the scratch rule the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;             /**< The operation. */
        LoweringRule rule;   /**< The row that lowers it. */
        ScratchRule scratch; /**< Its scratch rule, or nullptr. */
    };

    /** @brief Holds a traced batch normalization over a [2, 3, 3, 4] input and its gradient. */
    struct BatchNormTrace
    {
        Graph::Graph graph;                      /**< The traced graph. */
        Graph::NodeID forward = Graph::K_NO_NODE;  /**< The BATCHNORM2D node. */
        Graph::NodeID backward = Graph::K_NO_NODE; /**< The BATCHNORM2D_BACKWARD node. */
    };

    /** @brief Traces BATCHNORM2D and BATCHNORM2D_BACKWARD over an NHWC input of four channels. */
    BatchNormTrace trace_batchnorm()
    {
        BatchNormTrace trace;
        const GraphBuilder builder(trace.graph);

        const ValueID x = builder.input(Shape{2, 3, 3, 4}, "x");
        const ValueID gamma = builder.parameter(Shape{4}, "gamma");
        const ValueID beta = builder.parameter(Shape{4}, "beta");
        // The operands fit, so a refusal is a defect of the builder: value() stops the test.
        const Graph::BatchNormResult forward = builder.batchnorm2d(x, gamma, beta, 1e-5).value();

        const ValueID dy = builder.input(Shape{2, 3, 3, 4}, "dy");
        const Graph::BatchNormGrads backward =
            builder.batchnorm2d_backward(dy, x, gamma, forward.saved_mean, forward.saved_inv_std)
                .value();

        trace.forward = trace.graph.value(forward.output).producer;
        trace.backward = trace.graph.value(backward.dx).producer;
        return trace;
    }

    /** @brief Expects @p requests to belong to @p node in slot order, each [4] or [2, 3, 3, 4]. */
    void expect_requests(const std::vector<ScratchRequest>& requests, Graph::NodeID node)
    {
        for (std::size_t i = 0; i < requests.size(); ++i)
        {
            EXPECT_EQ(requests[i].node, node) << "request " << i;
            EXPECT_EQ(requests[i].slot, static_cast<int>(i)) << "request " << i;
            EXPECT_EQ(requests[i].dtype, DType::Float32) << "request " << i;
            EXPECT_TRUE(requests[i].shape == Shape({4}) || requests[i].shape == Shape({2, 3, 3, 4}))
                << "request " << i << " has extents " << requests[i].shape.to_string();
        }
    }

    /** @brief Runs once per row the header names for the norm table. */
    class Small_NormRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the norm table in the order the header writes them. */
    const std::array<ExpectedRow, 6> K_EXPECTED_ROWS = {{
        {0, Ops::K_BATCHNORM2D.id, &Graph::lower_batchnorm2d, &Graph::request_batchnorm2d_scratch},
        {1, Ops::K_BATCHNORM2D_BACKWARD.id, &Graph::lower_batchnorm2d_backward,
         &Graph::request_batchnorm2d_backward_scratch},
        {2, Ops::K_CHANNEL_AFFINE.id, &Graph::lower_channel_affine, nullptr},
        {3, Ops::K_BATCHNORM_FOLD.id, &Graph::lower_batchnorm_fold, nullptr},
        {4, Ops::K_LAYER_NORM.id, &Graph::lower_layer_norm, nullptr},
        {5, Ops::K_LAYER_NORM_BACKWARD.id, &Graph::lower_layer_norm_backward, nullptr},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_NormLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_NORM_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_NormRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_NORM_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_NORM_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, row.scratch) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
    EXPECT_EQ(Graph::find_scratch_rule(row.op), row.scratch) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_NormRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });

/** @brief Tests that the forward asks for three buffers, each of its own node, in slot order. */
TEST(Small_NormLowerings, BatchNormAsksForThreeBuffers)
{
    const BatchNormTrace trace = trace_batchnorm();

    std::vector<ScratchRequest> forward;
    Graph::request_batchnorm2d_scratch(trace.graph, trace.forward, forward);
    ASSERT_EQ(forward.size(), 3u);
    expect_requests(forward, trace.forward);
}

/** @brief Tests that the gradient asks for seven buffers, each of its own node, in slot order. */
TEST(Small_NormLowerings, BatchNormGradientAsksForSevenBuffers)
{
    const BatchNormTrace trace = trace_batchnorm();

    std::vector<ScratchRequest> backward;
    Graph::request_batchnorm2d_backward_scratch(trace.graph, trace.backward, backward);
    ASSERT_EQ(backward.size(), 7u);
    expect_requests(backward, trace.backward);
}

/** @brief Tests that a scratch rule appends to the list it is given and keeps what it holds. */
TEST(Small_NormLowerings, AScratchRuleAppendsBehindEarlierRequests)
{
    const BatchNormTrace trace = trace_batchnorm();

    std::vector<ScratchRequest> requests;
    Graph::request_batchnorm2d_scratch(trace.graph, trace.forward, requests);
    Graph::request_batchnorm2d_backward_scratch(trace.graph, trace.backward, requests);

    ASSERT_EQ(requests.size(), 10u);
    EXPECT_EQ(requests[0].node, trace.forward);
    EXPECT_EQ(requests[2].node, trace.forward);
    EXPECT_EQ(requests[3].node, trace.backward);
    EXPECT_EQ(requests[3].slot, 0);
}

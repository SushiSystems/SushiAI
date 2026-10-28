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
 * @file test_norm.cpp
 * @brief Tests the normalization lowering table and the scratch buffers of batch normalization.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
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
} // namespace

/** @brief Tests that each row pairs its operation with its function and its scratch rule. */
TEST(Small_NormLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 6> expected = {{
        {Ops::K_BATCHNORM2D.id, &Graph::lower_batchnorm2d, &Graph::request_batchnorm2d_scratch},
        {Ops::K_BATCHNORM2D_BACKWARD.id, &Graph::lower_batchnorm2d_backward,
         &Graph::request_batchnorm2d_backward_scratch},
        {Ops::K_CHANNEL_AFFINE.id, &Graph::lower_channel_affine, nullptr},
        {Ops::K_BATCHNORM_FOLD.id, &Graph::lower_batchnorm_fold, nullptr},
        {Ops::K_LAYER_NORM.id, &Graph::lower_layer_norm, nullptr},
        {Ops::K_LAYER_NORM_BACKWARD.id, &Graph::lower_layer_norm_backward, nullptr},
    }};

    ASSERT_EQ(Graph::K_NORM_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_NORM_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, expected[i].scratch) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_scratch_rule(expected[i].op), expected[i].scratch)
            << Ops::name_of(expected[i].op);
    }
}

/** @brief Tests that the forward asks for three buffers and the gradient for seven. */
TEST(Small_NormLowerings, BatchNormAsksForThreeBuffersAndItsGradientForSeven)
{
    const BatchNormTrace trace = trace_batchnorm();

    std::vector<ScratchRequest> forward;
    Graph::request_batchnorm2d_scratch(trace.graph, trace.forward, forward);
    ASSERT_EQ(forward.size(), 3u);
    expect_requests(forward, trace.forward);

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

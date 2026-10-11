/**************************************************************************/
/* test_causal_mask_backward.cpp                                          */
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
 * @file test_causal_mask_backward.cpp
 * @brief Tests the IR the backward rule of CAUSAL_MASK appends.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

namespace
{
    /** @brief Holds T, the extent of each axis of the scores every case masks. */
    constexpr int64_t K_LENGTH = 4;

    /** @brief Holds the value the forward mask writes above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Appends the mask of @p scores and the cross-entropy of its rows; returns the loss. */
    ValueID masked_loss(const GraphBuilder& builder, ValueID scores)
    {
        const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
        const ValueID targets = builder.input(Shape{K_LENGTH, K_LENGTH}, "targets");
        return builder.nll_loss(builder.log_softmax(masked.value()), targets);
    }

    /** @brief Returns every node of @p graph that carries CAUSAL_MASK, in order. */
    std::vector<Graph::Node> masks_of(const Graph::Graph& graph)
    {
        std::vector<Graph::Node> found;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_CAUSAL_MASK.id)
                found.push_back(node);
        }

        return found;
    }
}

/** @brief Tests that the mask is differentiable and has a rule. */
TEST(Unit_CausalMaskBackward, TheMaskHasARule)
{
    EXPECT_TRUE(Ops::K_CAUSAL_MASK.differentiable);
    EXPECT_FALSE(Ops::K_CAUSAL_MASK.blocks_gradient);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_CAUSAL_MASK.id));
}

/** @brief Tests that the gradient of the scores is a second mask whose fill is zero. */
TEST(Unit_CausalMaskBackward, TheScoreGradientIsTheGradientMaskedWithZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.parameter(Shape{K_LENGTH, K_LENGTH}, "scores");
    const ValueID loss = masked_loss(builder, scores);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], K_NO_VALUE);

    const std::vector<Graph::Node> masks = masks_of(graph);
    ASSERT_EQ(masks.size(), 2u);
    EXPECT_EQ(masks[0].attributes.values[0], K_FILL);
    EXPECT_EQ(masks[1].attributes.values[0], 0.0);
    ASSERT_EQ(masks[1].outputs.size(), 1u);
    EXPECT_EQ(masks[1].outputs[0], gradients[0]);
    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{K_LENGTH, K_LENGTH}));
}

/** @brief Tests that scores that need no gradient add no second mask. */
TEST(Unit_CausalMaskBackward, ScoresThatNeedNoGradientAddNoNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{K_LENGTH, K_LENGTH}, "scores");
    const ValueID loss = masked_loss(builder, scores);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    EXPECT_EQ(gradients[0], K_NO_VALUE);
    EXPECT_EQ(masks_of(graph).size(), 1u);
}

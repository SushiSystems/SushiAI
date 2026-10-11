/**************************************************************************/
/* test_layer_norm_backward.cpp                                           */
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
 * @file test_layer_norm_backward.cpp
 * @brief Tests the IR the backward rule of LAYER_NORM appends, and which operands it reaches.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

namespace
{
    /** @brief Holds the row count of the value every case normalises. */
    constexpr int64_t K_ROWS = 5;

    /** @brief Holds the row length, which is the class count of the loss. */
    constexpr int64_t K_WIDTH = 3;

    /** @brief Holds one graph: a normalised [K_ROWS, K_WIDTH] value under a cross-entropy. */
    struct Normalised
    {
        ValueID x;       /**< The value normalised. */
        ValueID gamma;   /**< The gain. */
        ValueID beta;    /**< The bias. */
        ValueID mean;    /**< The saved row means. */
        ValueID inv_std; /**< The saved inverse deviations. */
        ValueID loss;    /**< The scalar differentiated. */
    };

    /** @brief Builds the graph; @p x_is_parameter says whether x needs a gradient. */
    Normalised build(const GraphBuilder& builder, bool x_is_parameter)
    {
        const Shape shape{K_ROWS, K_WIDTH};
        const ValueID x = x_is_parameter ? builder.parameter(shape, "x")
                                         : builder.input(shape, "x");
        const ValueID gamma = builder.parameter(Shape{K_WIDTH}, "gamma");
        const ValueID beta = builder.parameter(Shape{K_WIDTH}, "beta");
        const Graph::LayerNormResult out = builder.layer_norm(x, gamma, beta, 1e-5).value();
        const ValueID targets = builder.input(shape, "targets");
        const ValueID loss = builder.nll_loss(builder.log_softmax(out.output), targets);

        return Normalised{x, gamma, beta, out.saved_mean, out.saved_inv_std, loss};
    }

    /** @brief Differentiates @p loss with respect to @p wrt. */
    std::vector<ValueID> differentiate(Graph::Graph& graph, ValueID loss,
                                       const std::vector<ValueID>& wrt)
    {
        return Autograd::differentiate(graph, loss,
                                       SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    }

    /** @brief Returns every node of @p graph that carries @p op, in order. */
    std::vector<Graph::Node> nodes_of(const Graph::Graph& graph, OpID op)
    {
        std::vector<Graph::Node> found;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                found.push_back(node);
        }

        return found;
    }
}

/** @brief Tests that the forward has a rule and its gradient identity has none. */
TEST(Unit_LayerNormBackward, TheForwardHasARuleAndItsGradientHasNone)
{
    EXPECT_TRUE(Ops::K_LAYER_NORM.differentiable);
    EXPECT_FALSE(Ops::K_LAYER_NORM_BACKWARD.differentiable);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_LAYER_NORM.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_LAYER_NORM_BACKWARD.id));
}

/** @brief Tests that one gradient node serves x, gamma and beta and reads the saved statistics. */
TEST(Unit_LayerNormBackward, OneGradientNodeServesTheThreeOperands)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Normalised built = build(builder, true);

    const std::vector<ValueID> gradients =
        differentiate(graph, built.loss, {built.x, built.gamma, built.beta});

    ASSERT_EQ(gradients.size(), 3u);
    const std::vector<Graph::Node> backward = nodes_of(graph, Ops::K_LAYER_NORM_BACKWARD.id);
    ASSERT_EQ(backward.size(), 1u);
    ASSERT_EQ(backward[0].inputs.size(), 5u);
    EXPECT_EQ(backward[0].inputs[1], built.x);
    EXPECT_EQ(backward[0].inputs[2], built.gamma);
    EXPECT_EQ(backward[0].inputs[3], built.mean);
    EXPECT_EQ(backward[0].inputs[4], built.inv_std);
    EXPECT_EQ(backward[0].outputs, gradients);

    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{K_ROWS, K_WIDTH}));
    EXPECT_EQ(graph.value(gradients[1]).shape, (Shape{K_WIDTH}));
    EXPECT_EQ(graph.value(gradients[2]).shape, (Shape{K_WIDTH}));
}

/** @brief Tests that an input leaf under the node comes back with no gradient value. */
TEST(Unit_LayerNormBackward, AnInputThatNeedsNoGradientReceivesNone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Normalised built = build(builder, false);

    const std::vector<ValueID> gradients =
        differentiate(graph, built.loss, {built.x, built.gamma});

    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_EQ(gradients[0], K_NO_VALUE);
    EXPECT_NE(gradients[1], K_NO_VALUE);
    EXPECT_EQ(nodes_of(graph, Ops::K_LAYER_NORM_BACKWARD.id).size(), 1u);
}

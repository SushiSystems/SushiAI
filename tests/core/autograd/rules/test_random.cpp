/**************************************************************************/
/* test_random.cpp                                                        */
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
 * @file test_random.cpp
 * @brief Tests the IR the backward rule of DROPOUT appends.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

namespace
{
    /** @brief Holds the row count of the value every case masks. */
    constexpr int64_t K_ROWS = 3;

    /** @brief Holds the row length of the value every case masks. */
    constexpr int64_t K_WIDTH = 4;

    /** @brief Holds the rate every case passes. */
    constexpr double K_RATE = 0.5;

    /** @brief Holds the site seed every case passes. */
    constexpr std::uint64_t K_SEED = 11;

    /** @brief Holds a graph that masks @c x and takes the cross-entropy of the masked rows. */
    struct Masked
    {
        Graph::Graph graph;         /**< The graph, with the gradient nodes once differentiated. */
        ValueID x = K_NO_VALUE;     /**< The [K_ROWS, K_WIDTH] value the mask is applied to. */
        ValueID mask = K_NO_VALUE;  /**< The mask the forward node saves. */
        ValueID loss = K_NO_VALUE;  /**< The one-element loss. */
    };

    /** @brief Fills @p out; @p x is a parameter when @p trainable, an input otherwise. */
    void build(Masked& out, bool trainable)
    {
        const GraphBuilder builder(out.graph);
        const Shape shape{K_ROWS, K_WIDTH};
        out.x = trainable ? builder.parameter(shape, "x") : builder.input(shape, "x");

        const Graph::DropoutResult dropped = builder.dropout(out.x, K_RATE, K_SEED).value();
        out.mask = dropped.saved_mask;

        const ValueID targets = builder.input(shape, "targets");
        SA_ASSERT_OK_AND_ASSIGN(out.loss,
                                builder.nll_loss(builder.log_softmax(dropped.output), targets));
    }

    /** @brief Appends the gradient of the loss at @c x and returns its id, or K_NO_VALUE. */
    ValueID differentiate_at_x(Masked& masked)
    {
        const std::vector<ValueID> wrt = {masked.x};
        const std::vector<ValueID> gradients = Autograd::differentiate(
            masked.graph, masked.loss,
            SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();

        return gradients.size() == 1u ? gradients[0] : K_NO_VALUE;
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_of(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
            found += node.op == op ? 1u : 0u;

        return found;
    }

    /** @brief Counts the nodes of @p graph that read @p value. */
    std::size_t readers_of(const Graph::Graph& graph, ValueID value)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            for (const ValueID input : node.inputs)
                found += input == value ? 1u : 0u;
        }

        return found;
    }
}

/** @brief Tests that dropout is differentiable, is no barrier and has the random table's rule. */
TEST(Small_DropoutBackward, DropoutHasARule)
{
    EXPECT_TRUE(Ops::K_DROPOUT.differentiable);
    EXPECT_FALSE(Ops::K_DROPOUT.blocks_gradient);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_DROPOUT.id));
    ASSERT_EQ(Autograd::K_RANDOM_RULES.size(), 1u);
    EXPECT_EQ(Autograd::K_RANDOM_RULES[0].op, Ops::K_DROPOUT.id);
    EXPECT_EQ(Autograd::K_RANDOM_RULES[0].rule, &Autograd::dropout_backward);
}

/** @brief Tests that the gradient of x is a MUL that reads the mask the forward node saved. */
TEST(Small_DropoutBackward, TheInputGradientIsTheGradientTimesTheSavedMask)
{
    Masked masked;
    build(masked, true);

    const ValueID gradient = differentiate_at_x(masked);

    ASSERT_NE(gradient, K_NO_VALUE);
    const Graph::Node& node = masked.graph.node(masked.graph.value(gradient).producer);
    EXPECT_EQ(node.op, Ops::K_MUL.id);
    ASSERT_EQ(node.inputs.size(), 2u);
    EXPECT_EQ(node.inputs[1], masked.mask);
    EXPECT_EQ(masked.graph.value(gradient).shape, (Shape{K_ROWS, K_WIDTH}));
    EXPECT_EQ(readers_of(masked.graph, masked.mask), 1u);
    EXPECT_EQ(count_of(masked.graph, Ops::K_DROPOUT.id), 1u)
        << "the gradient drew a mask of its own";
}

/** @brief Tests that an input that needs no gradient adds no node that reads the mask. */
TEST(Small_DropoutBackward, AnInputThatNeedsNoGradientAddsNoNode)
{
    Masked masked;
    build(masked, false);

    EXPECT_EQ(differentiate_at_x(masked), K_NO_VALUE);
    EXPECT_EQ(readers_of(masked.graph, masked.mask), 0u);
}

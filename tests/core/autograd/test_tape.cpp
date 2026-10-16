/**************************************************************************/
/* test_tape.cpp                                                          */
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
 * @file test_tape.cpp
 * @brief Tests that a rule reaches only the operands the tape says require a gradient.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using SushiAI::Shape;
using SushiAI::Autograd::GradientTape;
using SushiAI::Graph::GraphBuilder;
using SushiAI::Graph::K_NO_VALUE;
using SushiAI::Graph::ValueID;
namespace Graph = SushiAI::Graph;
namespace Ops = SushiAI::Ops;

namespace
{
    /** @brief Returns the nodes of @p graph from @p first on that run @p op, in order. */
    std::vector<Graph::Node> appended_of(const Graph::Graph& graph, std::size_t first,
                                         SushiAI::OpID op)
    {
        std::vector<Graph::Node> found;
        for (std::size_t i = first; i < graph.node_count(); ++i)
        {
            if (graph.nodes()[i].op == op)
                found.push_back(graph.nodes()[i]);
        }
        return found;
    }
} // namespace

/** @brief Tests that a tape exists only over a differentiation run and is never made empty. */
TEST(Small_GradientTape, ATapeIsNeverDefaultConstructed)
{
    static_assert(!std::is_default_constructible<GradientTape>::value,
                  "a tape is built by differentiate() over its run");
    SUCCEED();
}

/** @brief Tests that the product rule contributes to the parameter only, not to the input. */
TEST(Small_GradientTape, AnOperandThatNeedsNoGradientReceivesNoContribution)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{1}, "weight");
    const ValueID input = builder.input(Shape{1}, "input");
    const ValueID product = builder.mul(weight, input);
    const std::size_t forward_nodes = graph.node_count();

    const std::vector<ValueID> wrt{weight};
    const std::vector<ValueID> gradients = SushiAI::Autograd::differentiate(
        graph, product, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    EXPECT_NE(gradients[0], K_NO_VALUE);

    const std::vector<Graph::Node> products =
        appended_of(graph, forward_nodes, Ops::K_MUL.id);
    ASSERT_EQ(products.size(), 1u);
    ASSERT_EQ(products[0].inputs.size(), 2u);
    EXPECT_EQ(products[0].inputs[1], input);
    EXPECT_EQ(products[0].outputs.at(0), gradients[0]);
}

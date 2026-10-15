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
 * @brief Tests the shape rule and the emitted IR of DROPOUT, and its row of the lowering table.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/random.hpp>

using namespace SushiAI;
using Graph::DropoutResult;
using Graph::GraphBuilder;
using Graph::ValueID;

namespace
{
    /** @brief Holds the rate every accepted case passes. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds a site seed with bits set in both of its 32-bit halves. */
    constexpr std::uint64_t K_SEED = 0xFEDCBA9876543210ull;

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Result<DropoutResult>& refused,
                        std::size_t nodes, const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the node reads [x] and writes the value and a mask, both shaped like x. */
TEST(Small_DropoutOps, DropoutWritesItsValueAndTheMaskItSaves)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3, 4}, "x", DType::Float64);

    const Result<DropoutResult> out = builder.dropout(x, K_RATE, K_SEED);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
    EXPECT_EQ(node.op, Ops::K_DROPOUT.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{x}));
    EXPECT_EQ(node.outputs,
              (std::vector<ValueID>{out.value().output, out.value().saved_mask}));
    EXPECT_EQ(node.attributes.values[Graph::K_DROPOUT_RATE_SLOT], K_RATE);

    for (const ValueID id : node.outputs)
    {
        EXPECT_EQ(graph.value(id).shape, (Shape{2, 3, 4}));
        EXPECT_EQ(graph.value(id).dtype, DType::Float64);
        EXPECT_EQ(graph.value(id).lifetime, Graph::Lifetime::Activation);
    }

    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that a 64-bit seed comes back whole from the two attribute slots it is put in. */
TEST(Small_DropoutOps, TheSiteSeedSurvivesItsAttributeSlots)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4}, "x");

    for (const std::uint64_t seed : {K_SEED, std::uint64_t{0}, ~std::uint64_t{0}})
    {
        const Result<DropoutResult> out = builder.dropout(x, K_RATE, seed);
        ASSERT_TRUE(out.has_value());

        const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
        EXPECT_EQ(Graph::dropout_seed(node.attributes), seed);
    }
}

/** @brief Tests each refusal of the emitter: the element type, the rate and an empty input. */
TEST(Small_DropoutOps, DropoutRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 4}, "x");
    const ValueID whole = builder.input(Shape{2, 4}, "whole", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.dropout(whole, K_RATE, K_SEED), nodes,
                   "half, float32 or float64 input");
    expect_refused(graph, builder.dropout(x, 0.0, K_SEED), nodes, "rate in (0, 1)");
    expect_refused(graph, builder.dropout(x, 1.0, K_SEED), nodes, "rate in (0, 1)");
    expect_refused(graph, builder.dropout(x, -0.5, K_SEED), nodes, "rate in (0, 1)");
}

/** @brief Tests that DROPOUT has the one row of the random table and no other. */
TEST(Small_DropoutOps, DropoutHasTheRowOfTheRandomTable)
{
    ASSERT_EQ(Graph::K_RANDOM_LOWERINGS.size(), 1u);
    EXPECT_EQ(Graph::K_RANDOM_LOWERINGS[0].op, Ops::K_DROPOUT.id);
    EXPECT_EQ(Graph::find_lowering(Ops::K_DROPOUT.id), &Graph::lower_dropout);
    EXPECT_EQ(Graph::lowering_row_count(Ops::K_DROPOUT.id), 1u);
    EXPECT_FALSE(Graph::is_unlowered(Ops::K_DROPOUT.id));
}

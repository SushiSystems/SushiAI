/**************************************************************************/
/* test_rule.cpp                                                          */
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
 * @file test_rule.cpp
 * @brief Tests the lowering row type and the compile-time join of two row tables.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/rule.hpp>

using namespace SushiAI;
using Graph::LoweringEntry;
using Graph::ScratchRequest;

namespace
{
    /** @brief Stands in for a row that records nothing and succeeds. */
    Status first_rule(const Graph::LoweringContext&)
    {
        return Status();
    }

    /** @brief Stands in for a second row, so two rows are told apart by address. */
    Status second_rule(const Graph::LoweringContext&)
    {
        return Status();
    }

    /** @brief Appends one [2] FLOAT32 buffer in slot 3 of @p node. */
    void one_buffer(const Graph::Graph&, Graph::NodeID node, std::vector<ScratchRequest>& requests)
    {
        requests.push_back(ScratchRequest{node, 3, Shape{2}, DType::Float32});
    }

    /** @brief Holds a one-row table whose row has a scratch rule. */
    constexpr std::array<LoweringEntry, 1> K_LEFT = {{
        {Ops::K_RELU.id, &first_rule, &one_buffer},
    }};

    /** @brief Holds a two-row table whose rows have no scratch rule. */
    constexpr std::array<LoweringEntry, 2> K_RIGHT = {{
        {Ops::K_TANH.id, &second_rule},
        {Ops::K_SIGMOID.id, &first_rule},
    }};

    /** @brief Holds the join of the two tables, built at compile time. */
    constexpr std::array<LoweringEntry, 3> K_JOINED = Graph::concat_lowerings(K_LEFT, K_RIGHT);
} // namespace

/** @brief Tests that the join holds the rows of the first table, then those of the second. */
TEST(Small_LoweringRule, TheJoinKeepsTheFirstTableInFront)
{
    static_assert(K_JOINED.size() == K_LEFT.size() + K_RIGHT.size(),
                  "the join has one row per row of its two tables");

    EXPECT_EQ(K_JOINED[0].op, Ops::K_RELU.id);
    EXPECT_EQ(K_JOINED[0].rule, &first_rule);
    EXPECT_EQ(K_JOINED[1].op, Ops::K_TANH.id);
    EXPECT_EQ(K_JOINED[1].rule, &second_rule);
    EXPECT_EQ(K_JOINED[2].op, Ops::K_SIGMOID.id);
    EXPECT_EQ(K_JOINED[2].rule, &first_rule);
}

/** @brief Tests that a row written without a scratch rule has none, and the join keeps both. */
TEST(Small_LoweringRule, ARowWithoutAScratchRuleHasNone)
{
    EXPECT_EQ(K_JOINED[0].scratch, &one_buffer);
    EXPECT_EQ(K_JOINED[1].scratch, nullptr);
    EXPECT_EQ(K_JOINED[2].scratch, nullptr);
}

/** @brief Tests that a scratch rule reached through the row appends its buffer for the node. */
TEST(Small_LoweringRule, AScratchRuleAppendsTheBuffersOfItsNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2}, "x");
    const Graph::ValueID y = builder.relu(x);
    const Graph::NodeID node = graph.value(y).producer;

    std::vector<ScratchRequest> requests;
    K_JOINED[0].scratch(graph, node, requests);

    ASSERT_EQ(requests.size(), 1u);
    EXPECT_EQ(requests[0].node, node);
    EXPECT_EQ(requests[0].slot, 3);
    EXPECT_EQ(requests[0].shape, Shape{2});
    EXPECT_EQ(requests[0].dtype, DType::Float32);
}

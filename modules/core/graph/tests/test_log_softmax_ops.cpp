/**************************************************************************/
/* test_log_softmax_ops.cpp                                               */
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
 * @file test_log_softmax_ops.cpp
 * @brief Tests the shape rules and the emitted IR of LOG_SOFTMAX_BACKWARD.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Traced;
using Graph::ValueID;

namespace
{
    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Traced& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the gradient is one node over [dy, log_probs] shaped like both. */
TEST(Unit_LogSoftmaxOps, TheGradientWritesItsIdentityAndKeepsTheShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Float64);
    const ValueID dy = builder.input(Shape{4, 3}, "dy", DType::Float64);
    const ValueID log_probs = builder.log_softmax(x);

    const Traced out = builder.log_softmax_backward(dy, log_probs);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value()).producer);
    EXPECT_EQ(node.op, Ops::K_LOG_SOFTMAX_BACKWARD.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{dy, log_probs}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value()}));
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{4, 3}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests each refusal: a rank, an integer, a gradient of another shape or type. */
TEST(Unit_LogSoftmaxOps, TheGradientRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID cube = builder.input(Shape{2, 4, 3}, "cube");
    const ValueID whole = builder.input(Shape{4, 3}, "whole", DType::Int32);
    const ValueID log_probs = builder.input(Shape{4, 3}, "log_probs");
    const ValueID wide = builder.input(Shape{4, 5}, "wide");
    const ValueID doubled = builder.input(Shape{4, 3}, "doubled", DType::Float64);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.log_softmax_backward(cube, cube), nodes,
                   "rank-2 log-probabilities");
    expect_refused(graph, builder.log_softmax_backward(whole, whole), nodes,
                   "half, float32 or float64 log-probabilities");
    expect_refused(graph, builder.log_softmax_backward(wide, log_probs), nodes, "[4, 5]");
    expect_refused(graph, builder.log_softmax_backward(doubled, log_probs), nodes,
                   "a gradient like them");
}

/**************************************************************************/
/* test_causal_mask_ops.cpp                                               */
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
 * @file test_causal_mask_ops.cpp
 * @brief Tests the shape rules and the emitted IR of CAUSAL_MASK.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Traced;
using Graph::ValueID;

namespace
{
    /** @brief Holds the value every case writes above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Traced& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the mask is one node over its scores, with the fill in attribute slot 0. */
TEST(Unit_CausalMaskOps, TheMaskWritesItsIdentityAndKeepsTheShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{2, 5, 5}, "scores", DType::Float64);

    const Traced out = builder.causal_mask(scores, K_FILL);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value()).producer);
    EXPECT_EQ(node.op, Ops::K_CAUSAL_MASK.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{scores}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value()}));
    EXPECT_EQ(node.attributes.values[0], K_FILL);
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{2, 5, 5}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that a single matrix and a value with two leading axes are both accepted. */
TEST(Unit_CausalMaskOps, TheMaskAcceptsAnyNumberOfLeadingAxes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID matrix = builder.input(Shape{3, 3}, "matrix");
    const ValueID heads = builder.input(Shape{2, 4, 3, 3}, "heads");

    const Traced one = builder.causal_mask(matrix, K_FILL);
    const Traced many = builder.causal_mask(heads, K_FILL);

    ASSERT_TRUE(one.has_value());
    ASSERT_TRUE(many.has_value());
    EXPECT_EQ(graph.value(one.value()).shape, (Shape{3, 3}));
    EXPECT_EQ(graph.value(many.value()).shape, (Shape{2, 4, 3, 3}));
}

/** @brief Tests each refusal: a vector, a matrix that is not square, an integer, a fill. */
TEST(Unit_CausalMaskOps, TheMaskRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID vector = builder.input(Shape{5}, "vector");
    const ValueID oblong = builder.input(Shape{2, 4, 5}, "oblong");
    const ValueID whole = builder.input(Shape{5, 5}, "whole", DType::Int32);
    const ValueID scores = builder.input(Shape{5, 5}, "scores");
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.causal_mask(vector, K_FILL), nodes, "[..., T, T] scores");
    expect_refused(graph, builder.causal_mask(oblong, K_FILL), nodes, "[2, 4, 5]");
    expect_refused(graph, builder.causal_mask(whole, K_FILL), nodes,
                   "half, float32 or float64 scores");
    expect_refused(graph,
                   builder.causal_mask(scores, -std::numeric_limits<double>::infinity()), nodes,
                   "finite fill");
}

/**************************************************************************/
/* test_layer_norm_ops.cpp                                                */
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
 * @file test_layer_norm_ops.cpp
 * @brief Tests the shape rules and the emitted IR of LAYER_NORM and LAYER_NORM_BACKWARD.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::LayerNormGrads;
using Graph::LayerNormResult;
using Graph::ValueID;

namespace
{
    /** @brief Holds the last extent every case normalises over. */
    constexpr int64_t K_WIDTH = 4;

    /** @brief Holds the epsilon every case passes. */
    constexpr double K_EPSILON = 1e-5;

    /** @brief Holds the leaves of one graph: a [2, 3, K_WIDTH] input, a gain and a bias. */
    struct Leaves
    {
        ValueID x;     /**< The [2, 3, K_WIDTH] input. */
        ValueID gamma; /**< The [K_WIDTH] gain. */
        ValueID beta;  /**< The [K_WIDTH] bias. */
    };

    /** @brief Declares the three leaves in @p dtype. */
    Leaves declare_leaves(const GraphBuilder& builder, DType dtype = DType::Float32)
    {
        return Leaves{builder.input(Shape{2, 3, K_WIDTH}, "x", dtype),
                      builder.parameter(Shape{K_WIDTH}, "gamma", dtype),
                      builder.parameter(Shape{K_WIDTH}, "beta", dtype)};
    }

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    template<typename Outcome>
    void expect_refused(const Graph::Graph& graph, const Outcome& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the node reads [x, gamma, beta] and writes the value and two [rows] saves. */
TEST(Unit_LayerNormOps, LayerNormWritesItsValueAndTheTwoRowStatistics)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder, DType::Float64);

    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
    EXPECT_EQ(node.op, Ops::K_LAYER_NORM.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{leaves.x, leaves.gamma, leaves.beta}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value().output, out.value().saved_mean,
                                                  out.value().saved_inv_std}));
    EXPECT_EQ(node.attributes.values[0], K_EPSILON);

    EXPECT_EQ(graph.value(out.value().output).shape, (Shape{2, 3, K_WIDTH}));
    EXPECT_EQ(graph.value(out.value().saved_mean).shape, (Shape{6}));
    EXPECT_EQ(graph.value(out.value().saved_inv_std).shape, (Shape{6}));
    EXPECT_EQ(graph.value(out.value().output).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value().saved_mean).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value().output).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that a rank-1 input is one row, so both saved statistics hold one element. */
TEST(Unit_LayerNormOps, ARankOneInputIsOneRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{K_WIDTH}, "x");
    const ValueID gamma = builder.parameter(Shape{K_WIDTH}, "gamma");
    const ValueID beta = builder.parameter(Shape{K_WIDTH}, "beta");

    const Result<LayerNormResult> out = builder.layer_norm(x, gamma, beta, K_EPSILON);

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(graph.value(out.value().output).shape, (Shape{K_WIDTH}));
    EXPECT_EQ(graph.value(out.value().saved_mean).shape, (Shape{1}));
}

/** @brief Tests each refusal of the forward emitter: the input, the parameters, the epsilon. */
TEST(Unit_LayerNormOps, LayerNormRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder);
    const ValueID whole = builder.input(Shape{2, K_WIDTH}, "whole", DType::Int32);
    const ValueID narrow = builder.parameter(Shape{K_WIDTH - 1}, "narrow");
    const ValueID wide = builder.parameter(Shape{K_WIDTH}, "wide", DType::Float64);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.layer_norm(whole, leaves.gamma, leaves.beta, K_EPSILON), nodes,
                   "half, float32 or float64 input");
    expect_refused(graph, builder.layer_norm(leaves.x, narrow, leaves.beta, K_EPSILON), nodes,
                   "gamma as [4]");
    expect_refused(graph, builder.layer_norm(leaves.x, leaves.gamma, wide, K_EPSILON), nodes,
                   "beta as [4] of float32");
    expect_refused(graph, builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, 0.0), nodes,
                   "epsilon must be positive");
}

/** @brief Tests that the gradient node reads five operands and writes dx, dgamma and dbeta. */
TEST(Unit_LayerNormOps, LayerNormBackwardWritesTheThreeGradients)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder);
    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);
    ASSERT_TRUE(out.has_value());
    const ValueID dy = builder.input(Shape{2, 3, K_WIDTH}, "dy");

    const Result<LayerNormGrads> grads = builder.layer_norm_backward(
        dy, leaves.x, leaves.gamma, out.value().saved_mean, out.value().saved_inv_std);

    SA_EXPECT_OK(grads);
    ASSERT_TRUE(grads.has_value());
    const Graph::Node& node = graph.node(graph.value(grads.value().dx).producer);
    EXPECT_EQ(node.op, Ops::K_LAYER_NORM_BACKWARD.id);
    EXPECT_EQ(node.inputs,
              (std::vector<ValueID>{dy, leaves.x, leaves.gamma, out.value().saved_mean,
                                    out.value().saved_inv_std}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{grads.value().dx, grads.value().dgamma,
                                                  grads.value().dbeta}));
    EXPECT_EQ(graph.value(grads.value().dx).shape, (Shape{2, 3, K_WIDTH}));
    EXPECT_EQ(graph.value(grads.value().dgamma).shape, (Shape{K_WIDTH}));
    EXPECT_EQ(graph.value(grads.value().dbeta).shape, (Shape{K_WIDTH}));
}

/** @brief Tests that a gradient or a saved statistic of another shape is refused. */
TEST(Unit_LayerNormOps, LayerNormBackwardRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder);
    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);
    ASSERT_TRUE(out.has_value());
    const ValueID dy = builder.input(Shape{2, 3, K_WIDTH}, "dy");
    const ValueID flat = builder.input(Shape{6, K_WIDTH}, "flat");
    const ValueID per_column = builder.input(Shape{K_WIDTH}, "per_column");
    const std::size_t nodes = graph.node_count();

    expect_refused(graph,
                   builder.layer_norm_backward(flat, leaves.x, leaves.gamma,
                                               out.value().saved_mean,
                                               out.value().saved_inv_std),
                   nodes, "operand of [2, 3, 4]");
    expect_refused(graph,
                   builder.layer_norm_backward(dy, leaves.x, leaves.gamma, per_column,
                                               out.value().saved_inv_std),
                   nodes, "operand of [6]");
}

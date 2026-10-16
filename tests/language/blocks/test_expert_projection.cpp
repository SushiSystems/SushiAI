/**************************************************************************/
/* test_expert_projection.cpp                                             */
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
 * @file test_expert_projection.cpp
 * @brief Tests the parameters, the result and the refusals of ExpertProjection::mix.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_projection.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using Language::Blocks::ExpertProjection;

namespace
{
    /** @brief Holds the width of the result. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the hidden width of one expert. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Holds the number of experts. */
    constexpr int64_t K_EXPERTS = 3;

    /** @brief Holds the rows every case mixes. */
    constexpr int64_t K_ROWS = 5;

    /** @brief Returns a projection of the extents above, with no parameter declared. */
    ExpertProjection make_projection()
    {
        ExpertProjection projection;
        projection.dim = K_DIM;
        projection.hidden = K_HIDDEN;
        projection.experts = K_EXPERTS;
        return projection;
    }
} // namespace

/** @brief Tests that fields() names the weight and the bias, in that order. */
TEST(Small_BlockExpertProjection, TheFieldsAreTheWeightAndTheBias)
{
    constexpr auto fields = ExpertProjection::fields();

    EXPECT_EQ(std::get<0>(fields).name, "weight");
    EXPECT_EQ(std::get<1>(fields).name, "bias");
}

/** @brief Tests that a mix declares both parameters and sums the two products into [R, dim]. */
TEST(Small_BlockExpertProjection, AMixSumsTheGatedProjectionAndTheGatedBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mixed, projection.mix(builder, gated, gate));

    EXPECT_EQ(graph.value(mixed).shape, (Shape{K_ROWS, K_DIM}));
    EXPECT_EQ(projection.weight.shape(), (Shape{K_DIM, K_EXPERTS * K_HIDDEN}));
    EXPECT_EQ(projection.bias.shape(), (Shape{K_EXPERTS, K_DIM}));
    EXPECT_EQ(projection.weight.role(), NN::ParameterRole::Weight);
    EXPECT_EQ(projection.bias.role(), NN::ParameterRole::Bias);

    const Graph::Node& sum = graph.node(graph.value(mixed).producer);
    ASSERT_EQ(sum.op, Ops::K_ADD.id);
    ASSERT_EQ(sum.inputs.size(), 2u);
    const Graph::Node& projected = graph.node(graph.value(sum.inputs[0]).producer);
    const Graph::Node& biased = graph.node(graph.value(sum.inputs[1]).producer);
    EXPECT_EQ(projected.op, Ops::K_MATMUL.id);
    EXPECT_EQ(biased.op, Ops::K_MATMUL.id);
    EXPECT_EQ(projected.inputs[0], gated);
    EXPECT_EQ(biased.inputs[0], gate);
}

/** @brief Tests that a projection with an extent of zero is refused before any node. */
TEST(Small_BlockExpertProjection, AnEmptyExtentIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();
    projection.experts = 0;
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(projection.mix(builder, gated, gate), StatusCode::InvalidDescription,
                      "ExpertProjection", "experts=0");
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(projection.weight.declared());
}

/** @brief Tests that a gated value narrower than experts * hidden is refused. */
TEST(Small_BlockExpertProjection, AGatedValueOfAnotherWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();

    SA_EXPECT_FAILURE(projection.mix(builder, gated, gate), StatusCode::InvalidDescription);
}

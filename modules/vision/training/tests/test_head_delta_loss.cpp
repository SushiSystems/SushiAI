/**************************************************************************/
/* test_head_delta_loss.cpp                                               */
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
 * @file test_head_delta_loss.cpp
 * @brief Tests the nodes emit_head_delta_loss emits, its root under autograd and its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;
using Vision::Training::emit_head_delta_loss;

namespace
{
    /** @brief Bounds the error allowed on the factor the SCALE node stores. */
    constexpr double K_ATTRIBUTE_TOLERANCE = 1e-12;

    /** @brief Holds the factor the tests pass, one over a batch of four. */
    constexpr double K_SCALE = 0.25;

    /** @brief Returns a head of two samples on a 3x3 grid with seven channels, 126 elements. */
    Shape head_shape()
    {
        return Shape{2, 3, 3, 7};
    }

    /** @brief Returns the operation names of nodes [from, end). */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }
} // namespace

/** @brief Tests that the emitter appends MUL, RESHAPE, SUM_ROWS and SCALE, wired in that order. */
TEST(Unit_HeadDeltaLoss, EmitsFourNodesEndingInAScaleOfOneElement)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    const std::size_t leaves = graph.node_count();

    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    ASSERT_EQ(op_names(graph, leaves), (std::vector<std::string>{
        "sushiai.mul",
        "sushiai.reshape",
        "sushiai.sum_rows",
        "sushiai.scale",
    }));

    const Graph::Node& product = graph.nodes()[leaves];
    const Graph::Node& column = graph.nodes()[leaves + 1];
    const Graph::Node& sum = graph.nodes()[leaves + 2];
    const Graph::Node& root = graph.nodes()[leaves + 3];

    EXPECT_EQ(product.inputs, (std::vector<ValueID>{delta, head}));
    EXPECT_EQ(column.inputs, (std::vector<ValueID>{product.outputs.at(0)}));
    EXPECT_EQ(sum.inputs, (std::vector<ValueID>{column.outputs.at(0)}));
    EXPECT_EQ(root.inputs, (std::vector<ValueID>{sum.outputs.at(0)}));
    EXPECT_EQ(root.outputs, (std::vector<ValueID>{loss}));

    EXPECT_EQ(graph.value(column.outputs.at(0)).shape, (Shape{126, 1}));
    EXPECT_EQ(graph.value(loss).shape, (Shape{1}));
    EXPECT_NEAR(root.attributes.get(0), -K_SCALE, K_ATTRIBUTE_TOLERANCE);
}

/** @brief Tests that the result carries the element type of the head. */
TEST(Unit_HeadDeltaLoss, TheResultHasTheElementTypeOfTheHead)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head", DType::Float64);
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);

    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that autograd takes the result as a root and returns a head-shaped gradient. */
TEST(Unit_HeadDeltaLoss, TheResultIsARootAutogradDifferentiates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    const std::vector<ValueID> wrt = {head};
    std::vector<ValueID> gradients;
    ASSERT_NO_THROW(gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).shape, head_shape());
    EXPECT_FALSE(graph.value(delta).requires_grad);
}

/** @brief Tests that a delta of another shape throws, naming MUL and both shapes. */
TEST(Unit_HeadDeltaLoss, ADeltaOfAnotherShapeThrowsNamingBothShapes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(Shape{2, 3, 3, 6}, "delta");
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_ERROR((void)emit_head_delta_loss(builder, head, delta, K_SCALE),
                    "MUL needs equally shaped operands", "[2, 3, 3, 6]", "[2, 3, 3, 7]");
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that a delta of another element type throws, naming MUL and both types. */
TEST(Unit_HeadDeltaLoss, ADeltaOfAnotherElementTypeThrowsNamingBothTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_ERROR((void)emit_head_delta_loss(builder, head, delta, K_SCALE),
                    "MUL needs operands of one element type", "float32", "float64");
    EXPECT_EQ(graph.node_count(), leaves);
}

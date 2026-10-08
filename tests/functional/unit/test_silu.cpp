/**************************************************************************/
/* test_silu.cpp                                                          */
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
 * @file test_silu.cpp
 * @brief Tests the SiLU layer's trace, shape inference and empty state.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/silu.hpp>

using namespace SushiAI;
using NN::SiLU;

static_assert(NN::Module<SiLU>, "SiLU must satisfy the Module contract");
static_assert(!NN::ModuleContainer<SiLU>, "SiLU is a leaf, not a container");
static_assert(std::tuple_size<decltype(SiLU::fields())>::value == 0,
              "SiLU must declare no parameter field");

/** @brief Tests that tracing emits one SILU node reading the input. */
TEST(Unit_SiLU, ForwardEmitsOneSiluNodeReadingTheInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes_before = graph.node_count();

    const SiLU layer{};
    const Graph::ValueId y = layer.forward(builder, x);

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_SILU.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs.front(), y);
}

/** @brief Tests that a feature-vector batch keeps its shape and element type. */
TEST(Unit_SiLU, ForwardKeepsTheShapeAndTypeOfAFeatureBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");

    const SiLU layer{};
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));
    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
}

/** @brief Tests that an image batch keeps its shape and carries the NHWC layout tag. */
TEST(Unit_SiLU, ForwardKeepsTheShapeAndNhwcTagOfAnImageBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 6, 3}, "x");

    const SiLU layer{};
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 6, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::Nhwc);
}

/** @brief Tests that the result takes the layout of its input when that is NCHW. */
TEST(Unit_SiLU, ForwardCarriesTheLayoutOfItsInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    const SiLU layer{};
    const Graph::ValueId y = layer.forward(builder, nchw);

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::Nchw);
    EXPECT_EQ(graph.value(y).shape, graph.value(nchw).shape);
}

/** @brief Tests that output_shape() returns its input for a feature vector and for an image. */
TEST(Unit_SiLU, OutputShapeReturnsItsInput)
{
    const SiLU layer{};

    EXPECT_EQ(layer.output_shape(Shape{7}), (Shape{7}));
    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}), (Shape{8, 6, 4}));
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Unit_SiLU, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");

    SiLU layer{};
    (void)layer.forward(builder, x);

    std::size_t visits = 0;
    NN::for_each_parameter(layer, [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    });
    EXPECT_EQ(visits, 0u);
}

/**************************************************************************/
/* test_flatten.cpp                                                       */
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
 * @file test_flatten.cpp
 * @brief Tests the Flatten layer's trace, shape rule, rejections and parameter walk.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::Flatten;

namespace
{
    /** @brief Traces Flatten over a fresh input of @p batched and returns the result's shape. */
    Shape traced_shape(const Shape& batched)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x = builder.input(batched, "x");
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(Flatten{}.forward(builder, x).value()).shape;
    }
} // namespace

/** @brief Tests that Flatten satisfies the Module trait. */
TEST(Small_Flatten, SatisfiesTheModuleTrait)
{
    static_assert(NN::Module<Flatten>, "Flatten must be a Module");
    static_assert(!NN::ModuleContainer<Flatten>, "Flatten must not look like a container");
}

/** @brief Tests that a rank-4 input traces to one RESHAPE of shape [N, H * W * C]. */
TEST(Small_Flatten, ARank4InputTracesToOneReshape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 4, 5, 3}, "x");
    const std::size_t nodes_before = graph.node_count();

    const Flatten layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 60}));
    ASSERT_EQ(graph.node_count(), nodes_before + 1);

    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_RESHAPE.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs[0], x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs[0], y);
}

/** @brief Tests that a rank-3 input traces to [N, d1 * d2]. */
TEST(Small_Flatten, ARank3InputFoldsItsTwoInnerAxes)
{
    EXPECT_EQ(traced_shape(Shape{2, 3, 4}), (Shape{2, 12}));
}

/** @brief Tests that a rank-2 input comes back as the same value and adds no node. */
TEST(Small_Flatten, ARank2InputPassesThroughWithNoNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 7}, "x");
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    const Flatten layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(y, x);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}

/** @brief Tests that flattening an NHWC rank-4 input does not stamp the result with another tag. */
TEST(Small_Flatten, AnNHWCInputYieldsAnNHWCResult)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 4, 5, 3}, "x");
    ASSERT_EQ(graph.value(x).layout, Graph::TensorLayout::NHWC);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, Flatten{}.forward(builder, x));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that forward() throws on an input tagged NCHW. */
TEST(Small_Flatten, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 3, 4, 5}, "x");
    graph.value(x).layout = Graph::TensorLayout::NCHW;

    SA_EXPECT_FAILURE(Flatten{}.forward(builder, x), StatusCode::InvalidDescription);
}

/** @brief Tests that output_shape() matches the traced shape for an image and two vectors. */
TEST(Small_Flatten, OutputShapeAgreesWithTheTracedShape)
{
    const Flatten layer{};

    const Shape image = layer.output_shape(Shape{4, 5, 3}).value();
    EXPECT_EQ(image, (Shape{60}));
    EXPECT_EQ(traced_shape(Shape{2, 4, 5, 3}), (Shape{2, image[0]}));

    const Shape matrix = layer.output_shape(Shape{3, 4}).value();
    EXPECT_EQ(matrix, (Shape{12}));
    EXPECT_EQ(traced_shape(Shape{6, 3, 4}), (Shape{6, matrix[0]}));

    const Shape vector = layer.output_shape(Shape{7}).value();
    EXPECT_EQ(vector, (Shape{7}));
    EXPECT_EQ(traced_shape(Shape{2, 7}), (Shape{2, vector[0]}));
}

/** @brief Tests that forward() throws on a rank-1 input, which has no per-sample axis. */
TEST(Small_Flatten, ForwardRejectsARank1Input)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{6}, "x");
    const Flatten layer{};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);
}

/** @brief Tests that output_shape() fails on a rank-0 per-sample shape. */
TEST(Small_Flatten, OutputShapeRejectsARank0Shape)
{
    const Flatten layer{};

    SA_EXPECT_FAILURE(layer.output_shape(Shape{}), StatusCode::InvalidDescription);
}

/** @brief Tests that the parameter walk visits nothing, before and after a trace. */
TEST(Small_Flatten, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Flatten layer{};
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    const Graph::ValueID x = builder.input(Shape{2, 4, 5, 3}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    std::size_t visits = 0;
    NN::for_each_parameter(layer,
                           [&visits](const std::string&, NN::Parameter&)
                           {
                               ++visits;
                           });
    EXPECT_EQ(visits, 0u);
    EXPECT_EQ(NN::parameter_elements(layer), 0);
}

/**************************************************************************/
/* test_leaky_relu.cpp                                                    */
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
 * @file test_leaky_relu.cpp
 * @brief Tests the LeakyReLU layer's trace, shape inference and slope range.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::LeakyReLU;

static_assert(NN::Module<LeakyReLU>, "LeakyReLU must satisfy the Module contract");
static_assert(!NN::ModuleContainer<LeakyReLU>, "LeakyReLU is a leaf, not a container");
static_assert(std::tuple_size<decltype(LeakyReLU::fields())>::value == 0,
              "LeakyReLU must declare no parameter field");

namespace
{
    /** @brief Reports whether @p batched is @p sample with one batch axis in front. */
    bool is_batch_of(const Shape& batched, const Shape& sample)
    {
        if (batched.rank() != sample.rank() + 1)
            return false;

        for (int32_t axis = 0; axis < sample.rank(); ++axis)
        {
            if (batched[axis + 1] != sample[axis])
                return false;
        }
        return true;
    }

    /** @brief Reports whether tracing a layer of @p slope over a [2, 4] input is refused. */
    bool is_slope_rejected(double slope)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

        const LeakyReLU layer{slope};
        return layer.forward(builder, x).status().code() == StatusCode::InvalidDescription;
    }
} // namespace

/** @brief Tests that the default slope is 0.01. */
TEST(Small_LeakyReLU, TheDefaultSlopeIsOneHundredth)
{
    EXPECT_DOUBLE_EQ(LeakyReLU{}.slope, 0.01);
}

/** @brief Tests that tracing emits one LEAKY_RELU node reading the input with the slope. */
TEST(Small_LeakyReLU, ForwardEmitsOneLeakyReLUNodeCarryingTheSlope)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes_before = graph.node_count();

    const LeakyReLU layer{0.2};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_LEAKY_RELU.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs.front(), y);
    EXPECT_DOUBLE_EQ(node.attributes.get(0), 0.2);
}

/** @brief Tests that a feature-vector batch keeps its shape and element type. */
TEST(Small_LeakyReLU, ForwardKeepsTheShapeAndTypeOfAFeatureBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));
    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
}

/** @brief Tests that an image batch keeps its shape and carries the NHWC layout tag. */
TEST(Small_LeakyReLU, ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");

    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 6, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that the result takes the layout of its input when that is NCHW. */
TEST(Small_LeakyReLU, ForwardCarriesTheLayoutOfItsInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID nchw,
                            builder.layout_convert(x, Graph::TensorLayout::NCHW));

    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, nchw));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(y).shape, graph.value(nchw).shape);
}

/** @brief Tests that output_shape() matches the traced shape of a feature-vector batch. */
TEST(Small_LeakyReLU, OutputShapeAgreesWithForwardForAFeatureVector)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{5, 7}, "x");

    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    const Shape inferred = layer.output_shape(Shape{7}).value();
    EXPECT_EQ(inferred, (Shape{7}));
    EXPECT_TRUE(is_batch_of(graph.value(y).shape, inferred));
}

/** @brief Tests that output_shape() matches the traced shape of an image batch. */
TEST(Small_LeakyReLU, OutputShapeAgreesWithForwardForAnImage)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{3, 8, 6, 4}, "x");

    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    const Shape inferred = layer.output_shape(Shape{8, 6, 4}).value();
    EXPECT_EQ(inferred, (Shape{8, 6, 4}));
    EXPECT_TRUE(is_batch_of(graph.value(y).shape, inferred));
}

/** @brief Tests that a slope of exactly 0 is rejected. */
TEST(Small_LeakyReLU, ForwardRejectsASlopeOfZero)
{
    EXPECT_TRUE(is_slope_rejected(0.0));
}

/** @brief Tests that a slope of exactly 1 is rejected. */
TEST(Small_LeakyReLU, ForwardRejectsASlopeOfOne)
{
    EXPECT_TRUE(is_slope_rejected(1.0));
}

/** @brief Tests that a negative slope is rejected. */
TEST(Small_LeakyReLU, ForwardRejectsANegativeSlope)
{
    EXPECT_TRUE(is_slope_rejected(-0.01));
}

/** @brief Tests that a slope above 1 is rejected. */
TEST(Small_LeakyReLU, ForwardRejectsASlopeAboveOne)
{
    EXPECT_TRUE(is_slope_rejected(1.5));
}

/** @brief Tests that a NaN slope is rejected. */
TEST(Small_LeakyReLU, ForwardRejectsANaNSlope)
{
    EXPECT_TRUE(is_slope_rejected(std::numeric_limits<double>::quiet_NaN()));
}

/** @brief Tests that slopes a float32 step inside each end of the open range are accepted. */
TEST(Small_LeakyReLU, ForwardAcceptsSlopesJustInsideTheOpenRange)
{
    EXPECT_FALSE(is_slope_rejected(std::numeric_limits<float>::min()));
    EXPECT_FALSE(is_slope_rejected(1.0 - std::numeric_limits<float>::epsilon()));
}

/** @brief Tests that output_shape() refuses each slope forward() refuses and names it. */
TEST(Small_LeakyReLU, OutputShapeRejectsASlopeOutsideTheOpenRange)
{
    for (double slope : {0.0, 1.0, -0.01, 1.5, std::numeric_limits<double>::quiet_NaN()})
    {
        SCOPED_TRACE(slope);
        SA_EXPECT_FAILURE(LeakyReLU{slope}.output_shape(Shape{4}), StatusCode::InvalidDescription,
                          "LeakyReLU", "slope");
    }
}

/** @brief Tests that forward() refuses a slope in the layer's own message. */
TEST(Small_LeakyReLU, ForwardNamesTheLayerAndTheSlopeItRefuses)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    const LeakyReLU layer{1.5};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, "LeakyReLU",
                      "slope");
}

/** @brief Tests that a rejected slope leaves no node in the graph. */
TEST(Small_LeakyReLU, ARejectedSlopeEmitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t values_before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();

    const LeakyReLU layer{1.0};
    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}

/** @brief Tests that the parameter walk visits nothing, before and after a trace. */
TEST(Small_LeakyReLU, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    LeakyReLU layer{};
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    SA_EXPECT_OK(layer.forward(builder, x));

    std::size_t visits = 0;
    NN::for_each_parameter(layer, [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    });
    EXPECT_EQ(visits, 0u);
    EXPECT_EQ(NN::parameter_elements(layer), 0);
}

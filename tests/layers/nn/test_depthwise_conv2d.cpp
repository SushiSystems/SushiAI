/**************************************************************************/
/* test_depthwise_conv2d.cpp                                              */
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
 * @file test_depthwise_conv2d.cpp
 * @brief Tests the depthwise 2-D convolution's trace, parameters, shape inference and rejections.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/depthwise_conv2d.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::DepthwiseConv2d;

static_assert(NN::Module<DepthwiseConv2d>, "DepthwiseConv2d must satisfy the Module contract");
static_assert(!NN::ModuleContainer<DepthwiseConv2d>,
              "DepthwiseConv2d must not look like a container");

namespace
{
    /** @brief Builds a layer with a square window, one stride and one pad for both axes. */
    DepthwiseConv2d make_layer(int64_t channels, int64_t kernel, int64_t stride = 1,
                               int64_t pad = 0)
    {
        DepthwiseConv2d layer;
        layer.channels = channels;
        layer.geometry.kernel_h = kernel;
        layer.geometry.kernel_w = kernel;
        layer.geometry.stride_h = stride;
        layer.geometry.stride_w = stride;
        layer.geometry.pad_top = pad;
        layer.geometry.pad_left = pad;
        return layer;
    }

    /** @brief Prepends a batch extent to a per-sample [H, W, C] shape. */
    Shape with_batch(int64_t batch, const Shape& sample)
    {
        return Shape{batch, sample[0], sample[1], sample[2]};
    }

    /** @brief Traces @p layer over a batch of @p batch samples and returns the result's shape. */
    Shape traced_shape(DepthwiseConv2d layer, int64_t batch, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x = builder.input(with_batch(batch, sample), "x");
        return graph.value(layer.forward(builder, x)).shape;
    }

    /** @brief Collects the parameter walk's output as "name:extents" strings. */
    std::vector<std::string> walk(DepthwiseConv2d& layer)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(layer, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }

    /** @brief Counts the nodes of @p op the whole graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(), [op](const Graph::Node& node)
            {
                return node.op == op;
            }));
    }

    /** @brief Returns the first node of @p op, or nullptr when the graph holds none. */
    const Graph::Node* first_node(const Graph::Graph& graph, OpID op)
    {
        const auto& nodes = graph.nodes();
        const auto found = std::find_if(nodes.begin(), nodes.end(), [op](const Graph::Node& node)
        {
            return node.op == op;
        });
        return found == nodes.end() ? nullptr : &*found;
    }
} // namespace

/** @brief Tests that forward() emits one K_DEPTHWISE_CONV2D node and no BIAS_ADD node. */
TEST(Unit_DepthwiseConv2d, ForwardEmitsOneDepthwiseNodeAndNoBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(count_op(graph, Ops::K_DEPTHWISE_CONV2D.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
}

/** @brief Tests that the emitted node reads the input and the [3, 3, 4] taps. */
TEST(Unit_DepthwiseConv2d, TheEmittedNodeReadsTheInputAndTheWeight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    const Graph::Node* node = first_node(graph, Ops::K_DEPTHWISE_CONV2D.id);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->inputs.size(), 2u);
    EXPECT_EQ(node->inputs[0], x);
    EXPECT_EQ(graph.value(node->inputs[1]).shape, (Shape{3, 3, 4}));
    ASSERT_EQ(node->outputs.size(), 1u);
    EXPECT_EQ(node->outputs[0], y);
}

/** @brief Tests that a padded 3x3 stride-1 convolution keeps the spatial extents and tags NHWC. */
TEST(Unit_DepthwiseConv2d, ForwardKeepsTheSpatialExtentsAndTagsNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 8, 4}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that a stride-2 convolution over an even extent halves both spatial extents. */
TEST(Unit_DepthwiseConv2d, StrideTwoHalvesAnEvenExtent)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 2, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 4}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 3, 4}));
}

/** @brief Tests that the weight is [kernel_h, kernel_w, channels] under an anisotropic window. */
TEST(Unit_DepthwiseConv2d, TheWeightOfAnAnisotropicWindowIsHeightWidthChannels)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(3, 2);
    layer.geometry.kernel_w = 4;
    const Graph::ValueID x = builder.input(Shape{2, 6, 9, 3}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.shape(), (Shape{2, 4, 3}));
}

/** @brief Tests that the walk yields the weight alone, and that it is empty before forward(). */
TEST(Unit_DepthwiseConv2d, TheParameterWalkNamesOnlyTheWeight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    EXPECT_TRUE(walk(layer).empty());
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 3, 4]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 3 * 3 * 4);
}

/** @brief Tests that the weight is declared with the WEIGHT role. */
TEST(Unit_DepthwiseConv2d, TheWeightRecordsTheWeightRole)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Weight);
}

/** @brief Tests that the weight records its window as both fans, since each channel is its own. */
TEST(Unit_DepthwiseConv2d, TheWeightRecordsItsWindowAsBothFans)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.fan_in(), 3 * 3);
    EXPECT_EQ(layer.weight.fan_out(), 3 * 3);
}

/** @brief Tests that a FLOAT64 input gives a FLOAT64 weight and a FLOAT64 result. */
TEST(Unit_DepthwiseConv2d, TheWeightAndTheResultTakeTheInputElementType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(2, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x", DType::Float64);
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
}

/** @brief Tests that output_shape() keeps the extents of a padded 3x3 window at stride 1. */
TEST(Unit_DepthwiseConv2d, OutputShapeAtStrideOneKeepsTheExtents)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 1, 1);

    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}), (Shape{8, 6, 4}));
}

/** @brief Tests that output_shape() halves the extents of a padded 3x3 window at stride 2. */
TEST(Unit_DepthwiseConv2d, OutputShapeAtStrideTwoHalvesTheExtents)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 2, 1);

    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}), (Shape{4, 3, 4}));
}

/** @brief Tests that output_shape() matches the traced shape at stride 1 and at stride 2. */
TEST(Unit_DepthwiseConv2d, OutputShapeAgreesWithForwardAtStrideOneAndTwo)
{
    const Shape sample{8, 6, 4};

    const DepthwiseConv2d unit = make_layer(4, 3, 1, 1);
    EXPECT_EQ(traced_shape(unit, 3, sample), with_batch(3, unit.output_shape(sample)));

    const DepthwiseConv2d strided = make_layer(4, 3, 2, 1);
    EXPECT_EQ(traced_shape(strided, 3, sample), with_batch(3, strided.output_shape(sample)));
}

/** @brief Tests that output_shape() drops the remainder of an odd extent at stride 2 and agrees. */
TEST(Unit_DepthwiseConv2d, OutputShapeAgreesWithForwardOverAnOddExtent)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 2, 1);
    const Shape sample{7, 9, 4};

    EXPECT_EQ(layer.output_shape(sample), (Shape{4, 5, 4}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that same_padded() sets the kernel, the stride, kernel / 2 pads and dilation 1. */
TEST(Unit_DepthwiseConv2d, SamePaddedBuildsAPaddedLayerOfTheGivenStride)
{
    const DepthwiseConv2d layer = DepthwiseConv2d::same_padded(4, 4, 5, 2);

    EXPECT_EQ(layer.channels, 4);
    EXPECT_EQ(layer.geometry.kernel_h, 5);
    EXPECT_EQ(layer.geometry.kernel_w, 5);
    EXPECT_EQ(layer.geometry.stride_h, 2);
    EXPECT_EQ(layer.geometry.stride_w, 2);
    EXPECT_EQ(layer.geometry.pad_top, 2);
    EXPECT_EQ(layer.geometry.pad_left, 2);
    EXPECT_EQ(layer.geometry.dilation_h, 1);
    EXPECT_EQ(layer.geometry.dilation_w, 1);
}

/** @brief Tests that same_padded() throws when the input and output channel counts differ. */
TEST(Unit_DepthwiseConv2d, SamePaddedRefusesUnequalChannelCounts)
{
    SA_EXPECT_ERROR((void)DepthwiseConv2d::same_padded(4, 8, 3, 1), "keeps its channel count");
}

/** @brief Tests that forward() throws on a rank-2 input, naming the rank-4 requirement. */
TEST(Unit_DepthwiseConv2d, ForwardRejectsAnInputThatIsNotRankFour)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 16}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "rank-4");
}

/** @brief Tests that forward() throws on a rank-4 input tagged NCHW, naming NHWC. */
TEST(Unit_DepthwiseConv2d, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    const Graph::ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    SA_EXPECT_ERROR((void)layer.forward(builder, nchw), "NHWC");
}

/** @brief Tests that forward() throws when the input's channel count differs from the layer's. */
TEST(Unit_DepthwiseConv2d, ForwardRejectsAChannelCountThatDiffersFromTheLayer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "[H, W, 4]");
}

/** @brief Tests that forward() throws on a non-positive channel count, naming it. */
TEST(Unit_DepthwiseConv2d, ForwardRejectsANonPositiveChannelCount)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(0, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "positive channel count");
}

/** @brief Tests that forward() throws when the window is larger than the input. */
TEST(Unit_DepthwiseConv2d, ForwardRejectsAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 5, 1, 0);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 4}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that output_shape() throws when the sample's channel count differs. */
TEST(Unit_DepthwiseConv2d, OutputShapeRejectsAChannelCountThatDiffersFromTheLayer)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 1, 1);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 5}), "[H, W, 4]");
}

/** @brief Tests that output_shape() throws on a sample that is not an [H, W, C] shape. */
TEST(Unit_DepthwiseConv2d, OutputShapeRejectsASampleOfTheWrongRank)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 1, 1);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8}), "[H, W, 4]");
}

/**************************************************************************/
/* test_conv2d.cpp                                                        */
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
 * @file test_conv2d.cpp
 * @brief Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using NN::Conv2d;

static_assert(NN::Module<Conv2d>, "Conv2d must satisfy the Module contract");
static_assert(!NN::ModuleContainer<Conv2d>, "Conv2d must not look like a container");

namespace
{
    /** @brief Builds a layer with a square window, one stride and one pad for both axes. */
    Conv2d make_conv(int64_t in_channels, int64_t out_channels, int64_t kernel,
                     int64_t stride = 1, int64_t pad = 0)
    {
        Conv2d layer;
        layer.in_channels = in_channels;
        layer.out_channels = out_channels;
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
    Shape traced_shape(Conv2d layer, int64_t batch, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueId x = builder.input(with_batch(batch, sample), "x");
        return graph.value(layer.forward(builder, x)).shape;
    }

    /** @brief Collects the parameter walk's output as "name:extents" strings. */
    std::vector<std::string> walk(Conv2d& layer)
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
} // namespace

/** @brief Tests that a padded 3x3 convolution yields an NHWC [4, 28, 28, 8] result. */
TEST(Unit_Conv2d, ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 28, 28, 8}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that a FLOAT64 input gives FLOAT64 parameters and a FLOAT64 result. */
TEST(Unit_Conv2d, TheParametersAndTheResultTakeTheInputElementType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{2, 6, 6, 2}, "x", DType::FLOAT64);
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(layer.weight.dtype(), DType::FLOAT64);
    EXPECT_EQ(layer.bias.dtype(), DType::FLOAT64);
    EXPECT_EQ(graph.value(y).dtype, DType::FLOAT64);
}

/** @brief Tests that the walk yields weight [8, 3, 3, 1] then bias [8] under Linear's names. */
TEST(Unit_Conv2d, TheParameterWalkNamesTheWeightAndTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");
    EXPECT_TRUE(walk(layer).empty());
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[8, 3, 3, 1]", "bias:[8]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 8 * 3 * 3 * 1 + 8);
}

/** @brief Tests that the weight's extents follow an anisotropic window as [out, kh, kw, in]. */
TEST(Unit_Conv2d, TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 5, 1);
    layer.geometry.kernel_h = 2;
    layer.geometry.kernel_w = 4;
    const Graph::ValueId x = builder.input(Shape{2, 6, 9, 3}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.shape(), (Shape{5, 2, 4, 3}));
}

/** @brief Tests that a layer with a bias traces exactly one BIAS_ADD node. */
TEST(Unit_Conv2d, ALayerWithABiasTracesOneBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(count_op(graph, Ops::BIAS_ADD.id), 1u);
}

/** @brief Tests that use_bias = false yields the weight alone and traces no BIAS_ADD node. */
TEST(Unit_Conv2d, ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    layer.use_bias = false;
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[8, 3, 3, 1]"}));
    EXPECT_FALSE(layer.bias.declared());
    EXPECT_EQ(count_op(graph, Ops::BIAS_ADD.id), 0u);
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 28, 28, 8}));
}

/** @brief Tests that the weight and the bias record fan_in = kh*kw*in and fan_out = kh*kw*out. */
TEST(Unit_Conv2d, TheWeightAndTheBiasRecordTheirFans)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(4, 6, 3, 1, 1);
    layer.geometry.kernel_w = 5;
    layer.geometry.pad_left = 2;
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.fan_in(), 3 * 5 * 4);
    EXPECT_EQ(layer.weight.fan_out(), 3 * 5 * 6);
    EXPECT_EQ(layer.bias.fan_in(), 3 * 5 * 4);
    EXPECT_EQ(layer.bias.fan_out(), 3 * 5 * 6);
}

/** @brief Tests that the weight is declared as a WEIGHT and the bias as a BIAS. */
TEST(Unit_Conv2d, TheParametersRecordTheirRoles)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::WEIGHT);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::BIAS);
}

/** @brief Tests that a second trace reuses the parameters the first one declared. */
TEST(Unit_Conv2d, TracingTwiceDoesNotDuplicateParameters)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 28, 28, 1}, "x");

    (void)layer.forward(builder, x);
    const Graph::ValueId first_weight = layer.weight.id();
    const Graph::ValueId first_bias = layer.bias.id();

    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.id(), first_weight);
    EXPECT_EQ(layer.bias.id(), first_bias);
    EXPECT_EQ(NN::parameter_count(layer), 2u);
}

/** @brief Tests that output_shape() matches the traced shape for a stride-2 unpadded window. */
TEST(Unit_Conv2d, OutputShapeAgreesWithForwardForAStridedWindow)
{
    const Conv2d layer = make_conv(3, 16, 3, 2);
    const Shape sample{9, 12, 3};

    EXPECT_EQ(layer.output_shape(sample), (Shape{4, 5, 16}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape for a padded 5x5 window. */
TEST(Unit_Conv2d, OutputShapeAgreesWithForwardForAPaddedWindow)
{
    const Conv2d layer = make_conv(2, 4, 5, 1, 2);
    const Shape sample{7, 10, 2};

    EXPECT_EQ(layer.output_shape(sample), (Shape{7, 10, 4}));
    EXPECT_EQ(traced_shape(layer, 3, sample), with_batch(3, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape for a dilated strided window. */
TEST(Unit_Conv2d, OutputShapeAgreesWithForwardForADilatedStridedWindow)
{
    Conv2d layer = make_conv(2, 3, 3, 2, 1);
    layer.geometry.dilation_h = 2;
    layer.geometry.dilation_w = 3;
    const Shape sample{11, 13, 2};

    EXPECT_EQ(layer.output_shape(sample), (Shape{5, 5, 3}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape for a pointwise convolution. */
TEST(Unit_Conv2d, OutputShapeAgreesWithForwardForAPointwiseWindow)
{
    const Conv2d layer = make_conv(4, 6, 1);
    const Shape sample{5, 7, 4};

    EXPECT_EQ(layer.output_shape(sample), (Shape{5, 7, 6}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() refuses a layer whose input channel count is zero. */
TEST(Unit_Conv2d, OutputShapeRejectsANonPositiveInputChannelCount)
{
    const Conv2d layer = make_conv(0, 8, 3);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "Conv2d", "in=0", "out=8");
}

/** @brief Tests that output_shape() refuses a layer whose output channel count is negative. */
TEST(Unit_Conv2d, OutputShapeRejectsANonPositiveOutputChannelCount)
{
    const Conv2d layer = make_conv(3, -1, 3);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "Conv2d", "in=3", "out=-1");
}

/** @brief Tests that output_shape() refuses a per-sample channel count other than in_channels. */
TEST(Unit_Conv2d, OutputShapeRejectsAWrongChannelCount)
{
    const Conv2d layer = make_conv(3, 8, 3);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 4}),
                    "Conv2d(3, 8)", "[8, 8, 4]", "[H, W, 3]");
}

/** @brief Tests that output_shape() refuses a per-sample feature vector. */
TEST(Unit_Conv2d, OutputShapeRejectsAFeatureVector)
{
    const Conv2d layer = make_conv(3, 8, 3);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{3}), "Conv2d(3, 8)", "[3]", "[H, W, 3]");
}

/** @brief Tests that output_shape() refuses a shape that still carries the batch axis. */
TEST(Unit_Conv2d, OutputShapeRejectsABatchedShape)
{
    const Conv2d layer = make_conv(3, 8, 3);

    // The third extent equals in_channels, so only the rank check can refuse this shape.
    SA_EXPECT_ERROR((void)layer.output_shape(Shape{2, 8, 3, 3}),
                    "Conv2d(3, 8)", "[2, 8, 3, 3]", "[H, W, 3]");
}

/** @brief Tests that output_shape() refuses a window larger than the padded input. */
TEST(Unit_Conv2d, OutputShapeRejectsAWindowThatDoesNotFit)
{
    const Conv2d layer = make_conv(3, 8, 5);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{4, 4, 3}), "Conv2d(3, 8)", "[4, 4, 3]", "5x5");
}

/** @brief Tests that output_shape() refuses a zero stride. */
TEST(Unit_Conv2d, OutputShapeRejectsAZeroStride)
{
    const Conv2d layer = make_conv(3, 8, 3, 0);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "Conv2d(3, 8)", "stride 0,0");
}

/** @brief Tests that output_shape() refuses a negative padding and names it. */
TEST(Unit_Conv2d, OutputShapeRejectsANegativePadding)
{
    const Conv2d layer = make_conv(3, 8, 3, 1, -1);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "Conv2d(3, 8)", "padding -1,-1");
}

/** @brief Tests that forward() refuses an input whose channel count is not in_channels. */
TEST(Unit_Conv2d, ForwardRejectsAWrongChannelCount)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 8, 3);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 4}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Conv2d(3, 8)", "[8, 8, 4]", "[H, W, 3]");
    EXPECT_FALSE(layer.weight.declared());
}

/** @brief Tests that forward() refuses a rank-2 input and declares no parameter for it. */
TEST(Unit_Conv2d, ForwardRejectsAnInputThatIsNotRankFour)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 8, 3);
    const Graph::ValueId x = builder.input(Shape{2, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Conv2d(3, 8)", "rank-4", "[2, 3]");
    EXPECT_FALSE(layer.weight.declared());
}

/** @brief Tests that forward() refuses non-positive channel counts. */
TEST(Unit_Conv2d, ForwardRejectsNonPositiveChannelCounts)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 0, 3);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Conv2d", "in=3", "out=0");
}

/** @brief Tests that forward() refuses a window larger than the padded input. */
TEST(Unit_Conv2d, ForwardRejectsAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 8, 5);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Conv2d(3, 8)", "[4, 4, 3]", "5x5");
}

/** @brief Tests that forward() refuses an NCHW input before it declares a value or a node. */
TEST(Unit_Conv2d, ForwardRejectsAnNchwInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 8, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{2, 3, 3, 3}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    // Every extent is 3, so the permuted shape passes every check but the layout tag's.
    SA_EXPECT_ERROR((void)layer.forward(builder, nchw), "Conv2d", "NHWC");

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_FALSE(layer.bias.declared());
}

/** @brief Tests that differentiating a loss over the layer yields a weight and a bias gradient. */
TEST(Unit_Conv2d, DifferentiatingALossYieldsAGradientForTheWeightAndTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    const Graph::ValueId x = builder.input(Shape{4, 6, 6, 2}, "x");
    const Graph::ValueId targets = builder.input(Shape{4, 3}, "targets");

    const Graph::ValueId logits = builder.spatial_mean(layer.forward(builder, x));
    const Graph::ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::vector<Graph::ValueId> wrt = {layer.weight.id(), layer.bias.id()};
    const std::vector<Graph::ValueId> grads = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const Graph::ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 2u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], Graph::NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
    }
}

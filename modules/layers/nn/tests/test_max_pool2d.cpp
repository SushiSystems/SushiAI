/**************************************************************************/
/* test_max_pool2d.cpp                                                    */
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
 * @file test_max_pool2d.cpp
 * @brief Tests the 2-D max pooling layer's trace, shape inference and rejections.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using namespace SushiAI;
using NN::MaxPool2d;

static_assert(NN::Module<MaxPool2d>, "MaxPool2d must satisfy the Module contract");
static_assert(!NN::ModuleContainer<MaxPool2d>, "MaxPool2d must not look like a container");

namespace
{
    /** @brief Builds a layer whose window has the given taps, steps and leading pads. */
    MaxPool2d make_pool(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                        int64_t pad_top = 0, int64_t pad_left = 0)
    {
        MaxPool2d layer;
        layer.geometry.kernel_h = kernel_h;
        layer.geometry.kernel_w = kernel_w;
        layer.geometry.stride_h = stride_h;
        layer.geometry.stride_w = stride_w;
        layer.geometry.pad_top = pad_top;
        layer.geometry.pad_left = pad_left;
        return layer;
    }

    /** @brief Traces @p layer over a batch of @p batch samples and returns the result's shape. */
    Shape traced_shape(const MaxPool2d& layer, int64_t batch, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x =
            builder.input(Shape{batch, sample[0], sample[1], sample[2]}, "x");
        return graph.value(layer.forward(builder, x)).shape;
    }

    /** @brief Prepends a batch extent to a per-sample [H, W, C] shape. */
    Shape with_batch(int64_t batch, const Shape& sample)
    {
        return Shape{batch, sample[0], sample[1], sample[2]};
    }
} // namespace

/** @brief Tests that a 2x2 stride-2 pool halves both spatial extents and keeps N and C. */
TEST(Unit_MaxPool2d, ForwardHalvesTheSpatialExtentsOfAnEvenInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 3, 3}));
}

/** @brief Tests that the traced rank-4 result carries the NHWC layout tag. */
TEST(Unit_MaxPool2d, ForwardTagsItsResultNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that forward() returns the pooled value and not the INT64 argmax beside it. */
TEST(Unit_MaxPool2d, ForwardReturnsThePooledValueNotTheArgmax)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
    EXPECT_NE(graph.value(y).dtype, DType::Int64);
}

/** @brief Tests that output_shape() returns [out_h, out_w, C] for an unpadded square window. */
TEST(Unit_MaxPool2d, OutputShapeOfAnUnpaddedSquareWindow)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2);

    EXPECT_EQ(layer.output_shape(Shape{8, 6, 3}), (Shape{4, 3, 3}));
}

/** @brief Tests that output_shape() matches the traced shape for an unpadded square window. */
TEST(Unit_MaxPool2d, OutputShapeAgreesWithForwardForASquareWindow)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Shape sample{8, 6, 3};

    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape for a padded anisotropic window. */
TEST(Unit_MaxPool2d, OutputShapeAgreesWithForwardForAPaddedAnisotropicWindow)
{
    const MaxPool2d layer = make_pool(3, 2, 2, 1, 1, 0);
    const Shape sample{5, 7, 4};

    EXPECT_EQ(layer.output_shape(sample), (Shape{3, 6, 4}));
    EXPECT_EQ(traced_shape(layer, 3, sample), with_batch(3, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape when the stride drops a remainder. */
TEST(Unit_MaxPool2d, OutputShapeAgreesWithForwardWhenTheStrideLeavesARemainder)
{
    const MaxPool2d layer = make_pool(3, 3, 2, 2);
    const Shape sample{8, 10, 1};

    EXPECT_EQ(layer.output_shape(sample), (Shape{3, 4, 1}));
    EXPECT_EQ(traced_shape(layer, 1, sample), with_batch(1, layer.output_shape(sample)));
}

/** @brief Tests that output_shape() matches the traced shape for unequal dilations per axis. */
TEST(Unit_MaxPool2d, OutputShapeAgreesWithForwardForADilatedAnisotropicWindow)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.geometry.dilation_h = 2;
    layer.geometry.dilation_w = 3;
    const Shape sample{7, 9, 2};

    EXPECT_EQ(layer.output_shape(sample), (Shape{5, 6, 2}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that forward() throws on a rank-2 input. */
TEST(Unit_MaxPool2d, ForwardRejectsAnInputThatIsNotRankFour)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 16}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws when the window is larger than the padded input. */
TEST(Unit_MaxPool2d, ForwardRejectsAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(5, 5, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 3}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws on a rank-4 input tagged NCHW. */
TEST(Unit_MaxPool2d, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    const Graph::ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    EXPECT_THROW((void)layer.forward(builder, nchw), Error);
}

/** @brief Tests that forward() names the layout, not the window, for a narrow NCHW input. */
TEST(Unit_MaxPool2d, ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 8, 8, 2}, "x");
    const Graph::ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    const MaxPool2d layer = make_pool(3, 3, 1, 1);

    SA_EXPECT_ERROR((void)layer.forward(builder, nchw), "NHWC");
}

/** @brief Tests that forward() refuses a zero vertical stride in the layer's own message. */
TEST(Unit_MaxPool2d, ForwardRejectsAZeroStride)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 0, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "MaxPool2d", "stride 0,2");
}

/** @brief Tests that forward() refuses a negative padding in the layer's own message. */
TEST(Unit_MaxPool2d, ForwardRejectsANegativePadding)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2, 0, -1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "MaxPool2d", "padding 0,-1");
}

/** @brief Tests that output_shape() throws when the vertical stride is zero. */
TEST(Unit_MaxPool2d, OutputShapeRejectsAZeroStride)
{
    const MaxPool2d layer = make_pool(2, 2, 0, 2);

    EXPECT_THROW((void)layer.output_shape(Shape{8, 8, 3}), Error);
}

/** @brief Tests that output_shape() names the stride it refuses, not the window's fit. */
TEST(Unit_MaxPool2d, OutputShapeNamesAZeroStride)
{
    const MaxPool2d layer = make_pool(2, 2, 0, 2);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "MaxPool2d", "stride 0,2");
}

/** @brief Tests that output_shape() refuses a negative padding and names it. */
TEST(Unit_MaxPool2d, OutputShapeRejectsANegativePadding)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2, 0, -1);

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "MaxPool2d", "padding 0,-1");
}

/** @brief Tests that output_shape() throws on a per-sample feature vector. */
TEST(Unit_MaxPool2d, OutputShapeRejectsAFeatureVector)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2);

    EXPECT_THROW((void)layer.output_shape(Shape{16}), Error);
}

/** @brief Tests that output_shape() throws on a shape that still carries the batch axis. */
TEST(Unit_MaxPool2d, OutputShapeRejectsABatchedShape)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2);

    EXPECT_THROW((void)layer.output_shape(Shape{2, 8, 8, 3}), Error);
}

/** @brief Tests that output_shape() throws when the window is taller than the padded input. */
TEST(Unit_MaxPool2d, OutputShapeRejectsAWindowTallerThanTheInput)
{
    const MaxPool2d layer = make_pool(5, 2, 1, 1);

    EXPECT_THROW((void)layer.output_shape(Shape{4, 8, 3}), Error);
}

/** @brief Tests that output_shape() throws when the window is wider than the padded input. */
TEST(Unit_MaxPool2d, OutputShapeRejectsAWindowWiderThanTheInput)
{
    const MaxPool2d layer = make_pool(2, 5, 1, 1);

    EXPECT_THROW((void)layer.output_shape(Shape{8, 4, 3}), Error);
}

/** @brief Tests that a trailing row keeps 13x13 at stride one; without it the result is 12x12. */
TEST(Unit_MaxPool2d, TrailingRowKeepsTheExtentOfAStrideOnePool)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    const Shape sample{13, 13, 4};

    EXPECT_EQ(layer.output_shape(sample), (Shape{12, 12, 4}));

    layer.trailing_pad_h = 1;
    layer.trailing_pad_w = 1;

    EXPECT_EQ(layer.output_shape(sample), (Shape{13, 13, 4}));
    EXPECT_EQ(traced_shape(layer, 2, sample), with_batch(2, layer.output_shape(sample)));
}

/** @brief Tests that a trailing pad at stride two gives 7x7 over 13x13, as (13 + 1 - 2) / 2 + 1. */
TEST(Unit_MaxPool2d, TrailingPadAtStrideTwoGivesSevenFromThirteen)
{
    MaxPool2d layer = make_pool(2, 2, 2, 2);
    layer.trailing_pad_h = 1;
    layer.trailing_pad_w = 1;

    EXPECT_EQ(layer.output_shape(Shape{13, 13, 4}), (Shape{7, 7, 4}));
    EXPECT_EQ(traced_shape(layer, 1, Shape{13, 13, 4}), (Shape{1, 7, 7, 4}));
}

/** @brief Tests that a 2/2 pool over 416 with one trailing row and column gives 208. */
TEST(Unit_MaxPool2d, TrailingPadGivesTwoHundredEightFromFourHundredSixteen)
{
    MaxPool2d layer = make_pool(2, 2, 2, 2);
    layer.trailing_pad_h = 1;
    layer.trailing_pad_w = 1;

    EXPECT_EQ(layer.output_shape(Shape{416, 416, 16}), (Shape{208, 208, 16}));
}

/** @brief Tests that the trailing pad is set per axis: one row alone gives 3x2 over 3x3 at 2/1. */
TEST(Unit_MaxPool2d, TrailingPadAppliesToOneAxisOnly)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.trailing_pad_h = 1;

    EXPECT_EQ(layer.output_shape(Shape{3, 3, 2}), (Shape{3, 2, 2}));
}

/** @brief Tests that a trailing pad lets a 2-row window fit a 1-row input it would overrun. */
TEST(Unit_MaxPool2d, TrailingPadLetsAWindowFitAShortInput)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    const Shape sample{1, 4, 3};

    EXPECT_THROW((void)layer.output_shape(sample), Error);

    layer.trailing_pad_h = 1;
    layer.trailing_pad_w = 1;

    EXPECT_EQ(layer.output_shape(sample), (Shape{1, 4, 3}));
    EXPECT_EQ(traced_shape(layer, 1, sample), (Shape{1, 1, 4, 3}));
}

/** @brief Tests that output_shape() refuses a negative trailing pad and names the value. */
TEST(Unit_MaxPool2d, OutputShapeRejectsANegativeTrailingPad)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.trailing_pad_w = -1;

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "MaxPool2d",
                    "non-negative trailing padding", "0,-1");
}

/** @brief Tests that forward() refuses a negative trailing pad before it emits a node. */
TEST(Unit_MaxPool2d, ForwardRejectsANegativeTrailingPad)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.trailing_pad_h = -2;
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "MaxPool2d", "non-negative trailing padding",
                    "-2,0");
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a trailing pad too large for 64 bits is refused rather than wrapping. */
TEST(Unit_MaxPool2d, OutputShapeRejectsATrailingPadThatOverflows)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.trailing_pad_h = std::numeric_limits<int64_t>::max();

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{8, 8, 3}), "MaxPool2d", "within 64 bits");
}

/** @brief Tests that a trailing pad as tall as the window is refused: its last row is padding. */
TEST(Unit_MaxPool2d, OutputShapeRejectsATrailingPadThatLeavesAWindowNoTap)
{
    MaxPool2d layer = make_pool(2, 2, 1, 1);
    layer.trailing_pad_h = 2;

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{3, 3, 2}), "MaxPool2d",
                    "every window a tap inside the input", "2,0");
}

/** @brief Tests that forward() refuses an all-padding trailing window before it emits a node. */
TEST(Unit_MaxPool2d, ForwardRejectsATrailingPadThatLeavesAWindowNoTap)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MaxPool2d layer = make_pool(2, 2, 2, 2);
    layer.trailing_pad_w = 3;
    const Graph::ValueID x = builder.input(Shape{1, 4, 4, 2}, "x");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "MaxPool2d",
                    "every window a tap inside the input", "0,3");
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that the parameter walk visits nothing, before and after a trace. */
TEST(Unit_MaxPool2d, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MaxPool2d layer = make_pool(2, 2, 2, 2);
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(NN::parameter_count(layer), 0u);
}

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

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <common/test_throws.hpp>

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

        const Graph::ValueId x =
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
    const Graph::ValueId x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 3, 3}));
}

/** @brief Tests that the traced rank-4 result carries the NHWC layout tag. */
TEST(Unit_MaxPool2d, ForwardTagsItsResultNhwc)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueId x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::Nhwc);
}

/** @brief Tests that forward() returns the pooled value and not the INT64 argmax beside it. */
TEST(Unit_MaxPool2d, ForwardReturnsThePooledValueNotTheArgmax)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueId x = builder.input(Shape{2, 8, 6, 3}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

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
    const Graph::ValueId x = builder.input(Shape{2, 16}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws when the window is larger than the padded input. */
TEST(Unit_MaxPool2d, ForwardRejectsAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(5, 5, 1, 1);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws on a rank-4 input tagged NCHW. */
TEST(Unit_MaxPool2d, ForwardRejectsAnNchwInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    EXPECT_THROW((void)layer.forward(builder, nchw), Error);
}

/** @brief Tests that forward() names the layout, not the window, for a narrow NCHW input. */
TEST(Unit_MaxPool2d, ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 8, 8, 2}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    const MaxPool2d layer = make_pool(3, 3, 1, 1);

    SA_EXPECT_ERROR((void)layer.forward(builder, nchw), "NHWC");
}

/** @brief Tests that forward() refuses a zero vertical stride in the layer's own message. */
TEST(Unit_MaxPool2d, ForwardRejectsAZeroStride)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 0, 2);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "MaxPool2d", "stride 0,2");
}

/** @brief Tests that forward() refuses a negative padding in the layer's own message. */
TEST(Unit_MaxPool2d, ForwardRejectsANegativePadding)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2, 0, -1);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");

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

/** @brief Tests that the parameter walk visits nothing, before and after a trace. */
TEST(Unit_MaxPool2d, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MaxPool2d layer = make_pool(2, 2, 2, 2);
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(NN::parameter_count(layer), 0u);
}

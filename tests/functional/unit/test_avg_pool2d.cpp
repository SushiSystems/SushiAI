/**************************************************************************/
/* test_avg_pool2d.cpp                                                    */
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
 * @file test_avg_pool2d.cpp
 * @brief Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using NN::AvgPool2d;

namespace
{
    /** @brief Builds a geometry with the given window and step, undilated and unpadded. */
    Graph::ConvGeometry window(int64_t kernel_h, int64_t kernel_w,
                               int64_t stride_h, int64_t stride_w)
    {
        Graph::ConvGeometry g;
        g.kernel_h = kernel_h;
        g.kernel_w = kernel_w;
        g.stride_h = stride_h;
        g.stride_w = stride_w;
        return g;
    }

    /** @brief Traces @p layer over a batch of two @p sample images and returns the shape. */
    Shape traced_shape(const AvgPool2d& layer, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(Shape{2, sample[0], sample[1], sample[2]}, "x");
        return graph.value(layer.forward(builder, x)).shape;
    }

    /** @brief Drops the batch axis of a rank-4 @p batched shape. */
    Shape per_sample(const Shape& batched)
    {
        return Shape{batched[1], batched[2], batched[3]};
    }
} // namespace

/** @brief Tests that AvgPool2d satisfies the Module trait, through a const reference too. */
TEST(Unit_AvgPool2d, SatisfiesTheModuleTrait)
{
    static_assert(NN::Module<AvgPool2d>, "AvgPool2d must be a Module");
    static_assert(NN::Module<const AvgPool2d>, "AvgPool2d::forward must be const");
    static_assert(!NN::ModuleContainer<AvgPool2d>, "AvgPool2d must not look like a container");
    static_assert(!NN::Detail::HasParameterWalk<AvgPool2d>::value,
                  "AvgPool2d must be enumerated by its fields()");
}

/** @brief Tests that forward() yields the pooled [N, out_h, out_w, C] value tagged NHWC. */
TEST(Unit_AvgPool2d, ForwardTracesThePooledShapeInNhwc)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6, 6, 3}, "x");

    const AvgPool2d layer{window(3, 3, 3, 3)};
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 2, 2, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::Nhwc);
}

/** @brief Tests that output_shape() equals the traced per-sample shape for a square window. */
TEST(Unit_AvgPool2d, OutputShapeAgreesWithTheTracedShapeForASquareWindow)
{
    const AvgPool2d layer{window(3, 3, 3, 3)};
    const Shape sample{6, 6, 3};

    EXPECT_EQ(layer.output_shape(sample), (Shape{2, 2, 3}));
    EXPECT_EQ(layer.output_shape(sample), per_sample(traced_shape(layer, sample)));
}

/** @brief Tests that output_shape() equals the traced shape for a padded, uneven window. */
TEST(Unit_AvgPool2d, OutputShapeAgreesWithTheTracedShapeForAPaddedUnevenWindow)
{
    AvgPool2d layer{window(3, 2, 2, 1)};
    layer.geometry.pad_top = 1;
    layer.geometry.pad_left = 1;
    const Shape sample{7, 5, 4};

    EXPECT_EQ(layer.output_shape(sample), (Shape{4, 6, 4}));
    EXPECT_EQ(layer.output_shape(sample), per_sample(traced_shape(layer, sample)));
}

/** @brief Tests that output_shape() equals the traced shape for a window dilated on one axis. */
TEST(Unit_AvgPool2d, OutputShapeAgreesWithTheTracedShapeForADilatedWindow)
{
    AvgPool2d layer{window(3, 3, 1, 1)};
    layer.geometry.dilation_h = 2;
    layer.geometry.dilation_w = 1;
    const Shape sample{9, 8, 1};

    EXPECT_EQ(layer.output_shape(sample), (Shape{5, 6, 1}));
    EXPECT_EQ(layer.output_shape(sample), per_sample(traced_shape(layer, sample)));
}

/** @brief Tests that output_shape() throws for a per-sample shape that is not rank 3. */
TEST(Unit_AvgPool2d, OutputShapeRejectsAFeatureVector)
{
    const AvgPool2d layer{window(2, 2, 2, 2)};

    EXPECT_THROW((void)layer.output_shape(Shape{8}), Error);
}

/** @brief Tests that output_shape() throws for a shape that still carries the batch axis. */
TEST(Unit_AvgPool2d, OutputShapeRejectsABatchedShape)
{
    const AvgPool2d layer{window(2, 2, 2, 2)};

    EXPECT_THROW((void)layer.output_shape(Shape{2, 4, 4, 3}), Error);
}

/** @brief Tests that output_shape() throws when the window is larger than the image. */
TEST(Unit_AvgPool2d, OutputShapeRejectsAWindowLargerThanTheImage)
{
    const AvgPool2d layer{window(5, 5, 1, 1)};

    EXPECT_THROW((void)layer.output_shape(Shape{4, 4, 3}), Error);
}

/** @brief Tests that output_shape() throws when the geometry has a zero stride. */
TEST(Unit_AvgPool2d, OutputShapeRejectsAZeroStride)
{
    const AvgPool2d layer{window(2, 2, 0, 0)};

    EXPECT_THROW((void)layer.output_shape(Shape{4, 4, 3}), Error);
}

/** @brief Tests that output_shape() names the stride it refuses, not the window's fit. */
TEST(Unit_AvgPool2d, OutputShapeNamesAZeroStride)
{
    const AvgPool2d layer{window(2, 2, 0, 0)};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{4, 4, 3}), "AvgPool2d", "stride 0,0");
}

/** @brief Tests that output_shape() refuses a negative padding and names it. */
TEST(Unit_AvgPool2d, OutputShapeRejectsANegativePadding)
{
    AvgPool2d layer{window(2, 2, 2, 2)};
    layer.geometry.pad_top = -1;

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{4, 4, 3}), "AvgPool2d", "padding -1,0");
}

/** @brief Tests that forward() throws for an input that is not rank 4. */
TEST(Unit_AvgPool2d, ForwardRejectsARankTwoInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8}, "x");

    const AvgPool2d layer{window(2, 2, 2, 2)};

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws for a rank-4 input in NCHW layout. */
TEST(Unit_AvgPool2d, ForwardRejectsAnNchwInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 6, 6, 3}, "x");
    const Graph::ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    const AvgPool2d layer{window(3, 3, 3, 3)};

    EXPECT_THROW((void)layer.forward(builder, x_nchw), Error);
}

/** @brief Tests that forward() names the layout, not the window, for a narrow NCHW input. */
TEST(Unit_AvgPool2d, ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 8, 8, 2}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    const AvgPool2d layer{window(3, 3, 1, 1)};

    SA_EXPECT_ERROR((void)layer.forward(builder, nchw), "NHWC");
}

/** @brief Tests that forward() refuses a zero stride in the layer's own message. */
TEST(Unit_AvgPool2d, ForwardRejectsAZeroStride)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");

    const AvgPool2d layer{window(2, 2, 0, 0)};

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "AvgPool2d", "stride 0,0");
}

/** @brief Tests that forward() refuses a negative padding in the layer's own message. */
TEST(Unit_AvgPool2d, ForwardRejectsANegativePadding)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");

    AvgPool2d layer{window(2, 2, 2, 2)};
    layer.geometry.pad_top = -1;

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "AvgPool2d", "padding -1,0");
}

/** @brief Tests that forward() throws when the window is larger than the image. */
TEST(Unit_AvgPool2d, ForwardRejectsAWindowLargerThanTheImage)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");

    const AvgPool2d layer{window(5, 5, 1, 1)};

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that a traced AvgPool2d has no parameter and adds one value to the graph. */
TEST(Unit_AvgPool2d, ATracedLayerHasNoParameter)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6, 6, 3}, "x");

    AvgPool2d layer{window(3, 3, 3, 3)};
    (void)layer.forward(builder, x);

    int visited = 0;
    NN::for_each_parameter(layer, [&visited](const std::string&, NN::Parameter&) { ++visited; });

    EXPECT_EQ(visited, 0);
    EXPECT_EQ(NN::parameter_count(layer), 0u);
    EXPECT_EQ(graph.value_count(), 2u);
}

/**************************************************************************/
/* test_depthwise_ops.cpp                                                 */
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
 * @file test_depthwise_ops.cpp
 * @brief Tests the shape rules, geometry and rejections of the depthwise convolution emitters.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueId;

namespace
{
    /** @brief The geometry most tests share: a 3x3 window, unit stride and one pixel of padding. */
    Graph::ConvGeometry same_3x3()
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3;
        g.kernel_w = 3;
        g.pad_top = 1;
        g.pad_left = 1;
        return g;
    }

    /** @brief Declares a rank-4 input and converts it to NCHW; the extents survive the permute. */
    ValueId nchw_input(const GraphBuilder& builder, std::string_view debug_name,
                       DType dtype = DType::Float32)
    {
        const ValueId nhwc = builder.input(Shape{2, 5, 5, 3}, debug_name, dtype);
        return builder.layout_convert(nhwc, Graph::TensorLayout::Nchw);
    }

    /** @brief Expects @p out to be the single result of an @p op node that reads @p inputs. */
    void expect_produced_by(const Graph::Graph& graph, ValueId out, OpID op,
                            const std::vector<ValueId>& inputs)
    {
        const Graph::NodeId producer = graph.value(out).producer;
        ASSERT_NE(producer, Graph::K_NO_NODE);

        const Graph::Node& node = graph.node(producer);
        EXPECT_EQ(node.op, op) << "written by " << Ops::name_of(node.op);
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }
}

/** @brief Tests that the forward emitter writes its identity over [input, taps] in that order. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dWritesItsIdentityOverInputAndTapsInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    const ValueId out = builder.depthwise_conv2d(x, w, same_3x3());

    expect_produced_by(graph, out, Ops::K_DEPTHWISE_CONV2D.id, {x, w});
}

/** @brief Tests that the forward emitter yields the window grid over the padded input. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dYieldsTheWindowGridOverThePaddedInput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry unit = same_3x3();
    EXPECT_EQ(graph.value(builder.depthwise_conv2d(x, w, unit)).shape, Shape({2, 5, 5, 3}));

    Graph::ConvGeometry strided = same_3x3();
    strided.stride_h = 2;
    strided.stride_w = 2;
    EXPECT_EQ(graph.value(builder.depthwise_conv2d(x, w, strided)).shape, Shape({2, 3, 3, 3}));
}

/** @brief Tests that the forward emitter keeps the input's layout, element type and lifetime. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dKeepsLayoutDtypeAndLifetime)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float64);
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w", DType::Float64);

    const ValueId out = builder.depthwise_conv2d(x, w, same_3x3());

    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::Nhwc);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests that the forward emitter stores its geometry in the slots MAXPOOL2D also uses. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dStoresItsGeometryInTheGeometrySlots)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 7, 9, 3}, "x");
    const ValueId w = builder.parameter(Shape{3, 5, 3}, "w");

    Graph::ConvGeometry g;
    g.kernel_h = 3;
    g.kernel_w = 5;
    g.pad_top = 1;
    g.pad_left = 2;

    const ValueId out = builder.depthwise_conv2d(x, w, g);

    const Graph::Node& node = graph.node(graph.value(out).producer);
    const double expected[8] = {3.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0};
    for (std::size_t slot = 0; slot < 8; ++slot)
        EXPECT_DOUBLE_EQ(node.attributes.values[slot], expected[slot]) << "slot " << slot;
}

/** @brief Tests that the forward emitter refuses taps of rank four. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dRefusesARankFourWeight)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueId w = builder.parameter(Shape{3, 3, 3, 1}, "w");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.depthwise_conv2d(x, w, same_3x3()), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that the forward emitter refuses taps whose window or channel extent differs. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dRefusesTapsOfAnotherWindowOrChannelCount)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueId tall = builder.parameter(Shape{5, 3, 3}, "tall");
    const ValueId wide = builder.parameter(Shape{3, 5, 3}, "wide");
    const ValueId narrow = builder.parameter(Shape{3, 3, 4}, "narrow");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.depthwise_conv2d(x, tall, same_3x3()), Error);
    EXPECT_THROW((void)builder.depthwise_conv2d(x, wide, same_3x3()), Error);
    EXPECT_THROW((void)builder.depthwise_conv2d(x, narrow, same_3x3()), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that the forward emitter refuses an NCHW input. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dRefusesAnNchwInput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = nchw_input(builder, "x");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    EXPECT_THROW((void)builder.depthwise_conv2d(x, w, same_3x3()), Error);
}

/** @brief Tests that the forward emitter refuses taps of another element type. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dRefusesTapsOfAnotherDtype)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float32);
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w", DType::Float64);

    EXPECT_THROW((void)builder.depthwise_conv2d(x, w, same_3x3()), Error);
}

/** @brief Tests that the forward emitter refuses a window that does not fit the padded input. */
TEST(Unit_DepthwiseOps, DepthwiseConv2dRefusesAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 2, 2, 3}, "x");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry unpadded = same_3x3();
    unpadded.pad_top = 0;
    unpadded.pad_left = 0;

    EXPECT_THROW((void)builder.depthwise_conv2d(x, w, unpadded), Error);
}

/** @brief Tests that the input-gradient emitter writes its identity over [gradient, taps]. */
TEST(Unit_DepthwiseOps, BackwardInputWritesItsIdentityOverGradientAndTapsInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId dy = builder.input(Shape{2, 5, 5, 3}, "dy");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    const ValueId dx =
        builder.depthwise_conv2d_backward_input(dy, w, Shape{2, 5, 5, 3}, same_3x3());

    expect_produced_by(graph, dx, Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id, {dy, w});
}

/** @brief Tests that the input-gradient emitter returns the forward input's shape. */
TEST(Unit_DepthwiseOps, BackwardInputRecoversTheForwardInputShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId dy = builder.input(Shape{2, 3, 3, 3}, "dy");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry strided = same_3x3();
    strided.stride_h = 2;
    strided.stride_w = 2;

    const ValueId dx = builder.depthwise_conv2d_backward_input(dy, w, Shape{2, 5, 5, 3}, strided);

    EXPECT_EQ(graph.value(dx).shape, Shape({2, 5, 5, 3}));
    EXPECT_EQ(graph.value(dx).layout, Graph::TensorLayout::Nhwc);
    EXPECT_EQ(graph.value(dx).dtype, DType::Float32);
}

/** @brief Tests that the input-gradient emitter refuses a gradient on another window grid. */
TEST(Unit_DepthwiseOps, BackwardInputRefusesAGradientOnAnotherGrid)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId short_grid = builder.input(Shape{2, 4, 5, 3}, "short_grid");
    const ValueId wide_channels = builder.input(Shape{2, 5, 5, 4}, "wide_channels");
    const ValueId flat = builder.input(Shape{2, 5, 3}, "flat");
    const ValueId w = builder.parameter(Shape{3, 3, 3}, "w");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.depthwise_conv2d_backward_input(short_grid, w, Shape{2, 5, 5, 3},
                                                               same_3x3()), Error);
    EXPECT_THROW((void)builder.depthwise_conv2d_backward_input(wide_channels, w,
                                                               Shape{2, 5, 5, 3}, same_3x3()),
                 Error);
    EXPECT_THROW((void)builder.depthwise_conv2d_backward_input(flat, w, Shape{2, 5, 5, 3},
                                                               same_3x3()), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that the tap-gradient emitter writes its identity over [gradient, input]. */
TEST(Unit_DepthwiseOps, BackwardWeightWritesItsIdentityOverGradientAndInputInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId dy = builder.input(Shape{2, 5, 5, 3}, "dy");
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x");

    const ValueId dw = builder.depthwise_conv2d_backward_weight(dy, x, same_3x3());

    expect_produced_by(graph, dw, Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id, {dy, x});
}

/** @brief Tests that the tap-gradient emitter yields a [kernel_h, kernel_w, C] gradient. */
TEST(Unit_DepthwiseOps, BackwardWeightYieldsTheTapShapeInTheOperandsDtype)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId dy = builder.input(Shape{2, 5, 5, 3}, "dy", DType::Float64);
    const ValueId x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float64);

    const ValueId dw = builder.depthwise_conv2d_backward_weight(dy, x, same_3x3());

    EXPECT_EQ(graph.value(dw).shape, Shape({3, 3, 3}));
    EXPECT_EQ(graph.value(dw).layout, Graph::TensorLayout::Nhwc);
    EXPECT_EQ(graph.value(dw).dtype, DType::Float64);
    EXPECT_EQ(graph.value(dw).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests that the tap-gradient emitter refuses operands that disagree in batch or depth. */
TEST(Unit_DepthwiseOps, BackwardWeightRefusesMismatchedBatchChannelsOrDtype)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId dy = builder.input(Shape{2, 5, 5, 3}, "dy");
    const ValueId other_batch = builder.input(Shape{1, 5, 5, 3}, "other_batch");
    const ValueId other_channels = builder.input(Shape{2, 5, 5, 4}, "other_channels");
    const ValueId wide = builder.input(Shape{2, 5, 5, 3}, "wide", DType::Float64);
    const ValueId rank_three = builder.input(Shape{2, 5, 3}, "rank_three");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.depthwise_conv2d_backward_weight(dy, other_batch, same_3x3()),
                 Error);
    EXPECT_THROW((void)builder.depthwise_conv2d_backward_weight(dy, other_channels, same_3x3()),
                 Error);
    EXPECT_THROW((void)builder.depthwise_conv2d_backward_weight(dy, wide, same_3x3()), Error);
    EXPECT_THROW((void)builder.depthwise_conv2d_backward_weight(rank_three, dy, same_3x3()),
                 Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

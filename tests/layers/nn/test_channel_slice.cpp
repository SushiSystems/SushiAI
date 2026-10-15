/**************************************************************************/
/* test_channel_slice.cpp                                                 */
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
 * @file test_channel_slice.cpp
 * @brief Tests the shape, layout and rejections of the ChannelSlice layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/channel_slice.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::ChannelSlice;

static_assert(NN::Module<ChannelSlice>, "ChannelSlice must satisfy the Module contract");
static_assert(!NN::ModuleContainer<ChannelSlice>, "ChannelSlice is a leaf, not a container");

namespace
{
    /** @brief Returns the traced shape of @p layer's result with its batch axis dropped. */
    Shape traced_per_sample_shape(const ChannelSlice& layer, const Shape& batched_input)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(batched_input, "x");
        const Shape traced = graph.value(layer.forward(builder, x)).shape;
        return Shape{traced[1], traced[2], traced[3]};
    }
} // namespace

/** @brief Tests that the offset and count fields are the two values an aggregate list sets. */
TEST(Small_ChannelSlice, TheOffsetAndCountFieldsAreSetInOrder)
{
    const ChannelSlice layer{32, 16};

    EXPECT_EQ(layer.offset, 32);
    EXPECT_EQ(layer.count, 16);
}

/** @brief Tests the acceptance case: [H, W, 64] with offset 32 and count 32 gives [H, W, 32]. */
TEST(Small_ChannelSlice, OutputShapeKeepsHeightAndWidthAndTakesTheRange)
{
    const ChannelSlice layer{32, 32};

    EXPECT_EQ(layer.output_shape(Shape{5, 3, 64}), (Shape{5, 3, 32}));
}

/** @brief Tests that the first channel, the last channel and the whole range are all accepted. */
TEST(Small_ChannelSlice, OutputShapeAcceptsTheFirstLastAndWholeRanges)
{
    EXPECT_EQ((ChannelSlice{0, 1}.output_shape(Shape{4, 2, 4})), (Shape{4, 2, 1}));
    EXPECT_EQ((ChannelSlice{3, 1}.output_shape(Shape{4, 2, 4})), (Shape{4, 2, 1}));
    EXPECT_EQ((ChannelSlice{1, 3}.output_shape(Shape{4, 2, 4})), (Shape{4, 2, 3}));
    EXPECT_EQ((ChannelSlice{0, 4}.output_shape(Shape{4, 2, 4})), (Shape{4, 2, 4}));
}

/** @brief Tests that the traced result carries the NHWC layout tag. */
TEST(Small_ChannelSlice, TheResultIsTaggedNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{32, 32};
    const Graph::ValueID x = builder.input(Shape{2, 5, 3, 64}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/**
 * @brief Tests the acceptance case at trace level: one K_SLICE_CHANNELS node reads x and
 * carries the offset, and the result is [N, H, W, 32].
 */
TEST(Small_ChannelSlice, ForwardEmitsOneSliceNodeCarryingTheOffset)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{32, 32};
    const Graph::ValueID x = builder.input(Shape{2, 5, 3, 64}, "x");
    const std::size_t nodes_before = graph.node_count();
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_SLICE_CHANNELS.id);
    EXPECT_EQ(node.inputs, (std::vector<Graph::ValueID>{x}));
    EXPECT_DOUBLE_EQ(node.attributes.get(0), 32.0);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, 3, 32}));
}

/** @brief Tests that a range covering every channel traces to the input's own shape. */
TEST(Small_ChannelSlice, TheWholeChannelRangeTracesToTheInputShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{0, 64};
    const Graph::ValueID x = builder.input(Shape{2, 5, 3, 64}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, 3, 64}));
}

/** @brief Tests that output_shape equals the traced shape without its batch axis. */
TEST(Small_ChannelSlice, OutputShapeAgreesWithTheTracedShape)
{
    const ChannelSlice acceptance{32, 32};
    EXPECT_EQ(acceptance.output_shape(Shape{5, 3, 64}),
              traced_per_sample_shape(acceptance, Shape{2, 5, 3, 64}));

    const ChannelSlice leading{1, 2};
    EXPECT_EQ(leading.output_shape(Shape{4, 4, 6}),
              traced_per_sample_shape(leading, Shape{1, 4, 4, 6}));
}

/** @brief Tests that output_shape rejects a negative offset in a message naming the values. */
TEST(Small_ChannelSlice, OutputShapeRejectsANegativeOffset)
{
    const ChannelSlice layer{-1, 4};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "offset=-1, count=4", "64 channels");
}

/** @brief Tests that output_shape rejects a zero count in a message naming the values. */
TEST(Small_ChannelSlice, OutputShapeRejectsAZeroCount)
{
    const ChannelSlice layer{0, 0};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "offset=0, count=0", "64 channels");
}

/** @brief Tests that output_shape rejects a negative count in a message naming the values. */
TEST(Small_ChannelSlice, OutputShapeRejectsANegativeCount)
{
    const ChannelSlice layer{8, -2};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "offset=8, count=-2", "64 channels");
}

/** @brief Tests that output_shape rejects a range that runs past the channels. */
TEST(Small_ChannelSlice, OutputShapeRejectsARangePastTheChannels)
{
    const ChannelSlice layer{32, 40};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "offset=32, count=40", "64 channels");
}

/** @brief Tests that output_shape rejects an offset at the channel count, even with count 1. */
TEST(Small_ChannelSlice, OutputShapeRejectsAnOffsetAtTheChannelCount)
{
    const ChannelSlice layer{64, 1};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "offset=64, count=1", "64 channels");
}

/** @brief Tests that extreme offsets and counts are refused without overflowing the range test. */
TEST(Small_ChannelSlice, OutputShapeRejectsExtremeValuesWithoutOverflow)
{
    const int64_t largest = std::numeric_limits<int64_t>::max();
    const ChannelSlice far_offset{largest, 1};
    const ChannelSlice far_count{1, largest};

    SA_EXPECT_ERROR((void)far_offset.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "64 channels");
    SA_EXPECT_ERROR((void)far_count.output_shape(Shape{5, 3, 64}), "ChannelSlice",
                    "64 channels");
}

/** @brief Tests that output_shape rejects a feature vector in a message naming layer and shape. */
TEST(Small_ChannelSlice, OutputShapeRejectsARankOnePerSampleShape)
{
    const ChannelSlice layer{0, 1};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{16}), "ChannelSlice", "[H, W, C]", "[16]");
}

/** @brief Tests that output_shape throws for a shape that still carries its batch axis. */
TEST(Small_ChannelSlice, OutputShapeRejectsARankFourShape)
{
    const ChannelSlice layer{0, 1};

    EXPECT_THROW((void)layer.output_shape(Shape{2, 3, 5, 7}), Error);
}

/** @brief Tests that forward rejects a rank-2 input in a message naming layer and shape. */
TEST(Small_ChannelSlice, ForwardRejectsARankTwoInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{0, 1};
    const Graph::ValueID x = builder.input(Shape{2, 12}, "x");

    SA_EXPECT_ERROR((void)layer.forward(builder, x), "ChannelSlice", "rank-4", "[2, 12]");
}

/** @brief Tests that forward rejects a range past the channels in the layer's own message. */
TEST(Small_ChannelSlice, ForwardRejectsARangePastTheChannels)
{
    const ChannelSlice layer{2, 2};

    SA_EXPECT_ERROR((void)traced_per_sample_shape(layer, Shape{1, 2, 2, 3}), "ChannelSlice",
                    "offset=2, count=2", "3 channels");
}

/** @brief Tests that forward names the layout it reads for a rank-4 input tagged NCHW. */
TEST(Small_ChannelSlice, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{0, 1};
    const Graph::ValueID x = builder.input(Shape{1, 2, 2, 3}, "x");
    const Graph::ValueID x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    SA_EXPECT_ERROR((void)layer.forward(builder, x_nchw), "slice_channels", "NHWC");
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Small_ChannelSlice, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    ChannelSlice layer{0, 1};
    const Graph::ValueID x = builder.input(Shape{1, 2, 2, 3}, "x");
    (void)layer.forward(builder, x);

    std::size_t visited = 0;
    NN::for_each_parameter(layer, [&visited](const std::string&, NN::Parameter&)
    {
        ++visited;
    });

    EXPECT_EQ(visited, 0u);
    EXPECT_EQ(NN::parameter_count(layer), 0u);
}

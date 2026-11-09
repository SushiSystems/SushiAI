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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
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
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        const Shape traced = graph.value(layer.forward(builder, x).value()).shape;
        return Shape{traced[1], traced[2], traced[3]};
    }

} // namespace

/** @brief Holds a channel range, the per-sample shape it slices and the shape it gives. */
struct AcceptedRangeRow
{
    std::string name;  /**< The row's name. */
    int64_t offset;    /**< The first channel taken. */
    int64_t count;     /**< The number of channels taken. */
    Shape sample;      /**< The per-sample input shape. */
    Shape expected;    /**< The per-sample shape the layer gives. */
};

/** @brief Runs the accepted-range promise over each kind of range. */
class Small_ChannelSliceAccepted : public ::testing::TestWithParam<AcceptedRangeRow>
{
};

/** @brief Tests that output_shape keeps height and width and takes the range of channels. */
TEST_P(Small_ChannelSliceAccepted, OutputShapeTakesTheRange)
{
    const AcceptedRangeRow& row = GetParam();
    const ChannelSlice layer{row.offset, row.count};

    EXPECT_EQ(layer.output_shape(row.sample).value(), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ChannelSliceAccepted,
    ::testing::Values(
        AcceptedRangeRow{"AMiddleRange", 32, 32, Shape{5, 3, 64}, Shape{5, 3, 32}},
        AcceptedRangeRow{"TheFirstChannel", 0, 1, Shape{4, 2, 4}, Shape{4, 2, 1}},
        AcceptedRangeRow{"TheLastChannel", 3, 1, Shape{4, 2, 4}, Shape{4, 2, 1}},
        AcceptedRangeRow{"AllButTheFirst", 1, 3, Shape{4, 2, 4}, Shape{4, 2, 3}},
        AcceptedRangeRow{"TheWholeRange", 0, 4, Shape{4, 2, 4}, Shape{4, 2, 4}}),
    [](const ::testing::TestParamInfo<AcceptedRangeRow>& info) { return info.param.name; });

/** @brief Tests that the traced result carries the NHWC layout tag. */
TEST(Small_ChannelSlice, TheResultIsTaggedNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const ChannelSlice layer{32, 32};
    const Graph::ValueID x = builder.input(Shape{2, 5, 3, 64}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

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
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

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
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, 3, 64}));
}

/** @brief Tests that output_shape equals the traced shape without its batch axis. */
TEST(Small_ChannelSlice, OutputShapeAgreesWithTheTracedShape)
{
    const ChannelSlice acceptance{32, 32};
    EXPECT_EQ(acceptance.output_shape(Shape{5, 3, 64}).value(),
              traced_per_sample_shape(acceptance, Shape{2, 5, 3, 64}));

    const ChannelSlice leading{1, 2};
    EXPECT_EQ(leading.output_shape(Shape{4, 4, 6}).value(),
              traced_per_sample_shape(leading, Shape{1, 4, 4, 6}));
}

/** @brief Holds a range and a per-sample shape that output_shape refuses, with the message. */
struct RefusedRangeRow
{
    std::string name;   /**< The row's name. */
    int64_t offset;     /**< The first channel taken. */
    int64_t count;      /**< The number of channels taken. */
    Shape sample;       /**< The shape given to output_shape. */
    std::string first;  /**< A substring the message holds; empty for none. */
    std::string second; /**< A second substring the message holds; empty for none. */
    std::string third;  /**< A third substring the message holds; empty for none. */
};

/** @brief Runs the output_shape refusal over each range and shape it cannot take. */
class Small_ChannelSliceRefused : public ::testing::TestWithParam<RefusedRangeRow>
{
};

/** @brief Tests that output_shape refuses the row and names the values the row says. */
TEST_P(Small_ChannelSliceRefused, OutputShapeRefuses)
{
    const RefusedRangeRow& row = GetParam();
    const ChannelSlice layer{row.offset, row.count};

    SA_EXPECT_FAILURE(layer.output_shape(row.sample), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ChannelSliceRefused,
    ::testing::Values(
        RefusedRangeRow{"ANegativeOffset", -1, 4, Shape{5, 3, 64}, "ChannelSlice",
                        "offset=-1, count=4", "64 channels"},
        RefusedRangeRow{"AZeroCount", 0, 0, Shape{5, 3, 64}, "ChannelSlice", "offset=0, count=0",
                        "64 channels"},
        RefusedRangeRow{"ANegativeCount", 8, -2, Shape{5, 3, 64}, "ChannelSlice",
                        "offset=8, count=-2", "64 channels"},
        RefusedRangeRow{"ARangePastTheChannels", 32, 40, Shape{5, 3, 64}, "ChannelSlice",
                        "offset=32, count=40", "64 channels"},
        RefusedRangeRow{"AnOffsetAtTheChannelCount", 64, 1, Shape{5, 3, 64}, "ChannelSlice",
                        "offset=64, count=1", "64 channels"},
        RefusedRangeRow{"AnExtremeOffsetWithoutOverflow", std::numeric_limits<int64_t>::max(), 1,
                        Shape{5, 3, 64}, "ChannelSlice", "64 channels", ""},
        RefusedRangeRow{"AnExtremeCountWithoutOverflow", 1, std::numeric_limits<int64_t>::max(),
                        Shape{5, 3, 64}, "ChannelSlice", "64 channels", ""},
        RefusedRangeRow{"ARankOnePerSampleShape", 0, 1, Shape{16}, "ChannelSlice", "[H, W, C]",
                        "[16]"},
        RefusedRangeRow{"ARankFourShape", 0, 1, Shape{2, 3, 5, 7}, "", "", ""}),
    [](const ::testing::TestParamInfo<RefusedRangeRow>& info) { return info.param.name; });

/** @brief Holds a range, a batched input, whether it is NCHW, and the refusal forward meets. */
struct ForwardRefusalRow
{
    std::string name;   /**< The row's name. */
    int64_t offset;     /**< The first channel taken. */
    int64_t count;      /**< The number of channels taken. */
    Shape input;        /**< The input shape, batch axis included. */
    bool nchw;          /**< Whether the input is converted to NCHW first. */
    std::string first;  /**< A substring the message holds. */
    std::string second; /**< A second substring the message holds. */
    std::string third;  /**< A third substring the message holds; empty for none. */
};

/** @brief Runs the forward refusal over each input and range it cannot take. */
class Small_ChannelSliceForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};

/** @brief Tests that forward refuses the row and names what the row says. */
TEST_P(Small_ChannelSliceForwardRefusal, ForwardRefuses)
{
    const ForwardRefusalRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Graph::ValueID x = builder.input(row.input, "x");
    if (row.nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }
    const ChannelSlice layer{row.offset, row.count};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ChannelSliceForwardRefusal,
    ::testing::Values(
        ForwardRefusalRow{"ARankTwoInput", 0, 1, Shape{2, 12}, false, "ChannelSlice", "rank-4",
                          "[2, 12]"},
        ForwardRefusalRow{"ARangePastTheChannels", 2, 2, Shape{1, 2, 2, 3}, false, "ChannelSlice",
                          "offset=2, count=2", "3 channels"},
        ForwardRefusalRow{"AnNCHWInput", 0, 1, Shape{1, 2, 2, 3}, true, "slice_channels", "NHWC",
                          ""}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

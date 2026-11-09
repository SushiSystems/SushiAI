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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
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
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(layer.forward(builder, x).value()).shape;
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
TEST(Small_DepthwiseConv2d, ForwardEmitsOneDepthwiseNodeAndNoBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(count_op(graph, Ops::K_DEPTHWISE_CONV2D.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
}

/** @brief Tests that the emitted node reads the input and the [3, 3, 4] taps. */
TEST(Small_DepthwiseConv2d, TheEmittedNodeReadsTheInputAndTheWeight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    const Graph::Node* node = first_node(graph, Ops::K_DEPTHWISE_CONV2D.id);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->inputs.size(), 2u);
    EXPECT_EQ(node->inputs[0], x);
    EXPECT_EQ(graph.value(node->inputs[1]).shape, (Shape{3, 3, 4}));
    ASSERT_EQ(node->outputs.size(), 1u);
    EXPECT_EQ(node->outputs[0], y);
}

/** @brief Tests that a padded 3x3 stride-1 convolution keeps the spatial extents and tags NHWC. */
TEST(Small_DepthwiseConv2d, ForwardKeepsTheSpatialExtentsAndTagsNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 8, 4}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that a stride-2 convolution over an even extent halves both spatial extents. */
TEST(Small_DepthwiseConv2d, StrideTwoHalvesAnEvenExtent)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 2, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 3, 4}));
}

/** @brief Tests that the weight is [kernel_h, kernel_w, channels] under an anisotropic window. */
TEST(Small_DepthwiseConv2d, TheWeightOfAnAnisotropicWindowIsHeightWidthChannels)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(3, 2);
    layer.geometry.kernel_w = 4;
    const Graph::ValueID x = builder.input(Shape{2, 6, 9, 3}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.shape(), (Shape{2, 4, 3}));
}

/** @brief Tests that the walk yields the weight alone, and that it is empty before forward(). */
TEST(Small_DepthwiseConv2d, TheParameterWalkNamesOnlyTheWeight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    EXPECT_TRUE(walk(layer).empty());
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 3, 4]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 3 * 3 * 4);
}

/** @brief Tests that the weight is declared with the WEIGHT role. */
TEST(Small_DepthwiseConv2d, TheWeightRecordsTheWeightRole)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Weight);
}

/** @brief Tests that the weight records its window as both fans, since each channel is its own. */
TEST(Small_DepthwiseConv2d, TheWeightRecordsItsWindowAsBothFans)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(4, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.fan_in(), 3 * 3);
    EXPECT_EQ(layer.weight.fan_out(), 3 * 3);
}

/** @brief Tests that a FLOAT64 input gives a FLOAT64 weight and a FLOAT64 result. */
TEST(Small_DepthwiseConv2d, TheWeightAndTheResultTakeTheInputElementType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    DepthwiseConv2d layer = make_layer(2, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
}

namespace
{
/** @brief Holds a layer, a batch, a sample and the per-sample shape the layer gives. */
struct ExtentRow
{
    std::string name;      /**< The row's name. */
    DepthwiseConv2d layer; /**< The layer under test. */
    int64_t batch;         /**< The batch extent of the trace. */
    Shape sample;          /**< The per-sample input shape. */
    Shape expected;        /**< The per-sample shape the layer gives. */
};
} // namespace

namespace
{
/** @brief Runs the extent promise over each stride and sample. */
class Small_DepthwiseConv2dExtents : public ::testing::TestWithParam<ExtentRow>
{
};
} // namespace

/** @brief Tests that output_shape gives the expected extents and agrees with the traced shape. */
TEST_P(Small_DepthwiseConv2dExtents, OutputShapeAgreesWithForward)
{
    const ExtentRow& row = GetParam();

    EXPECT_EQ(row.layer.output_shape(row.sample).value(), row.expected);
    EXPECT_EQ(traced_shape(row.layer, row.batch, row.sample), with_batch(row.batch, row.expected));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DepthwiseConv2dExtents,
    ::testing::Values(
        ExtentRow{"StrideOneKeepsTheExtents", make_layer(4, 3, 1, 1), 3, Shape{8, 6, 4},
                  Shape{8, 6, 4}},
        ExtentRow{"StrideTwoHalvesTheExtents", make_layer(4, 3, 2, 1), 3, Shape{8, 6, 4},
                  Shape{4, 3, 4}},
        ExtentRow{"StrideTwoDropsTheRemainderOfAnOddExtent", make_layer(4, 3, 2, 1), 2,
                  Shape{7, 9, 4}, Shape{4, 5, 4}}),
    [](const ::testing::TestParamInfo<ExtentRow>& info) { return info.param.name; });

/** @brief Tests that same_padded() sets the kernel, the stride, kernel / 2 pads and dilation 1. */
TEST(Small_DepthwiseConv2d, SamePaddedBuildsAPaddedLayerOfTheGivenStride)
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

namespace
{
/** @brief Holds the channel counts check_depthwise_channels() is asked about. */
struct ChannelCountRow
{
    std::string name;     /**< The row's name. */
    int64_t in_channels;  /**< The channel count entering the layer. */
    int64_t out_channels; /**< The channel count asked of the layer. */
};
} // namespace

namespace
{
/** @brief Runs the channel count check over each pair it refuses. */
class Small_DepthwiseConv2dChannelCheck : public ::testing::TestWithParam<ChannelCountRow>
{
};
} // namespace

/** @brief Tests that the check refuses a pair of unequal counts and names both. */
TEST_P(Small_DepthwiseConv2dChannelCheck, UnequalCountsAreRefused)
{
    const ChannelCountRow& row = GetParam();

    SA_EXPECT_FAILURE(NN::check_depthwise_channels(row.in_channels, row.out_channels),
                      StatusCode::InvalidDescription, "keeps its channel count",
                      "in=" + std::to_string(row.in_channels),
                      "out=" + std::to_string(row.out_channels));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DepthwiseConv2dChannelCheck,
    ::testing::Values(ChannelCountRow{"FewerOut", 8, 4}, ChannelCountRow{"MoreOut", 4, 8}),
    [](const ::testing::TestParamInfo<ChannelCountRow>& info) { return info.param.name; });

/** @brief Tests that the check accepts equal counts. */
TEST(Small_DepthwiseConv2d, EqualChannelCountsAreAccepted)
{
    SA_EXPECT_OK(NN::check_depthwise_channels(4, 4));
}

/** @brief Tests that same_padded() stops the process when the channel counts differ. */
TEST(Medium_DepthwiseConv2dDeathTest, SamePaddedRefusesUnequalChannelCounts)
{
    SA_EXPECT_FATAL((void)DepthwiseConv2d::same_padded(4, 8, 3, 1), "keeps its channel count");
}

namespace
{
/** @brief Holds a layer, a batched input, whether it is NCHW, and the refusal forward meets. */
struct ForwardRefusalRow
{
    std::string name;      /**< The row's name. */
    DepthwiseConv2d layer; /**< The layer under test. */
    Shape input;           /**< The input shape, batch axis included. */
    bool nchw;             /**< Whether the input is converted to NCHW first. */
    std::string message;   /**< A substring the message holds; empty for none. */
};
} // namespace

namespace
{
/** @brief Runs the forward refusal over each layer and input it cannot take. */
class Small_DepthwiseConv2dForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};
} // namespace

/** @brief Tests that forward refuses the row and names what the row says. */
TEST_P(Small_DepthwiseConv2dForwardRefusal, ForwardRefuses)
{
    const ForwardRefusalRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    DepthwiseConv2d layer = row.layer;
    Graph::ValueID x = builder.input(row.input, "x");
    if (row.nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DepthwiseConv2dForwardRefusal,
    ::testing::Values(
        ForwardRefusalRow{"AnInputThatIsNotRankFour", make_layer(4, 3, 1, 1), Shape{2, 16}, false,
                          "rank-4"},
        ForwardRefusalRow{"AnNCHWInput", make_layer(4, 3, 1, 1), Shape{2, 8, 8, 4}, true, "NHWC"},
        ForwardRefusalRow{"AChannelCountThatDiffersFromTheLayer", make_layer(4, 3, 1, 1),
                          Shape{2, 8, 8, 3}, false, "[H, W, 4]"},
        ForwardRefusalRow{"ANonPositiveChannelCount", make_layer(0, 3, 1, 1), Shape{2, 8, 8, 4},
                          false, "positive channel count"},
        ForwardRefusalRow{"AWindowThatDoesNotFit", make_layer(4, 5, 1, 0), Shape{2, 4, 4, 4},
                          false, ""}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

namespace
{
/** @brief Holds a per-sample shape that output_shape refuses. */
struct ShapeRefusalRow
{
    std::string name; /**< The row's name. */
    Shape sample;     /**< The shape given to output_shape of a 4-channel layer. */
};
} // namespace

namespace
{
/** @brief Runs the output_shape refusal over each sample it cannot take. */
class Small_DepthwiseConv2dShapeRefusal : public ::testing::TestWithParam<ShapeRefusalRow>
{
};
} // namespace

/** @brief Tests that output_shape refuses the sample and names the shape the layer takes. */
TEST_P(Small_DepthwiseConv2dShapeRefusal, OutputShapeRefuses)
{
    const DepthwiseConv2d layer = make_layer(4, 3, 1, 1);

    SA_EXPECT_FAILURE(layer.output_shape(GetParam().sample), StatusCode::InvalidDescription,
                      "[H, W, 4]");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DepthwiseConv2dShapeRefusal,
    ::testing::Values(ShapeRefusalRow{"AChannelCountThatDiffersFromTheLayer", Shape{8, 8, 5}},
                      ShapeRefusalRow{"ASampleOfTheWrongRank", Shape{8, 8}}),
    [](const ::testing::TestParamInfo<ShapeRefusalRow>& info) { return info.param.name; });

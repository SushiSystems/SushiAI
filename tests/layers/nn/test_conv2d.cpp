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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>

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

    /** @brief Returns @p layer with the dilation @p h by @p w. */
    Conv2d dilated(Conv2d layer, int64_t h, int64_t w)
    {
        layer.geometry.dilation_h = h;
        layer.geometry.dilation_w = w;
        return layer;
    }

    /** @brief Checks that @p gradient exists and has the shape of @p parameter. */
    void expect_gradient_of(const Graph::Graph& graph, Graph::ValueID gradient,
                            Graph::ValueID parameter)
    {
        ASSERT_NE(gradient, Graph::K_NO_VALUE);
        EXPECT_EQ(graph.value(gradient).shape, graph.value(parameter).shape);
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

        const Graph::ValueID x = builder.input(with_batch(batch, sample), "x");
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(layer.forward(builder, x).value()).shape;
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
TEST(Small_Conv2d, ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 28, 28, 8}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that a FLOAT64 input gives FLOAT64 parameters and a FLOAT64 result. */
TEST(Small_Conv2d, TheParametersAndTheResultTakeTheInputElementType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(layer.bias.dtype(), DType::Float64);
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
}

/** @brief Tests that the walk yields weight [8, 3, 3, 1] then bias [8] under Linear's names. */
TEST(Small_Conv2d, TheParameterWalkNamesTheWeightAndTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");
    EXPECT_TRUE(walk(layer).empty());
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[8, 3, 3, 1]", "bias:[8]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 8 * 3 * 3 * 1 + 8);
}

/** @brief Tests that the weight's extents follow an anisotropic window as [out, kh, kw, in]. */
TEST(Small_Conv2d, TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(3, 5, 1);
    layer.geometry.kernel_h = 2;
    layer.geometry.kernel_w = 4;
    const Graph::ValueID x = builder.input(Shape{2, 6, 9, 3}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.shape(), (Shape{5, 2, 4, 3}));
}

/** @brief Tests that a layer with a bias traces exactly one BIAS_ADD node. */
TEST(Small_Conv2d, ALayerWithABiasTracesOneBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 1u);
}

/** @brief Tests that use_bias = false yields the weight alone and traces no BIAS_ADD node. */
TEST(Small_Conv2d, ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    layer.use_bias = false;
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[8, 3, 3, 1]"}));
    EXPECT_FALSE(layer.bias.declared());
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 28, 28, 8}));
}

/** @brief Tests that the weight and the bias record fan_in = kh*kw*in and fan_out = kh*kw*out. */
TEST(Small_Conv2d, TheWeightAndTheBiasRecordTheirFans)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(4, 6, 3, 1, 1);
    layer.geometry.kernel_w = 5;
    layer.geometry.pad_left = 2;
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 4}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.fan_in(), 3 * 5 * 4);
    EXPECT_EQ(layer.weight.fan_out(), 3 * 5 * 6);
    EXPECT_EQ(layer.bias.fan_in(), 3 * 5 * 4);
    EXPECT_EQ(layer.bias.fan_out(), 3 * 5 * 6);
}

/** @brief Tests that the weight is declared as a WEIGHT and the bias as a BIAS. */
TEST(Small_Conv2d, TheParametersRecordTheirRoles)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Weight);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::Bias);
}

/** @brief Tests that a second trace reuses the parameters the first one declared. */
TEST(Small_Conv2d, TracingTwiceDoesNotDuplicateParameters)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(1, 8, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 28, 28, 1}, "x");

    SA_EXPECT_OK(layer.forward(builder, x));
    const Graph::ValueID first_weight = layer.weight.id_in(graph);
    const Graph::ValueID first_bias = layer.bias.id_in(graph);

    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.id_in(graph), first_weight);
    EXPECT_EQ(layer.bias.id_in(graph), first_bias);
    EXPECT_EQ(NN::parameter_count(layer), 2u);
}

/** @brief Holds a layer, a batch, a sample and the per-sample shape the layer gives. */
struct AgreementRow
{
    std::string name; /**< The row's name. */
    Conv2d layer;     /**< The layer under test. */
    int64_t batch;    /**< The batch extent of the trace. */
    Shape sample;     /**< The per-sample input shape. */
    Shape expected;   /**< The per-sample shape the layer gives. */
};

/** @brief Runs the shape agreement over each kind of window. */
class Small_Conv2dShape : public ::testing::TestWithParam<AgreementRow>
{
};

/** @brief Tests that output_shape equals the expected shape and the traced shape. */
TEST_P(Small_Conv2dShape, OutputShapeAgreesWithForward)
{
    const AgreementRow& row = GetParam();

    EXPECT_EQ(row.layer.output_shape(row.sample).value(), row.expected);
    EXPECT_EQ(traced_shape(row.layer, row.batch, row.sample), with_batch(row.batch, row.expected));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_Conv2dShape,
    ::testing::Values(
        AgreementRow{"AStridedWindow", make_conv(3, 16, 3, 2), 2, Shape{9, 12, 3},
                     Shape{4, 5, 16}},
        AgreementRow{"APaddedWindow", make_conv(2, 4, 5, 1, 2), 3, Shape{7, 10, 2},
                     Shape{7, 10, 4}},
        AgreementRow{"ADilatedStridedWindow", dilated(make_conv(2, 3, 3, 2, 1), 2, 3), 2,
                     Shape{11, 13, 2}, Shape{5, 5, 3}},
        AgreementRow{"APointwiseWindow", make_conv(4, 6, 1), 2, Shape{5, 7, 4}, Shape{5, 7, 6}},
        AgreementRow{"SamePaddedAtStrideOne", Conv2d::same_padded(4, 4, 3, 1), 2, Shape{7, 10, 4},
                     Shape{7, 10, 4}},
        AgreementRow{"SamePaddedPointwise", Conv2d::same_padded(2, 8, 1, 1), 2, Shape{9, 6, 2},
                     Shape{9, 6, 8}},
        AgreementRow{"SamePaddedFiveByFive", Conv2d::same_padded(2, 8, 5, 1), 2, Shape{9, 6, 2},
                     Shape{9, 6, 8}}),
    [](const ::testing::TestParamInfo<AgreementRow>& info) { return info.param.name; });

/** @brief Tests that same_padded() pads by kernel / 2, sets one stride and drops the bias. */
TEST(Small_Conv2d, SamePaddedPadsByHalfTheKernelAndDropsTheBias)
{
    const Conv2d layer = Conv2d::same_padded(3, 16, 5, 2);

    EXPECT_EQ(layer.in_channels, 3);
    EXPECT_EQ(layer.out_channels, 16);
    EXPECT_EQ(layer.geometry.kernel_h, 5);
    EXPECT_EQ(layer.geometry.kernel_w, 5);
    EXPECT_EQ(layer.geometry.stride_h, 2);
    EXPECT_EQ(layer.geometry.stride_w, 2);
    EXPECT_EQ(layer.geometry.pad_top, 2);
    EXPECT_EQ(layer.geometry.pad_left, 2);
    EXPECT_EQ(layer.geometry.dilation_h, 1);
    EXPECT_EQ(layer.geometry.dilation_w, 1);
    EXPECT_FALSE(layer.use_bias);
}

/** @brief Tests that same_padded() with stride 2 halves an even H and an even W. */
TEST(Small_Conv2d, SamePaddedAtStrideTwoHalvesAnEvenExtent)
{
    const Conv2d layer = Conv2d::same_padded(3, 16, 3, 2);
    const Shape sample{640, 320, 3};

    EXPECT_EQ(layer.output_shape(sample).value(), (Shape{320, 160, 16}));
    EXPECT_EQ(traced_shape(layer, 1, sample), (Shape{1, 320, 160, 16}));
}

/** @brief Tests that a same_padded() layer traces a weight alone and no BIAS_ADD node. */
TEST(Small_Conv2d, SamePaddedTracesNoBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = Conv2d::same_padded(2, 8, 3, 1);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[8, 3, 3, 2]"}));
    EXPECT_FALSE(layer.bias.declared());
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
}

/** @brief Tests that a non-zero bias_init reaches the bias parameter's init() as a constant. */
TEST(Small_Conv2d, ANonZeroBiasInitReachesTheBiasParameter)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    layer.bias_init = -4.5;
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    // -4.5 is exact in binary, so the stored value compares equal with no tolerance.
    EXPECT_EQ(layer.bias.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_EQ(layer.bias.init().constant, -4.5);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::Bias);
}

/** @brief Tests that a layer left at the default starts its bias from the constant zero. */
TEST(Small_Conv2d, TheDefaultBiasInitIsTheConstantZero)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.bias.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_EQ(layer.bias.init().constant, 0.0);
}

/** @brief Tests that bias_init leaves the weight on its fan-scaled fill. */
TEST(Small_Conv2d, BiasInitLeavesTheWeightFanScaled)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    layer.bias_init = 1.25;
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 2}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(layer.weight.init().kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Weight);
}

/** @brief Holds a layer and a per-sample shape that output_shape refuses, with the message. */
struct ShapeRefusalRow
{
    std::string name;   /**< The row's name. */
    Conv2d layer;       /**< The layer under test. */
    Shape sample;       /**< The shape given to output_shape. */
    std::string first;  /**< A substring the message holds. */
    std::string second; /**< A second substring the message holds. */
    std::string third;  /**< A third substring the message holds; empty for none. */
};

/** @brief Runs the output_shape refusal over each layer and shape it cannot take. */
class Small_Conv2dShapeRefusal : public ::testing::TestWithParam<ShapeRefusalRow>
{
};

/** @brief Tests that output_shape refuses the row and names the layer and what the row says. */
TEST_P(Small_Conv2dShapeRefusal, OutputShapeRefuses)
{
    const ShapeRefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(row.layer.output_shape(row.sample), StatusCode::InvalidDescription,
                      row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_Conv2dShapeRefusal,
    ::testing::Values(
        ShapeRefusalRow{"ANonPositiveInputChannelCount", make_conv(0, 8, 3), Shape{8, 8, 3},
                        "Conv2d", "in=0", "out=8"},
        ShapeRefusalRow{"ANonPositiveOutputChannelCount", make_conv(3, -1, 3), Shape{8, 8, 3},
                        "Conv2d", "in=3", "out=-1"},
        ShapeRefusalRow{"AWrongChannelCount", make_conv(3, 8, 3), Shape{8, 8, 4}, "Conv2d(3, 8)",
                        "[8, 8, 4]", "[H, W, 3]"},
        ShapeRefusalRow{"AFeatureVector", make_conv(3, 8, 3), Shape{3}, "Conv2d(3, 8)", "[3]",
                        "[H, W, 3]"},
        // The third extent equals in_channels, so only the rank check can refuse this shape.
        ShapeRefusalRow{"ABatchedShape", make_conv(3, 8, 3), Shape{2, 8, 3, 3}, "Conv2d(3, 8)",
                        "[2, 8, 3, 3]", "[H, W, 3]"},
        ShapeRefusalRow{"AWindowThatDoesNotFit", make_conv(3, 8, 5), Shape{4, 4, 3},
                        "Conv2d(3, 8)", "[4, 4, 3]", "5x5"},
        ShapeRefusalRow{"AZeroStride", make_conv(3, 8, 3, 0), Shape{8, 8, 3}, "Conv2d(3, 8)",
                        "stride 0,0", ""},
        ShapeRefusalRow{"ANegativePadding", make_conv(3, 8, 3, 1, -1), Shape{8, 8, 3},
                        "Conv2d(3, 8)", "padding -1,-1", ""}),
    [](const ::testing::TestParamInfo<ShapeRefusalRow>& info) { return info.param.name; });

/** @brief Holds a layer, a batched input, whether it is NCHW, and the refusal forward meets. */
struct ForwardRefusalRow
{
    std::string name;   /**< The row's name. */
    Conv2d layer;       /**< The layer under test. */
    Shape input;        /**< The input shape, batch axis included. */
    bool nchw;          /**< Whether the input is converted to NCHW first. */
    std::string first;  /**< A substring the message holds. */
    std::string second; /**< A second substring the message holds; empty for none. */
    std::string third;  /**< A third substring the message holds; empty for none. */
};

/** @brief Runs the forward refusal over each layer and input it cannot take. */
class Small_Conv2dForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};

/** @brief Tests that forward refuses the row, names what the row says and declares nothing. */
TEST_P(Small_Conv2dForwardRefusal, ForwardRefuses)
{
    const ForwardRefusalRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Conv2d layer = row.layer;
    Graph::ValueID x = builder.input(row.input, "x");
    if (row.nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_FALSE(layer.bias.declared());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_Conv2dForwardRefusal,
    ::testing::Values(
        ForwardRefusalRow{"AWrongChannelCount", make_conv(3, 8, 3), Shape{2, 8, 8, 4}, false,
                          "Conv2d(3, 8)", "[8, 8, 4]", "[H, W, 3]"},
        ForwardRefusalRow{"AnInputThatIsNotRankFour", make_conv(3, 8, 3), Shape{2, 3}, false,
                          "Conv2d(3, 8)", "rank-4", "[2, 3]"},
        ForwardRefusalRow{"ANonPositiveChannelCount", make_conv(3, 0, 3), Shape{2, 8, 8, 3},
                          false, "Conv2d", "in=3", "out=0"},
        ForwardRefusalRow{"AWindowThatDoesNotFit", make_conv(3, 8, 5), Shape{2, 4, 4, 3}, false,
                          "Conv2d(3, 8)", "[4, 4, 3]", "5x5"},
        // Every extent is 3, so the permuted shape passes every check but the layout tag.
        ForwardRefusalRow{"AnNCHWInput", make_conv(3, 8, 3, 1, 1), Shape{2, 3, 3, 3}, true,
                          "Conv2d", "NHWC", ""}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

/** @brief Tests that a loss yields a gradient for the weight and for the bias. */
TEST(Small_Conv2d, DifferentiatingALossYieldsAGradientForTheWeightAndTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Conv2d layer = make_conv(2, 3, 3, 1, 1);
    const Graph::ValueID x = builder.input(Shape{4, 6, 6, 2}, "x");
    const Graph::ValueID targets = builder.input(Shape{4, 3}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID logits,
                            builder.spatial_mean(layer.forward(builder, x)));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));
    const std::vector<Graph::ValueID> wrt = {layer.weight.id_in(graph), layer.bias.id_in(graph)};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Graph::ValueID> grads, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const Graph::ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(grads.size(), 2u);
    expect_gradient_of(graph, grads[0], wrt[0]);
    expect_gradient_of(graph, grads[1], wrt[1]);
}

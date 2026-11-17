/**************************************************************************/
/* test_groups.cpp                                                 */
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
 * @file test_groups.cpp
 * @brief Tests the groups of the conv2d layer: grouped and depthwise filters, their values and gradients.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/host_row_values.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::Conv2d;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::require_value;

    /** @brief Builds a layer with a square window of @p kernel at stride one and no pad. */
    Conv2d make_grouped(int64_t in_channels, int64_t out_channels, int64_t groups, int64_t kernel,
                        bool use_bias = true)
    {
        Conv2d layer;
        layer.in_channels = in_channels;
        layer.out_channels = out_channels;
        layer.groups = groups;
        layer.geometry.kernel_h = kernel;
        layer.geometry.kernel_w = kernel;
        layer.use_bias = use_bias;
        return layer;
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

    /** @brief Holds one layer with the weight and result shapes it must give over a sample. */
    struct ShapeRow
    {
        std::string name;      /**< The row's name. */
        Conv2d layer;          /**< The layer under test. */
        Shape weight;          /**< The weight's shape after a trace. */
        Shape result;          /**< The per-sample result shape. */
    };

    /** @brief Names a ShapeRow for gtest. */
    std::string shape_name(const ::testing::TestParamInfo<ShapeRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each shape row. */
    class Small_Conv2dGroupsShape : public ::testing::TestWithParam<ShapeRow>
    {

    };
} // namespace

/** @brief Tests that the weight has in / groups input channels, or the depthwise taps, and the result agrees. */
TEST_P(Small_Conv2dGroupsShape, TheWeightAndTheResultFollowTheGroups)
{
    ShapeRow row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, row.layer.in_channels}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, row.layer.forward(builder, x));

    EXPECT_EQ(row.layer.weight.shape(), row.weight);
    EXPECT_EQ(graph.value(y).shape,
              (Shape{2, row.result[0], row.result[1], row.result[2]}));
    SA_ASSERT_OK_AND_ASSIGN(const Shape per_sample, row.layer.output_shape(Shape{6, 6,
                                                                       row.layer.in_channels}));
    EXPECT_EQ(per_sample, row.result);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_Conv2dGroupsShape,
    ::testing::Values(
        ShapeRow{"TwoGroupsOfTwoFilters", make_grouped(4, 4, 2, 3), Shape{4, 3, 3, 2},
                 Shape{4, 4, 4}},
        ShapeRow{"TwoGroupsOfThreeInputChannels", make_grouped(6, 4, 2, 3), Shape{4, 3, 3, 3},
                 Shape{4, 4, 4}},
        ShapeRow{"DepthwiseKeepsTheChannelsAndHoldsTaps", make_grouped(4, 4, 4, 3),
                 Shape{3, 3, 4}, Shape{4, 4, 4}},
        ShapeRow{"AChannelMultiplierIsAGroupedConvolution", make_grouped(4, 8, 4, 3),
                 Shape{8, 3, 3, 1}, Shape{4, 4, 8}},
        ShapeRow{"OneGroupIsTheDenseConvolution", make_grouped(4, 4, 1, 3), Shape{4, 3, 3, 4},
                 Shape{4, 4, 4}}),
    shape_name);

namespace
{
    /** @brief Holds one description the layer refuses and the words the message holds. */
    struct RefusalRow
    {
        std::string name;     /**< The row's name. */
        Conv2d layer;         /**< The layer under test. */
        std::string fragment; /**< What the message holds. */
    };

    /** @brief Names a RefusalRow for gtest. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Returns @p layer with a trailing pad of one row and one column. */
    Conv2d with_trailing_pad(Conv2d layer)
    {
        layer.trailing_pad_h = 1;
        layer.trailing_pad_w = 1;
        return layer;
    }

    /** @brief Runs each refusal row. */
    class Small_Conv2dGroupsRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };
} // namespace

/** @brief Tests that a description with groups that do not fit is refused by the shape rule. */
TEST_P(Small_Conv2dGroupsRefusal, OutputShapeRefusesTheGroups)
{
    const RefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(row.layer.output_shape(Shape{6, 6, row.layer.in_channels}),
                      StatusCode::InvalidDescription, row.fragment);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_Conv2dGroupsRefusal,
    ::testing::Values(
        RefusalRow{"ZeroGroups", make_grouped(4, 4, 0, 3), "groups that divide both channel counts"},
        RefusalRow{"GroupsThatDoNotDivideTheInput", make_grouped(4, 6, 3, 3),
                   "groups that divide both channel counts; got 3"},
        RefusalRow{"GroupsThatDoNotDivideTheOutput", make_grouped(6, 4, 3, 3),
                   "groups that divide both channel counts; got 3"},
        RefusalRow{"ADepthwiseLayerWithATrailingPad", with_trailing_pad(make_grouped(4, 4, 4, 3)),
                   "depthwise, which has no trailing padding"}),
    refusal_name);

/** @brief Tests that a grouped layer traces one window expansion per group and one join. */
TEST(Small_Conv2dGroups, AGroupedLayerTracesOneConvolutionPerGroup)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    // A 3x3 window, since a pointwise layer multiplies its input without expanding windows.
    Conv2d layer = make_grouped(6, 6, 3, 3);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 6}, "x");

    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(count_op(graph, Ops::K_IM2COL.id), 3u);
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 3u);
    EXPECT_EQ(count_op(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_DEPTHWISE_CONV2D.id), 0u);
}

/** @brief Tests that a depthwise layer reuses the depthwise node and adds its bias per channel. */
TEST(Small_Conv2dGroups, ADepthwiseLayerTracesTheDepthwiseNodeAndABroadcastBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Conv2d layer = make_grouped(4, 4, 4, 3);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 4}, "x");

    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(count_op(graph, Ops::K_DEPTHWISE_CONV2D.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_BROADCAST_ADD.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_IM2COL.id), 0u);
    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 3, 4]", "bias:[4]"}));
}

/** @brief Tests that a bias-free depthwise layer adds no broadcast. */
TEST(Small_Conv2dGroups, ABiasFreeDepthwiseLayerAddsNoBroadcast)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Conv2d layer = make_grouped(4, 4, 4, 3, false);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 4}, "x");

    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(count_op(graph, Ops::K_BROADCAST_ADD.id), 0u);
    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 3, 4]"}));
}

/** @brief Tests that a grouped layer holds one weight and one bias, and each fan counts one group. */
TEST(Small_Conv2dGroups, AGroupedLayerHoldsOneWeightAndOneBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Conv2d layer = make_grouped(4, 6, 2, 3);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 4}, "x");

    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[6, 3, 3, 2]", "bias:[6]"}));
    EXPECT_EQ(layer.weight.fan_in(), 3 * 3 * 2);
    EXPECT_EQ(layer.weight.fan_out(), 3 * 3 * 3);
}

namespace
{
    /** @brief Holds the extents the value cases trace over and the two leaves of the layer. */
    struct TracedLayer
    {
        Graph::Graph graph;                         /**< The graph the layer was traced into. */
        Graph::ValueID input = Graph::K_NO_VALUE;   /**< The input leaf. */
        Graph::ValueID output = Graph::K_NO_VALUE;  /**< The result. */
        Graph::ValueID weight = Graph::K_NO_VALUE;  /**< The weight leaf. */
        Graph::ValueID bias = Graph::K_NO_VALUE;    /**< The bias leaf. */
    };

    /** @brief Traces @p layer over a float64 input of @p extents into @p traced. */
    void trace_layer(TracedLayer& traced, Conv2d layer, const Shape& extents)
    {
        const Graph::GraphBuilder builder(traced.graph);
        traced.input = builder.input(extents, "x", DType::Float64);
        traced.output = layer.forward(builder, traced.input).value();
        traced.weight = layer.weight.id_in(traced.graph);
        traced.bias = layer.bias.id_in(traced.graph);
    }

    /** @brief Expects every element of @p got to equal @p expected exactly. */
    void expect_exact(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_TRUE(TestSupport::is_within(
                TestSupport::measure_error(got[index], expected[index], 0.0)))
                << "element " << index;
    }
} // namespace

/**
 * @brief Tests the values of two groups of two filters over two pixels against the hand results.
 *
 * Filters 0 and 1 read channels 0 and 1, and filters 2 and 3 read channels 2 and 3. With the
 * weights (1, 2), (3, 4), (5, 6), (7, 8) and the biases 10, 20, 30, 40, the pixel (1, 2, 3, 4) gives
 * 1 + 4 + 10, 3 + 8 + 20, 15 + 24 + 30 and 21 + 32 + 40, and the pixel (0, 1, 0, -1) gives 2 + 10,
 * 4 + 20, -6 + 30 and -8 + 40.
 */
TEST(Small_Conv2dGroups, TheValuesOfTwoGroupsMatchTheHandResults)
{
    TracedLayer traced;
    trace_layer(traced, make_grouped(4, 4, 2, 1), Shape{1, 2, 1, 4});

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph,
                                 {{traced.input, {1, 2, 3, 4, 0, 1, 0, -1}},
                                  {traced.weight, {1, 2, 3, 4, 5, 6, 7, 8}},
                                  {traced.bias, {10, 20, 30, 40}}},
                                 NN::HostTableList(), traced.output));

    expect_exact(got, {15, 31, 69, 93, 12, 24, 24, 32});
}

/**
 * @brief Tests the values of a depthwise layer over three rows against the hand results.
 *
 * Channel 0 reads 1, 2, 4 and channel 1 reads 3, 5, 7. The taps are (1, 2) for channel 0 and
 * (10, 20) for channel 1, and the biases are 100 and 200, so channel 0 gives 1 + 4 + 100 and
 * 2 + 8 + 100, and channel 1 gives 30 + 100 + 200 and 50 + 140 + 200.
 */
TEST(Small_Conv2dGroups, TheValuesOfADepthwiseLayerMatchTheHandResults)
{
    Conv2d layer = make_grouped(2, 2, 2, 1);
    layer.geometry.kernel_h = 2;
    layer.geometry.kernel_w = 1;
    TracedLayer traced;
    trace_layer(traced, layer, Shape{1, 3, 1, 2});

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph,
                                 {{traced.input, {1, 3, 2, 5, 4, 7}},
                                  {traced.weight, {1, 10, 2, 20}},
                                  {traced.bias, {100, 200}}},
                                 NN::HostTableList(), traced.output));

    expect_exact(got, {105, 330, 110, 390});
}

/** @brief Tests that a bias-free grouped layer gives the products alone. */
TEST(Small_Conv2dGroups, ABiasFreeGroupedLayerGivesTheProductsAlone)
{
    TracedLayer traced;
    trace_layer(traced, make_grouped(4, 4, 2, 1, false), Shape{1, 1, 1, 4});

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph,
                                 {{traced.input, {1, 2, 3, 4}},
                                  {traced.weight, {1, 2, 3, 4, 5, 6, 7, 8}}},
                                 NN::HostTableList(), traced.output));

    expect_exact(got, {5, 11, 39, 53});
}

namespace
{
    /** @brief Names a model-file row of the conv2d type and the shape it reads. */
    Result<std::unique_ptr<NN::Layer>> make_row(const Shape& input, const std::string& attributes)
    {
        const Config::Value row = require_value(Config::parse_json(attributes, "<test>"));

        return NN::make_conv2d(input, row, "layer 1 'conv2d'");
    }
} // namespace

/** @brief Tests that the row key "groups" reaches the layer and the result shape. */
TEST(Small_Conv2dGroups, TheRowKeyGroupsBuildsAGroupedLayer)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Layer> layer,
        make_row(Shape{8, 8, 4}, R"({"filters": 6, "kernel": 3, "groups": 2})"));

    EXPECT_EQ(layer->output_shape(), (Shape{6, 6, 6}));
}

/** @brief Tests that a row whose groups do not divide the channels is refused with its context. */
TEST(Small_Conv2dGroups, TheRowRefusesGroupsThatDoNotDivideTheChannels)
{
    SA_EXPECT_FAILURE(make_row(Shape{8, 8, 4}, R"({"filters": 6, "kernel": 3, "groups": 3})"),
                      StatusCode::InvalidDescription, "layer 1 'conv2d'",
                      "groups that divide both channel counts; got 3");
}

/** @brief Tests that a row with no groups key is the dense convolution. */
TEST(Small_Conv2dGroups, TheRowDefaultsToOneGroup)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Layer> layer,
                            make_row(Shape{8, 8, 4}, R"({"filters": 6, "kernel": 3})"));

    EXPECT_EQ(layer->output_shape(), (Shape{6, 6, 6}));
}

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_Conv2dGroupsGradient : public SaTest::EngineTest
    {

    };

    /** @brief Returns @p count values that differ and lie within a few units. */
    std::vector<double> spread(std::size_t count, double offset)
    {
        std::vector<double> values(count);
        for (std::size_t index = 0; index < count; ++index)
            values[index] = static_cast<double>((index * 5) % 9) / 3.0 - 1.3 + offset;

        return values;
    }
} // namespace

/** @brief Tests that the input gradient of a grouped layer matches central differences. */
TEST_F(Medium_Conv2dGroupsGradient, TheInputGradientOfAGroupedLayerMatchesCentralDifferences)
{
    Conv2d layer = make_grouped(4, 4, 2, 1);
    Graph::ValueID weight = Graph::K_NO_VALUE;
    Graph::ValueID bias = Graph::K_NO_VALUE;

    SaTest::GradientProbe probe(backend(), Shape{2, 2, 1, 4},
                                [&](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        const Graph::ValueID y = layer.forward(builder, x).value();
        weight = layer.weight.id_in(builder.graph());
        bias = layer.bias.id_in(builder.graph());
        return builder.reshape(y, Shape{2, 8});
    });
    probe.fill(weight, spread(8, 0.4));
    probe.fill(bias, spread(4, 0.1));

    std::vector<double> point = spread(16, 0.0);
    SaTest::expect_gradient_matches(probe, point);
}

/** @brief Tests that the weight gradient of a grouped layer matches central differences. */
TEST_F(Medium_Conv2dGroupsGradient, TheWeightGradientOfAGroupedLayerMatchesCentralDifferences)
{
    Conv2d layer = make_grouped(4, 4, 2, 1, false);
    Graph::ValueID input = Graph::K_NO_VALUE;

    SaTest::GradientProbe probe(
        backend(), SaTest::BuildsSubject{},
        [&](const Graph::GraphBuilder& builder)
    {
        input = builder.input(Shape{2, 2, 1, 4}, "x", DType::Float64);
        const Graph::ValueID y = layer.forward(builder, input).value();
        const Graph::ValueID flat = builder.reshape(y, Shape{2, 8}).value();
        return SaTest::GradientSubject{layer.weight.id_in(builder.graph()), flat};
    });
    probe.fill(input, spread(16, 0.2));

    std::vector<double> point = spread(8, 0.4);
    SaTest::expect_gradient_matches(probe, point);
}

/** @brief Tests that the input gradient of a depthwise layer matches central differences. */
TEST_F(Medium_Conv2dGroupsGradient, TheInputGradientOfADepthwiseLayerMatchesCentralDifferences)
{
    Conv2d layer = make_grouped(2, 2, 2, 1);
    layer.geometry.kernel_h = 2;
    Graph::ValueID weight = Graph::K_NO_VALUE;
    Graph::ValueID bias = Graph::K_NO_VALUE;

    SaTest::GradientProbe probe(backend(), Shape{2, 3, 1, 2},
                                [&](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        const Graph::ValueID y = layer.forward(builder, x).value();
        weight = layer.weight.id_in(builder.graph());
        bias = layer.bias.id_in(builder.graph());
        return builder.reshape(y, Shape{2, 4});
    });
    probe.fill(weight, spread(4, 0.5));
    probe.fill(bias, spread(2, 0.1));

    std::vector<double> point = spread(12, 0.0);
    SaTest::expect_gradient_matches(probe, point);
}

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
 * @file test_avg_pool2d.cpp
 * @brief Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/ops/ops.hpp>

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

    /** @brief Returns @p g with the top and the left padded by @p pad. */
    Graph::ConvGeometry padded_top_left(Graph::ConvGeometry g, int64_t pad)
    {
        g.pad_top = pad;
        g.pad_left = pad;
        return g;
    }

    /** @brief Returns @p g with the top padded by @p pad. */
    Graph::ConvGeometry padded_top(Graph::ConvGeometry g, int64_t pad)
    {
        g.pad_top = pad;
        return g;
    }

    /** @brief Returns @p g dilated by @p h on the height and @p w on the width. */
    Graph::ConvGeometry dilated(Graph::ConvGeometry g, int64_t h, int64_t w)
    {
        g.dilation_h = h;
        g.dilation_w = w;
        return g;
    }

    /** @brief Traces @p layer over a batch of two @p sample images and returns the shape. */
    Shape traced_shape(const AvgPool2d& layer, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, sample[0], sample[1], sample[2]}, "x");
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(layer.forward(builder, x).value()).shape;
    }

    /** @brief Drops the batch axis of a rank-4 @p batched shape. */
    Shape per_sample(const Shape& batched)
    {
        return Shape{batched[1], batched[2], batched[3]};
    }
} // namespace

/** @brief Tests that forward() yields the pooled [N, out_h, out_w, C] value tagged NHWC. */
TEST(Small_AvgPool2d, ForwardTracesThePooledShapeInNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 3}, "x");

    const AvgPool2d layer{window(3, 3, 3, 3)};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 2, 2, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Holds a layer geometry, the sample it pools and the per-sample shape it gives. */
struct AgreementRow
{
    std::string name;             /**< The row's name. */
    Graph::ConvGeometry geometry; /**< The window, step, padding and dilation. */
    Shape sample;                 /**< The per-sample input shape. */
    Shape expected;               /**< The per-sample shape the layer gives. */
};

/** @brief Runs the shape agreement over each kind of window. */
class Small_AvgPool2dShape : public ::testing::TestWithParam<AgreementRow>
{
};

/** @brief Tests that output_shape() equals the expected shape and the traced per-sample shape. */
TEST_P(Small_AvgPool2dShape, OutputShapeAgreesWithTheTracedShape)
{
    const AgreementRow& row = GetParam();
    const AvgPool2d layer{row.geometry};

    EXPECT_EQ(layer.output_shape(row.sample).value(), row.expected);
    EXPECT_EQ(layer.output_shape(row.sample).value(),
              per_sample(traced_shape(layer, row.sample)));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AvgPool2dShape,
    ::testing::Values(AgreementRow{"Square", window(3, 3, 3, 3), Shape{6, 6, 3}, Shape{2, 2, 3}},
                      AgreementRow{"PaddedUneven", padded_top_left(window(3, 2, 2, 1), 1),
                                   Shape{7, 5, 4}, Shape{4, 6, 4}},
                      AgreementRow{"Dilated", dilated(window(3, 3, 1, 1), 2, 1), Shape{9, 8, 1},
                                   Shape{5, 6, 1}}),
    [](const ::testing::TestParamInfo<AgreementRow>& info) { return info.param.name; });

/** @brief Holds a geometry and a per-sample shape that output_shape refuses, with the message. */
struct ShapeRefusalRow
{
    std::string name;             /**< The row's name. */
    Graph::ConvGeometry geometry; /**< The window, step, padding and dilation. */
    Shape sample;                 /**< The shape given to output_shape. */
    std::string first;            /**< A substring the message holds; empty for none. */
    std::string second;           /**< A second substring the message holds; empty for none. */
};

/** @brief Runs the output_shape refusal over each input it cannot take. */
class Small_AvgPool2dShapeRefusal : public ::testing::TestWithParam<ShapeRefusalRow>
{
};

/** @brief Tests that output_shape() refuses the row and names what the row says. */
TEST_P(Small_AvgPool2dShapeRefusal, OutputShapeRefuses)
{
    const ShapeRefusalRow& row = GetParam();
    const AvgPool2d layer{row.geometry};

    SA_EXPECT_FAILURE(layer.output_shape(row.sample), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AvgPool2dShapeRefusal,
    ::testing::Values(
        ShapeRefusalRow{"AFeatureVector", window(2, 2, 2, 2), Shape{8}, "", ""},
        ShapeRefusalRow{"ABatchedShape", window(2, 2, 2, 2), Shape{2, 4, 4, 3}, "", ""},
        ShapeRefusalRow{"AWindowLargerThanTheImage", window(5, 5, 1, 1), Shape{4, 4, 3}, "", ""},
        ShapeRefusalRow{"AZeroStride", window(2, 2, 0, 0), Shape{4, 4, 3}, "AvgPool2d",
                        "stride 0,0"},
        ShapeRefusalRow{"ANegativePadding", padded_top(window(2, 2, 2, 2), -1), Shape{4, 4, 3},
                        "AvgPool2d", "padding -1,0"}),
    [](const ::testing::TestParamInfo<ShapeRefusalRow>& info) { return info.param.name; });

/** @brief Holds an input of forward, whether it is in NCHW layout, and the refusal it meets. */
struct ForwardRefusalRow
{
    std::string name;             /**< The row's name. */
    Graph::ConvGeometry geometry; /**< The window, step, padding and dilation. */
    Shape input;                  /**< The input shape, batch axis included. */
    bool nchw;                    /**< Whether the input is converted to NCHW first. */
    std::string first;            /**< A substring the message holds; empty for none. */
    std::string second;           /**< A second substring the message holds; empty for none. */
};

/** @brief Runs the forward refusal over each input and geometry it cannot take. */
class Small_AvgPool2dForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};

/** @brief Tests that forward() refuses the row and names what the row says. */
TEST_P(Small_AvgPool2dForwardRefusal, ForwardRefuses)
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
    const AvgPool2d layer{row.geometry};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AvgPool2dForwardRefusal,
    ::testing::Values(
        ForwardRefusalRow{"ARankTwoInput", window(2, 2, 2, 2), Shape{2, 8}, false, "", ""},
        ForwardRefusalRow{"AnNCHWInput", window(3, 3, 3, 3), Shape{1, 6, 6, 3}, true, "", ""},
        ForwardRefusalRow{"AnNCHWInputNarrowerThanTheWindowNamesTheLayout", window(3, 3, 1, 1),
                          Shape{1, 8, 8, 2}, true, "NHWC", ""},
        ForwardRefusalRow{"AZeroStride", window(2, 2, 0, 0), Shape{2, 4, 4, 3}, false, "AvgPool2d",
                          "stride 0,0"},
        ForwardRefusalRow{"ANegativePadding", padded_top(window(2, 2, 2, 2), -1),
                          Shape{2, 4, 4, 3}, false, "AvgPool2d", "padding -1,0"},
        ForwardRefusalRow{"AWindowLargerThanTheImage", window(5, 5, 1, 1), Shape{2, 4, 4, 3},
                          false, "", ""}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

/** @brief Tests that a traced AvgPool2d adds one value to the graph beside its input. */
TEST(Small_AvgPool2d, ForwardAddsOneValue)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 3}, "x");

    const AvgPool2d layer{window(3, 3, 3, 3)};
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(graph.value_count(), 2u);
}

/**
 * @brief Tests that a traced AvgPool2d adds one average-pool node that reads its input and
 *        carries the window geometry.
 */
TEST(Small_AvgPool2d, ForwardEmitsOneAveragePoolNodeThatCarriesTheWindow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    const std::size_t nodes_before = graph.node_count();
    const AvgPool2d layer{padded_top_left(dilated(window(3, 2, 2, 1), 2, 1), 1)};

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.node_count(), nodes_before + 1u);
    const Graph::Node& pool = graph.nodes()[nodes_before];
    EXPECT_EQ(pool.op, Ops::K_AVGPOOL2D.id);
    EXPECT_EQ(pool.inputs, (std::vector<Graph::ValueID>{x}));
    EXPECT_EQ(pool.outputs, (std::vector<Graph::ValueID>{y}));
    EXPECT_EQ(pool.attributes.values[0], 3.0);
    EXPECT_EQ(pool.attributes.values[1], 2.0);
    EXPECT_EQ(pool.attributes.values[2], 2.0);
    EXPECT_EQ(pool.attributes.values[3], 1.0);
    EXPECT_EQ(pool.attributes.values[4], 2.0);
    EXPECT_EQ(pool.attributes.values[5], 1.0);
    EXPECT_EQ(pool.attributes.values[6], 1.0);
    EXPECT_EQ(pool.attributes.values[7], 1.0);
}

/**************************************************************************/
/* test_global_avg_pool.cpp                                               */
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
 * @file test_global_avg_pool.cpp
 * @brief Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::GlobalAvgPool;

namespace
{
    /** @brief Traces the layer over an [N, H, W, C] input and returns the result's shape. */
    Shape traced_shape(int64_t batch, int64_t height, int64_t width, int64_t channels)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{batch, height, width, channels}, "x");

        const GlobalAvgPool layer{};
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(layer.forward(builder, x).value()).shape;
    }
} // namespace

/** @brief Tests that forward() reduces an [N, H, W, C] input to [N, C]. */
TEST(Small_GlobalAvgPool, ForwardReducesTheSpatialAxes)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 7, 5, 16}, "x");

    const GlobalAvgPool layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 16}));
}

/** @brief Tests that forward() emits one SPATIAL_MEAN node reading its input. */
TEST(Small_GlobalAvgPool, ForwardEmitsOneSpatialMeanNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, 3, 8}, "x");

    const GlobalAvgPool layer{};
    const std::size_t nodes_before = graph.nodes().size();
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.nodes().size(), nodes_before + 1);
    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_SPATIAL_MEAN.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs.front(), y);
}

/** @brief Tests that the traced result keeps the NHWC layout tag of the rank-4 input it reads. */
TEST(Small_GlobalAvgPool, ForwardKeepsTheNHWCLayoutTag)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 3}, "x");
    ASSERT_EQ(graph.value(x).layout, Graph::TensorLayout::NHWC);

    const GlobalAvgPool layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
    SA_EXPECT_OK(graph.finalise());
}

/** @brief Tests that output_shape() maps a per-sample [H, W, C] shape to [C]. */
TEST(Small_GlobalAvgPool, OutputShapeKeepsOnlyTheChannels)
{
    const GlobalAvgPool layer{};

    EXPECT_EQ(layer.output_shape(Shape{7, 5, 16}).value(), (Shape{16}));
}

/** @brief Tests that output_shape() agrees with the traced shape for two different inputs. */
TEST(Small_GlobalAvgPool, OutputShapeAgreesWithForward)
{
    const GlobalAvgPool layer{};

    const Shape square = layer.output_shape(Shape{7, 7, 512}).value();
    ASSERT_EQ(square.rank(), 1);
    EXPECT_EQ(traced_shape(4, 7, 7, 512), (Shape{4, square[0]}));

    const Shape oblong = layer.output_shape(Shape{3, 9, 2}).value();
    ASSERT_EQ(oblong.rank(), 1);
    EXPECT_EQ(traced_shape(1, 3, 9, 2), (Shape{1, oblong[0]}));
}

/** @brief Holds a name and a shape the layer refuses. */
struct RefusedShapeRow
{
    std::string name; /**< The row's name. */
    Shape shape;      /**< The shape given to output_shape or forward. */
};

/** @brief Runs the output_shape refusal over each per-sample shape of the wrong rank. */
class Small_GlobalAvgPoolShapeRefusal : public ::testing::TestWithParam<RefusedShapeRow>
{
};

/** @brief Tests that output_shape() rejects a per-sample shape that is not rank 3. */
TEST_P(Small_GlobalAvgPoolShapeRefusal, OutputShapeRefuses)
{
    const GlobalAvgPool layer{};

    SA_EXPECT_FAILURE(layer.output_shape(GetParam().shape), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_GlobalAvgPoolShapeRefusal,
    ::testing::Values(RefusedShapeRow{"AFeatureVector", Shape{16}},
                      RefusedShapeRow{"ARankTwoShape", Shape{7, 16}},
                      RefusedShapeRow{"AShapeWithTheBatchAxis", Shape{4, 7, 5, 16}}),
    [](const ::testing::TestParamInfo<RefusedShapeRow>& info) { return info.param.name; });

/** @brief Holds an input shape forward refuses and whether it is converted to NCHW. */
struct ForwardRefusalRow
{
    std::string name; /**< The row's name. */
    Shape input;      /**< The input shape. */
    bool nchw;        /**< Whether the input is converted to NCHW first. */
};

/** @brief Runs the forward refusal over each input the layer does not take. */
class Small_GlobalAvgPoolForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};

/** @brief Tests that forward() rejects the input of the row. */
TEST_P(Small_GlobalAvgPoolForwardRefusal, ForwardRefuses)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Graph::ValueID x = builder.input(GetParam().input, "x");
    if (GetParam().nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }
    const GlobalAvgPool layer{};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_GlobalAvgPoolForwardRefusal,
    ::testing::Values(ForwardRefusalRow{"AFeatureMatrix", Shape{4, 16}, false},
                      ForwardRefusalRow{"AnInputWithoutTheBatchAxis", Shape{7, 5, 16}, false},
                      ForwardRefusalRow{"AnNCHWInput", Shape{1, 4, 4, 3}, true}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

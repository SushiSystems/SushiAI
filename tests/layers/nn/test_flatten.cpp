/**************************************************************************/
/* test_flatten.cpp                                                       */
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
 * @file test_flatten.cpp
 * @brief Tests the Flatten layer's trace, shape rule, rejections and parameter walk.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::Flatten;

namespace
{
    /** @brief Traces Flatten over a fresh input of @p batched and returns the result's shape. */
    Shape traced_shape(const Shape& batched)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x = builder.input(batched, "x");
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(Flatten{}.forward(builder, x).value()).shape;
    }
} // namespace

/** @brief Tests that a rank-4 input traces to one RESHAPE of shape [N, H * W * C]. */
TEST(Small_Flatten, ARank4InputTracesToOneReshape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 4, 5, 3}, "x");
    const std::size_t nodes_before = graph.node_count();

    const Flatten layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 60}));
    ASSERT_EQ(graph.node_count(), nodes_before + 1);

    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_RESHAPE.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs[0], x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs[0], y);
}

/** @brief Tests that a rank-2 input comes back as the same value and adds no node. */
TEST(Small_Flatten, ARank2InputPassesThroughWithNoNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 7}, "x");
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    const Flatten layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(y, x);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}

/** @brief Tests that flattening an NHWC rank-4 input does not stamp the result with another tag. */
TEST(Small_Flatten, AnNHWCInputYieldsAnNHWCResult)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 4, 5, 3}, "x");
    ASSERT_EQ(graph.value(x).layout, Graph::TensorLayout::NHWC);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, Flatten{}.forward(builder, x));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

namespace
{
/** @brief Holds a batched input, its per-sample shape and the per-sample shape Flatten gives. */
struct RankRow
{
    std::string name;  /**< The row's name. */
    Shape batched;     /**< The input shape, batch axis included. */
    Shape sample;      /**< The per-sample input shape. */
    Shape flattened;   /**< The per-sample shape Flatten gives. */
};
} // namespace

namespace
{
/** @brief Runs the flattening promise over each input rank. */
class Small_FlattenRank : public ::testing::TestWithParam<RankRow>
{
};
} // namespace

/** @brief Tests that output_shape and the traced shape fold every axis after the batch. */
TEST_P(Small_FlattenRank, TheInnerAxesFold)
{
    const RankRow& row = GetParam();
    const Flatten layer{};

    EXPECT_EQ(layer.output_shape(row.sample).value(), row.flattened);
    EXPECT_EQ(traced_shape(row.batched), (Shape{row.batched[0], row.flattened[0]}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_FlattenRank,
    ::testing::Values(RankRow{"AnImage", Shape{2, 4, 5, 3}, Shape{4, 5, 3}, Shape{60}},
                      RankRow{"AMatrix", Shape{6, 3, 4}, Shape{3, 4}, Shape{12}},
                      RankRow{"AVector", Shape{2, 7}, Shape{7}, Shape{7}}),
    [](const ::testing::TestParamInfo<RankRow>& info) { return info.param.name; });

namespace
{
/** @brief Holds an input shape Flatten refuses and whether it is tagged NCHW. */
struct RefusedInputRow
{
    std::string name; /**< The row's name. */
    Shape input;      /**< The input shape, batch axis included. */
    bool nchw;        /**< Whether the input is tagged NCHW. */
};
} // namespace

namespace
{
/** @brief Runs the forward refusal over each input Flatten does not take. */
class Small_FlattenRefused : public ::testing::TestWithParam<RefusedInputRow>
{
};
} // namespace

/** @brief Tests that forward refuses the input. */
TEST_P(Small_FlattenRefused, ForwardRefuses)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(GetParam().input, "x");
    if (GetParam().nchw)
        graph.value(x).layout = Graph::TensorLayout::NCHW;
    const Flatten layer{};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_FlattenRefused,
    ::testing::Values(RefusedInputRow{"AnNCHWInput", Shape{2, 3, 4, 5}, true},
                      RefusedInputRow{"ARank1Input", Shape{6}, false}),
    [](const ::testing::TestParamInfo<RefusedInputRow>& info) { return info.param.name; });

/** @brief Tests that output_shape() fails on a rank-0 per-sample shape. */
TEST(Small_Flatten, OutputShapeRejectsARank0Shape)
{
    const Flatten layer{};

    SA_EXPECT_FAILURE(layer.output_shape(Shape{}), StatusCode::InvalidDescription);
}

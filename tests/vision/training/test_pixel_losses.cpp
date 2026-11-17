/**************************************************************************/
/* test_pixel_losses.cpp                                                  */
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
 * @file test_pixel_losses.cpp
 * @brief Tests the value each per-pixel loss graph emits on the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>

using namespace SushiAI;
using Vision::Training::DiceOperands;
using Vision::Training::emit_pixel_cross_entropy;
using Vision::Training::emit_soft_dice;

namespace
{
    /** @brief Holds the rows of the scores, R. */
    constexpr std::int64_t K_ROWS = 6;

    /** @brief Holds the classes of the scores, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Returns how many nodes of @p graph carry the operation @p op. */
    std::size_t count_of(const Graph::Graph& graph, const Ops::OpName& op)
    {
        std::size_t count = 0;
        for (std::size_t node = 0; node < graph.node_count(); ++node)
            count += graph.node(static_cast<Graph::NodeID>(node)).op == op.id ? 1 : 0;

        return count;
    }
} // namespace

/** @brief Tests that the cross entropy is one value of the logits' type. */
TEST(Small_VisionTrainingPixelLosses, TheCrossEntropyIsOneValueOfTheLogitsType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits", DType::Float64);
    const Graph::ValueID labels = builder.input(Shape{K_ROWS}, "labels", DType::Int32);
    const Graph::ValueID weights = builder.input(Shape{K_ROWS, 1}, "weights", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            emit_pixel_cross_entropy(builder, graph, logits, labels, weights));

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that the soft Dice is one float32 value. */
TEST(Small_VisionTrainingPixelLosses, TheSoftDiceIsOneFloat32Value)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits");
    DiceOperands operands;
    operands.labels = builder.input(Shape{K_ROWS}, "labels", DType::Int32);
    operands.kept = builder.input(Shape{K_ROWS, 1}, "kept");
    operands.class_pixels = builder.input(Shape{K_CLASSES}, "class_pixels");
    operands.class_weights = builder.input(Shape{K_CLASSES}, "class_weights");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            emit_soft_dice(builder, graph, logits, operands));

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float32);
}

/** @brief Tests that the cross entropy picks each label with one TAKE_ALONG and no GATHER_ROWS. */
TEST(Small_VisionTrainingPixelLosses, TheCrossEntropyPicksEachLabelWithOneTakeAlong)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits");
    const Graph::ValueID labels = builder.input(Shape{K_ROWS}, "labels", DType::Int32);
    const Graph::ValueID weights = builder.input(Shape{K_ROWS, 1}, "weights");

    SA_EXPECT_OK(emit_pixel_cross_entropy(builder, graph, logits, labels, weights));

    EXPECT_EQ(count_of(graph, Ops::K_TAKE_ALONG), 1u);
    EXPECT_EQ(count_of(graph, Ops::K_GATHER_ROWS), 0u);
}

/** @brief Tests that the soft Dice picks each label with one TAKE_ALONG and sums by SCATTER_ADD. */
TEST(Small_VisionTrainingPixelLosses, TheSoftDicePicksEachLabelWithOneTakeAlong)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits");
    DiceOperands operands;
    operands.labels = builder.input(Shape{K_ROWS}, "labels", DType::Int32);
    operands.kept = builder.input(Shape{K_ROWS, 1}, "kept");
    operands.class_pixels = builder.input(Shape{K_CLASSES}, "class_pixels");
    operands.class_weights = builder.input(Shape{K_CLASSES}, "class_weights");

    SA_EXPECT_OK(emit_soft_dice(builder, graph, logits, operands));

    EXPECT_EQ(count_of(graph, Ops::K_TAKE_ALONG), 1u);
    EXPECT_EQ(count_of(graph, Ops::K_SCATTER_ADD_ROWS), 1u);
    EXPECT_EQ(count_of(graph, Ops::K_GATHER_ROWS), 0u);
}

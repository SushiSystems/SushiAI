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
 * @file test_pixel_losses.cpp
 * @brief Tests the value and the operations each per-pixel loss graph emits on the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>

using namespace SushiAI;
using Vision::Training::DiceOperands;
using Vision::Training::emit_pixel_cross_entropy;
using Vision::Training::emit_soft_dice;
using Vision::Training::K_DICE_EPSILON;

namespace
{
    /** @brief Holds the rows of the scores, R. */
    constexpr std::int64_t K_ROWS = 6;

    /** @brief Holds the classes of the scores, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }
} // namespace

/** @brief Tests that the cross entropy is one value of the logits' type from one gather. */
TEST(Small_VisionTrainingPixelLosses, TheCrossEntropyIsOneValueOfTheLogitsType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits", DType::Float64);
    const Graph::ValueID indices = builder.input(Shape{K_ROWS}, "indices", DType::Int32);
    const Graph::ValueID weights = builder.input(Shape{K_ROWS, 1}, "weights", DType::Float64);

    const Graph::ValueID loss = emit_pixel_cross_entropy(builder, graph, logits, indices, weights);

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_GATHER_ROWS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_NLL_LOSS.id), 0u);
}

/** @brief Tests that the soft Dice is one value that gathers once and scatters once. */
TEST(Small_VisionTrainingPixelLosses, TheSoftDiceIsOneValueThatGathersAndScattersOnce)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{K_ROWS, K_CLASSES}, "logits");
    DiceOperands operands;
    operands.score_indices = builder.input(Shape{K_ROWS}, "score_indices", DType::Int32);
    operands.class_indices = builder.input(Shape{K_ROWS}, "class_indices", DType::Int32);
    operands.kept = builder.input(Shape{K_ROWS, 1}, "kept");
    operands.class_pixels = builder.input(Shape{K_CLASSES}, "class_pixels");
    operands.class_weights = builder.input(Shape{K_CLASSES}, "class_weights");

    const Graph::ValueID loss = emit_soft_dice(builder, graph, logits, operands);

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float32);
    EXPECT_EQ(count_nodes(graph, Ops::K_GATHER_ROWS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCATTER_ADD_ROWS.id), 1u);
}

/** @brief Tests that default Dice operands name no value and that the epsilon is 1e-6. */
TEST(Small_VisionTrainingPixelLosses, DefaultOperandsNameNoValueAndTheEpsilonIsAMillionth)
{
    const DiceOperands operands;

    EXPECT_EQ(operands.score_indices, Graph::K_NO_VALUE);
    EXPECT_EQ(operands.class_indices, Graph::K_NO_VALUE);
    EXPECT_EQ(operands.kept, Graph::K_NO_VALUE);
    EXPECT_EQ(operands.class_pixels, Graph::K_NO_VALUE);
    EXPECT_EQ(operands.class_weights, Graph::K_NO_VALUE);
    EXPECT_NEAR(K_DICE_EPSILON, 1e-6, 0.0);
}

/**************************************************************************/
/* test_linalg.cpp                                                        */
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
 * @file test_linalg.cpp
 * @brief Tests the backward rules of the linear-algebra table: the causal mask and the row gather.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

namespace
{
    /** @brief Holds T, the extent of each axis of the scores every case masks. */
    constexpr int64_t K_LENGTH = 4;

    /** @brief Holds the value the forward mask writes above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Appends the mask of @p scores and the cross-entropy of its rows; returns the loss. */
    ValueID masked_loss(const GraphBuilder& builder, ValueID scores)
    {
        const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
        const ValueID targets = builder.input(Shape{K_LENGTH, K_LENGTH}, "targets");
        return builder.nll_loss(builder.log_softmax(masked.value()), targets);
    }

    /** @brief Returns every node of @p graph that carries CAUSAL_MASK, in order. */
    std::vector<Graph::Node> masks_of(const Graph::Graph& graph)
    {
        std::vector<Graph::Node> found;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_CAUSAL_MASK.id)
                found.push_back(node);
        }

        return found;
    }
}

/** @brief Tests that the mask is differentiable and has a rule. */
TEST(Small_CausalMaskBackward, TheMaskHasARule)
{
    EXPECT_TRUE(Ops::K_CAUSAL_MASK.differentiable);
    EXPECT_FALSE(Ops::K_CAUSAL_MASK.blocks_gradient);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_CAUSAL_MASK.id));
}

/** @brief Tests that the gradient of the scores is a second mask whose fill is zero. */
TEST(Small_CausalMaskBackward, TheScoreGradientIsTheGradientMaskedWithZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.parameter(Shape{K_LENGTH, K_LENGTH}, "scores");
    const ValueID loss = masked_loss(builder, scores);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], K_NO_VALUE);

    const std::vector<Graph::Node> masks = masks_of(graph);
    ASSERT_EQ(masks.size(), 2u);
    EXPECT_EQ(masks[0].attributes.values[0], K_FILL);
    EXPECT_EQ(masks[1].attributes.values[0], 0.0);
    ASSERT_EQ(masks[1].outputs.size(), 1u);
    EXPECT_EQ(masks[1].outputs[0], gradients[0]);
    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{K_LENGTH, K_LENGTH}));
}

/** @brief Tests that scores that need no gradient add no second mask. */
TEST(Small_CausalMaskBackward, ScoresThatNeedNoGradientAddNoNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{K_LENGTH, K_LENGTH}, "scores");
    const ValueID loss = masked_loss(builder, scores);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    EXPECT_EQ(gradients[0], K_NO_VALUE);
    EXPECT_EQ(masks_of(graph).size(), 1u);
}

namespace
{
    /** @brief Holds the row count of the table every case reads. */
    constexpr int64_t K_TABLE_ROWS = 7;

    /** @brief Holds the row length of the table, which is the class count of the loss. */
    constexpr int64_t K_ROW_LENGTH = 3;

    /** @brief Differentiates @p loss with respect to @p wrt. */
    std::vector<ValueID> differentiate(Graph::Graph& graph, ValueID loss,
                                       const std::vector<ValueID>& wrt)
    {
        return Autograd::differentiate(graph, loss,
                                       SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    }

    /** @brief Returns every node of @p graph that carries @p op, in order. */
    std::vector<Graph::Node> nodes_of(const Graph::Graph& graph, OpID op)
    {
        std::vector<Graph::Node> found;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                found.push_back(node);
        }

        return found;
    }

    /** @brief Appends the cross-entropy of @p rows, an [N, K_ROW_LENGTH] value, to the graph. */
    ValueID loss_over(const GraphBuilder& builder, ValueID rows)
    {
        const Shape shape = builder.graph().value(rows).shape;
        const ValueID targets = builder.input(shape, "targets");
        return builder.nll_loss(builder.log_softmax(rows), targets);
    }
}

/** @brief Tests that the gather and its scatter-add each have a rule. */
TEST(Small_GatherBackward, TheGatherAndTheScatterAddEachHaveARule)
{
    EXPECT_TRUE(Ops::K_GATHER_ROWS.differentiable);
    EXPECT_TRUE(Ops::K_SCATTER_ADD_ROWS.differentiable);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_GATHER_ROWS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SCATTER_ADD_ROWS.id));
}

/** @brief Tests that the gradient of scattered rows is one GATHER_ROWS over the forward index. */
TEST(Small_GatherBackward, TheScatteredRowsGradientIsOneGatherOverTheForwardIndex)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.parameter(Shape{5, K_ROW_LENGTH}, "rows");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const Graph::Traced sums = builder.scatter_add_rows(rows, indices, K_TABLE_ROWS);
    ASSERT_TRUE(sums.has_value());
    const ValueID loss = loss_over(builder, sums.value());

    const std::vector<ValueID> wrt = {rows};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_EQ(gradients.size(), 1u);

    const std::vector<Graph::Node> gathers = nodes_of(graph, Ops::K_GATHER_ROWS.id);
    ASSERT_EQ(gathers.size(), 1u);
    ASSERT_EQ(gathers[0].inputs.size(), 2u);
    ASSERT_EQ(gathers[0].outputs.size(), 1u);
    EXPECT_EQ(gathers[0].inputs[1], indices);
    EXPECT_EQ(gathers[0].outputs[0], gradients[0]);
    EXPECT_EQ(graph.value(gathers[0].inputs[0]).shape, (Shape{K_TABLE_ROWS, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{5, K_ROW_LENGTH}));
    EXPECT_EQ(nodes_of(graph, Ops::K_SCATTER_ADD_ROWS.id).size(), 1u);
}

/** @brief Tests that the index of a scatter-add gets no gradient. */
TEST(Small_GatherBackward, TheScatterIndexReceivesNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.parameter(Shape{5, K_ROW_LENGTH}, "rows");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const Graph::Traced sums = builder.scatter_add_rows(rows, indices, K_TABLE_ROWS);
    ASSERT_TRUE(sums.has_value());
    const ValueID loss = loss_over(builder, sums.value());

    const std::vector<ValueID> wrt = {rows, indices};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_NE(gradients[0], Graph::K_NO_VALUE);
    EXPECT_EQ(gradients[1], Graph::K_NO_VALUE);
}

/** @brief Tests that scattered rows that require no gradient get no gather. */
TEST(Small_GatherBackward, FrozenScatteredRowsAppendNoGather)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{5, K_ROW_LENGTH}, "rows");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const ValueID weight = builder.parameter(Shape{K_ROW_LENGTH, K_ROW_LENGTH}, "weight");
    const Graph::Traced sums = builder.scatter_add_rows(rows, indices, K_TABLE_ROWS);
    ASSERT_TRUE(sums.has_value());
    const ValueID loss = loss_over(builder, builder.matmul(sums.value(), weight));

    const std::vector<ValueID> wrt = {weight};
    (void)Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    EXPECT_TRUE(nodes_of(graph, Ops::K_GATHER_ROWS.id).empty());
}

/** @brief Tests that the table's gradient is one SCATTER_ADD_ROWS over the forward index. */
TEST(Small_GatherBackward, TheTableGradientIsOneScatterAddOverTheForwardIndex)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const Graph::Traced gathered = builder.gather_rows(table, indices);
    ASSERT_TRUE(gathered.has_value());
    const ValueID loss = loss_over(builder, gathered.value());

    const std::vector<ValueID> gradients = differentiate(graph, loss, {table});

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], K_NO_VALUE);

    const std::vector<Graph::Node> scatters = nodes_of(graph, Ops::K_SCATTER_ADD_ROWS.id);
    ASSERT_EQ(scatters.size(), 1u);
    ASSERT_EQ(scatters[0].inputs.size(), 2u);
    ASSERT_EQ(scatters[0].outputs.size(), 1u);
    EXPECT_EQ(scatters[0].inputs[1], indices);
    EXPECT_EQ(scatters[0].outputs[0], gradients[0]);
    EXPECT_EQ(graph.value(scatters[0].inputs[0]).shape, (Shape{5, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{K_TABLE_ROWS, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(gradients[0]).dtype, DType::Float32);
    EXPECT_EQ(graph.value(gradients[0]).lifetime, Graph::Lifetime::Gradient);
}

/** @brief Tests that a requested INT32 index comes back with no gradient value. */
TEST(Small_GatherBackward, TheIndexReceivesNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const Graph::Traced gathered = builder.gather_rows(table, indices);
    ASSERT_TRUE(gathered.has_value());
    const ValueID loss = loss_over(builder, gathered.value());

    const std::vector<ValueID> gradients = differentiate(graph, loss, {table, indices});

    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_NE(gradients[0], K_NO_VALUE);
    EXPECT_EQ(gradients[1], K_NO_VALUE);
    EXPECT_FALSE(graph.value(indices).requires_grad);
}

/** @brief Tests that a rank-2 index scatters the gradient in the gather's own [B, T, D] shape. */
TEST(Small_GatherBackward, ARankTwoIndexScattersTheGradientOfTheGathersShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);
    const Graph::Traced gathered = builder.gather_rows(table, indices);
    ASSERT_TRUE(gathered.has_value());
    const ValueID rows = builder.reshape(gathered.value(), Shape{8, K_ROW_LENGTH});
    const ValueID loss = loss_over(builder, rows);

    const std::vector<ValueID> gradients = differentiate(graph, loss, {table});

    ASSERT_NE(gradients.at(0), K_NO_VALUE);
    const std::vector<Graph::Node> scatters = nodes_of(graph, Ops::K_SCATTER_ADD_ROWS.id);
    ASSERT_EQ(scatters.size(), 1u);
    EXPECT_EQ(graph.value(scatters[0].inputs.at(0)).shape, (Shape{2, 4, K_ROW_LENGTH}));
    EXPECT_EQ(scatters[0].inputs.at(1), indices);
    EXPECT_EQ(graph.value(gradients[0]).shape, (Shape{K_TABLE_ROWS, K_ROW_LENGTH}));
}

/** @brief Tests that a table that requires no gradient gets no scatter-add. */
TEST(Small_GatherBackward, AFrozenTableAppendsNoScatterAdd)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.buffer(Shape{K_TABLE_ROWS, 4}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const ValueID weight = builder.parameter(Shape{4, K_ROW_LENGTH}, "weight");
    const Graph::Traced gathered = builder.gather_rows(table, indices);
    ASSERT_TRUE(gathered.has_value());
    const ValueID loss = loss_over(builder, builder.matmul(gathered.value(), weight));

    const std::vector<ValueID> gradients = differentiate(graph, loss, {weight});

    EXPECT_NE(gradients.at(0), K_NO_VALUE);
    EXPECT_TRUE(nodes_of(graph, Ops::K_SCATTER_ADD_ROWS.id).empty());
}

/** @brief Tests that two gathers of one table give two scatter-adds joined by an ACCUMULATE. */
TEST(Small_GatherBackward, TwoGathersOfOneTableAccumulateTheirScatterAdds)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID first = builder.input(Shape{5}, "first", DType::Int32);
    const ValueID second = builder.input(Shape{5}, "second", DType::Int32);
    const Graph::Traced left = builder.gather_rows(table, first);
    const Graph::Traced right = builder.gather_rows(table, second);
    ASSERT_TRUE(left.has_value());
    ASSERT_TRUE(right.has_value());
    const ValueID loss = loss_over(builder, builder.add(left.value(), right.value()));

    const std::vector<ValueID> gradients = differentiate(graph, loss, {table});

    ASSERT_NE(gradients.at(0), K_NO_VALUE);
    EXPECT_EQ(nodes_of(graph, Ops::K_SCATTER_ADD_ROWS.id).size(), 2u);

    const Graph::Node& producer = graph.node(graph.value(gradients[0]).producer);
    EXPECT_EQ(producer.op, Ops::K_ACCUMULATE.id);
}

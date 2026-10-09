/**************************************************************************/
/* test_batched_matmul.cpp                                                */
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
 * @file test_batched_matmul.cpp
 * @brief Tests the shapes, flags and rejections of rank-3 MATMUL and the softmax emitter.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueId;

namespace
{
    /**
     * @brief Returns the node that writes @p value.
     *
     * Every emitter adds the node that writes its result, so the producer exists.
     */
    const Graph::Node& producer_of(const Graph::Graph& graph, ValueId value)
    {
        return graph.node(graph.value(value).producer);
    }

    /** @brief Expects @p node to read @p inputs and to write exactly one value, @p out. */
    void expect_reads_and_writes(const Graph::Node& node, ValueId out,
                                 const std::vector<ValueId>& inputs)
    {
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }
}

/** @brief Tests that a rank-2 product still yields [m, n] and keeps the flags it was given. */
TEST(Unit_BatchedMatmul, Rank2ProductKeepsItsShapeAndFlags)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId a = builder.input(Shape{3, 4}, "a");
    const ValueId b = builder.input(Shape{5, 4}, "b");

    const ValueId out = builder.matmul(a, b, false, true);

    EXPECT_EQ(graph.value(out).shape, Shape({3, 5}));
    const Graph::Node& node = producer_of(graph, out);
    EXPECT_EQ(node.op, Ops::K_MATMUL.id);
    EXPECT_FALSE(node.attributes.flag(0));
    EXPECT_TRUE(node.attributes.flag(1));
}

/** @brief Tests that a rank-3 product yields [B, m, n] under each of the four flag pairs. */
TEST(Unit_BatchedMatmul, FourFlagCombinationsYieldBatchedProducts)
{
    for (int flags = 0; flags < 4; ++flags)
    {
        const bool transpose_a = (flags & 1) != 0;
        const bool transpose_b = (flags & 2) != 0;

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        // A is [B, m, k] or [B, k, m]; B is [B, k, n] or [B, n, k]; here m=3, k=4, n=5, B=2.
        const ValueId a = builder.input(transpose_a ? Shape{2, 4, 3} : Shape{2, 3, 4}, "a");
        const ValueId b = builder.input(transpose_b ? Shape{2, 5, 4} : Shape{2, 4, 5}, "b");

        const ValueId out = builder.matmul(a, b, transpose_a, transpose_b);

        EXPECT_EQ(graph.value(out).shape, Shape({2, 3, 5})) << "flags " << flags;
        const Graph::Node& node = producer_of(graph, out);
        EXPECT_EQ(node.op, Ops::K_MATMUL.id) << "flags " << flags;
        expect_reads_and_writes(node, out, {a, b});
        EXPECT_EQ(node.attributes.flag(0), transpose_a) << "flags " << flags;
        EXPECT_EQ(node.attributes.flag(1), transpose_b) << "flags " << flags;
    }
}

/** @brief Tests that a rank-3 product takes its batch count from the operands' shared axis. */
TEST(Unit_BatchedMatmul, BatchedProductTakesTheSharedBatchCount)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId a = builder.input(Shape{7, 2, 3}, "a");
    const ValueId b = builder.input(Shape{7, 3, 6}, "b");

    const ValueId out = builder.matmul(a, b);

    EXPECT_EQ(graph.value(out).shape, Shape({7, 2, 6}));
    EXPECT_EQ(graph.value(out).layout, graph.value(a).layout);
    EXPECT_EQ(graph.value(out).dtype, DType::Float32);
}

/** @brief Tests that a product refuses operands whose ranks differ or are neither 2 nor 3. */
TEST(Unit_BatchedMatmul, RefusesMixedRanksAndRanksOutsideTwoAndThree)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId matrix = builder.input(Shape{3, 4}, "matrix");
    const ValueId batch = builder.input(Shape{2, 4, 5}, "batch");
    const ValueId vector = builder.input(Shape{4}, "vector");
    const ValueId quad_a = builder.input(Shape{2, 3, 4, 1}, "quad_a");
    const ValueId quad_b = builder.input(Shape{2, 4, 5, 1}, "quad_b");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.matmul(matrix, batch), Error);
    EXPECT_THROW((void)builder.matmul(batch, matrix), Error);
    EXPECT_THROW((void)builder.matmul(vector, batch), Error);
    EXPECT_THROW((void)builder.matmul(quad_a, quad_b), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that a rank-3 product refuses operands whose batches or inner extents differ. */
TEST(Unit_BatchedMatmul, RefusesUnequalBatchesAndInnerExtents)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId a = builder.input(Shape{2, 3, 4}, "a");
    const ValueId other_batch = builder.input(Shape{3, 4, 5}, "other_batch");
    const ValueId wrong_inner = builder.input(Shape{2, 5, 3}, "wrong_inner");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.matmul(a, other_batch), Error);
    EXPECT_THROW((void)builder.matmul(a, wrong_inner), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that softmax emits RESHAPE, LOG_SOFTMAX, EXP and RESHAPE, in that order. */
TEST(Unit_BatchedMatmul, SoftmaxEmitsReshapeLogSoftmaxExpReshape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{2, 3, 3}, "x");
    const std::size_t nodes_before = graph.node_count();

    const ValueId out = builder.softmax(x);

    ASSERT_EQ(graph.node_count(), nodes_before + 4);
    const Graph::Node& last = producer_of(graph, out);
    EXPECT_EQ(last.op, Ops::K_RESHAPE.id);
    EXPECT_EQ(graph.value(out).shape, Shape({2, 3, 3}));

    const ValueId exponentials = last.inputs[0];
    const Graph::Node& exp_node = producer_of(graph, exponentials);
    EXPECT_EQ(exp_node.op, Ops::K_EXP.id);

    const ValueId normalised = exp_node.inputs[0];
    const Graph::Node& log_node = producer_of(graph, normalised);
    EXPECT_EQ(log_node.op, Ops::K_LOG_SOFTMAX.id);
    EXPECT_EQ(graph.value(normalised).shape, Shape({6, 3}));

    const ValueId flat = log_node.inputs[0];
    const Graph::Node& flatten = producer_of(graph, flat);
    EXPECT_EQ(flatten.op, Ops::K_RESHAPE.id);
    EXPECT_EQ(flatten.inputs, std::vector<ValueId>{x});
    EXPECT_EQ(graph.value(flat).shape, Shape({6, 3}));
}

/** @brief Tests that a rank-2 softmax keeps its [rows, last] extents through the flattening. */
TEST(Unit_BatchedMatmul, SoftmaxOverRank2KeepsItsRowsAndLastAxis)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{4, 5}, "x");

    const ValueId out = builder.softmax(x);

    EXPECT_EQ(graph.value(out).shape, Shape({4, 5}));
    const Graph::Node& exp_node = producer_of(graph, producer_of(graph, out).inputs[0]);
    EXPECT_EQ(exp_node.op, Ops::K_EXP.id);
    EXPECT_EQ(graph.value(exp_node.inputs[0]).shape, Shape({4, 5}));
}

/** @brief Tests that softmax refuses an operand of rank below two. */
TEST(Unit_BatchedMatmul, SoftmaxRefusesRankBelowTwo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId vector = builder.input(Shape{5}, "vector");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.softmax(vector), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

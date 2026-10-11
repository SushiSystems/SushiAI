/**************************************************************************/
/* test_block_rotary_embedding.cpp                                        */
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
 * @file test_block_rotary_embedding.cpp
 * @brief Tests the two traces, the tables and the refusals of RotaryEmbedding.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Language::Blocks::RotaryEmbedding;
using NN::Trace;

namespace
{
    /** @brief Holds the channels every case turns. */
    constexpr int64_t K_WIDTH = 8;

    /** @brief Holds the groups K_WIDTH is cut into. */
    constexpr int64_t K_GROUPS = 2;

    /** @brief Holds the pairs of a group of K_WIDTH / K_GROUPS channels. */
    constexpr int64_t K_PAIRS = 2;

    /** @brief Holds T, the positions of the sequence the window cases read. */
    constexpr int64_t K_LENGTH = 5;

    /** @brief Holds the nodes a trace adds besides the two leaves of its angles. */
    constexpr std::size_t K_TURN_NODES = 18;

    /** @brief Bounds a table entry against the host's cosine or sine, absolutely. */
    constexpr double K_TOLERANCE = 1e-15;

    /** @brief Counts the nodes of @p graph that carry @p op. */
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

/** @brief Tests that a trace adds two table leaves and eighteen nodes, two of them affine. */
TEST(Unit_BlockRotaryEmbedding, ATraceWeighsTheValueAndItsSwappedHalves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    const std::size_t nodes_before = graph.node_count();

    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, K_LENGTH, K_WIDTH}));
    EXPECT_EQ(graph.node_count(), nodes_before + 2 + K_TURN_NODES);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);

    // A weight per position and channel stands as one channel of a [N, 1, 1, T * C] view.
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op != Ops::K_CHANNEL_AFFINE.id)
            continue;

        EXPECT_EQ(graph.value(node.inputs[0]).shape, (Shape{2, 1, 1, K_LENGTH * K_WIDTH}));
        EXPECT_EQ(graph.value(node.inputs[1]).shape, (Shape{K_LENGTH * K_WIDTH}));
        EXPECT_FALSE(graph.value(node.inputs[1]).requires_grad);
    }
}

/** @brief Tests that the two tables hold the cosine and the sine of every row and pair. */
TEST(Unit_BlockRotaryEmbedding, TheTablesHoldTheAnglesOfEveryPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x")));
    SA_EXPECT_OK(trace.status());

    ASSERT_EQ(tables.tables().size(), 2u);
    const NN::HostTable& cosines = tables.tables()[0];
    const NN::HostTable& sines = tables.tables()[1];
    EXPECT_EQ(graph.value(cosines.leaf).shape, (Shape{K_LENGTH, K_PAIRS}));
    EXPECT_EQ(graph.value(sines.leaf).shape, (Shape{K_LENGTH, K_PAIRS}));
    EXPECT_EQ(graph.value(cosines.leaf).dtype, DType::Float32);
    ASSERT_EQ(cosines.values.size(), static_cast<std::size_t>(K_LENGTH * K_PAIRS));
    ASSERT_EQ(sines.values.size(), cosines.values.size());

    for (int64_t row = 0; row < K_LENGTH; ++row)
    {
        for (int64_t pair = 0; pair < K_PAIRS; ++pair)
        {
            const double angle = Language::Blocks::rotary_angle(row, pair, K_PAIRS);
            const std::size_t at = static_cast<std::size_t>(row * K_PAIRS + pair);
            EXPECT_NEAR(cosines.values[at], std::cos(angle), K_TOLERANCE) << "element " << at;
            EXPECT_NEAR(sines.values[at], std::sin(angle), K_TOLERANCE) << "element " << at;
        }
    }
}

/** @brief Tests that a float64 value reads float64 tables and adds no cast. */
TEST(Unit_BlockRotaryEmbedding, ADoubleValueReadsDoubleTables)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x =
        builder.input(Shape{1, K_LENGTH, K_WIDTH}, "x", DType::Float64);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
    ASSERT_EQ(tables.tables().size(), 2u);
    EXPECT_EQ(graph.value(tables.tables()[0].leaf).dtype, DType::Float64);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAST.id), 0u);
}

/** @brief Tests that the block lists no parameter and no buffer. */
TEST(Unit_BlockRotaryEmbedding, TheBlockHoldsNoState)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x")));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(block, "rotary", visitor);

    EXPECT_TRUE(names.empty());
    EXPECT_TRUE(graph.state_leaves().empty());
    EXPECT_EQ(block.output_shape(Shape{K_LENGTH, K_WIDTH}), (Shape{K_LENGTH, K_WIDTH}));
}

/** @brief Tests that one position is fed one row of cosines and one of sines, and no table. */
TEST(Unit_BlockRotaryEmbedding, TheIncrementalFormIsFedTheAnglesOfItsPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y =
        block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH});

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_WIDTH}));
    EXPECT_EQ(graph.node_count(), nodes_before + 2 + K_TURN_NODES);
    EXPECT_TRUE(carried.values().empty());

    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_ROTARY_COSINE_LEAF);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ROTARY_SINE_LEAF);
    for (const NN::FedLeaf& leaf : fed.leaves())
    {
        EXPECT_EQ(graph.value(leaf.value).shape, (Shape{1, K_PAIRS}));
        EXPECT_EQ(graph.value(leaf.value).dtype, DType::Float32);
    }
}

/** @brief Tests that widths that leave no pair, another input and a plain trace are refused. */
TEST(Unit_BlockRotaryEmbedding, ABlockOrAnInputThatDoesNotFitIsRefused)
{
    SA_EXPECT_OK(Language::Blocks::check_rotary_widths(K_WIDTH, K_GROUPS));
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(6, 2),
                      StatusCode::InvalidDescription, "pairs", "width=6 groups=2");
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(8, 3),
                      StatusCode::InvalidDescription, "width=8 groups=3");
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(0, 0),
                      StatusCode::InvalidDescription, "width=0 groups=0");

    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH + 2}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        NN::HostTableList tables;
        Trace trace = Trace::evaluation(builder);
        trace.collect_tables(tables);

        EXPECT_EQ(block.forward(trace, x), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "RotaryEmbedding",
                          "[2, 5, 10]");
        EXPECT_TRUE(tables.tables().empty());
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        const Trace trace = Trace::evaluation(builder);

        EXPECT_EQ(block.forward(trace, x), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                          "collect_tables");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        const Trace trace = Trace::evaluation(builder);

        EXPECT_EQ(block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH}), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                          "carrying trace");
    }
}

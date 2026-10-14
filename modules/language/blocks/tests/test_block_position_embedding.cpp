/**************************************************************************/
/* test_block_position_embedding.cpp                                      */
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
 * @file test_block_position_embedding.cpp
 * @brief Tests PositionEmbedding's trace, its refusals, its state and its node factory.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Language::Blocks::PositionEmbedding;
using NN::Trace;

namespace
{
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the positions of the sequence every case reads. */
    constexpr int64_t K_LENGTH = 5;

    /**
     * @brief Holds the nodes of a trace whose table has a row per position: the leaf, the
     *        two views of the operands, the sum and the view of the result.
     */
    constexpr std::size_t K_WHOLE_TABLE_NODES = 5;

    /** @brief Holds the nodes of a trace that cuts the table: two more views and the cut. */
    constexpr std::size_t K_CUT_TABLE_NODES = 7;
} // namespace

/** @brief Tests that a table of T rows is added to the [N, T * dim] rows in one BIAS_ADD. */
TEST(Unit_BlockPositionEmbedding, ATraceAddsTheTableToEveryRow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, K_LENGTH, K_DIM}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_WHOLE_TABLE_NODES);
    EXPECT_EQ(block.weight.shape(), (Shape{K_LENGTH, K_DIM}));
    EXPECT_EQ(block.weight.role(), NN::ParameterRole::Embedding);

    const Graph::Node& restored = graph.node(graph.value(y).producer);
    ASSERT_EQ(restored.op, Ops::K_RESHAPE.id);
    const Graph::Node& sum = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(sum.op, Ops::K_BIAS_ADD.id);
    EXPECT_EQ(graph.value(sum.inputs[0]).shape, (Shape{2, K_LENGTH * K_DIM}));
    EXPECT_EQ(graph.value(sum.inputs[1]).shape, (Shape{K_LENGTH * K_DIM}));
}

/** @brief Tests that a sequence shorter than the table reads the table's leading rows. */
TEST(Unit_BlockPositionEmbedding, AShorterSequenceReadsTheLeadingRows)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH + 3, K_DIM);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.node_count(), nodes_before + K_CUT_TABLE_NODES);
    EXPECT_EQ(block.weight.shape(), (Shape{K_LENGTH + 3, K_DIM}));

    const Graph::Node& restored = graph.node(graph.value(y).producer);
    const Graph::Node& sum = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(sum.op, Ops::K_BIAS_ADD.id);
    const Graph::Node& offsets = graph.node(graph.value(sum.inputs[1]).producer);
    ASSERT_EQ(offsets.op, Ops::K_RESHAPE.id);
    const Graph::Node& cut = graph.node(graph.value(offsets.inputs[0]).producer);
    ASSERT_EQ(cut.op, Ops::K_SLICE_CHANNELS.id);
    EXPECT_EQ(graph.value(cut.outputs[0]).shape, (Shape{1, 1, 1, K_LENGTH * K_DIM}));
    EXPECT_EQ(cut.attributes.values[0], 0.0);
}

/** @brief Tests that a sequence longer than the table, and a misfit input, fail the trace. */
TEST(Unit_BlockPositionEmbedding, ASequenceLongerThanTheTableIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_DIM}, "x");

    PositionEmbedding shorter(K_LENGTH - 1, K_DIM);
    const Trace long_trace = Trace::evaluation(builder);
    EXPECT_EQ(shorter.forward(long_trace, x), x);
    SA_EXPECT_FAILURE(long_trace.status(), StatusCode::InvalidDescription, "5 positions",
                      "4 rows");
    EXPECT_FALSE(shorter.weight.declared());

    PositionEmbedding wider(K_LENGTH, K_DIM + 1);
    const Trace wide_trace = Trace::evaluation(builder);
    EXPECT_EQ(wider.forward(wide_trace, x), x);
    SA_EXPECT_FAILURE(wide_trace.status(), StatusCode::InvalidDescription, "PositionEmbedding",
                      "[N, T, 7]");

    PositionEmbedding undescribed;
    const Trace empty_trace = Trace::evaluation(builder);
    EXPECT_EQ(undescribed.forward(empty_trace, x), x);
    SA_EXPECT_FAILURE(empty_trace.status(), StatusCode::InvalidDescription, "positive extents");
}

/** @brief Tests that the walk names the one table. */
TEST(Unit_BlockPositionEmbedding, TheWalkYieldsTheTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    PositionEmbedding block(K_LENGTH, K_DIM);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, K_LENGTH, K_DIM}, "x")));

    std::vector<std::string> names;
    NN::for_each_parameter(block, "positions", [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    });

    EXPECT_EQ(names, (std::vector<std::string>{"positions.weight"}));
}

/** @brief Tests that the factory takes T for an absent "length" and refuses one below T. */
TEST(Unit_BlockPositionEmbedding, TheFactoryReadsTheLength)
{
    const Shape input{K_LENGTH, K_DIM};
    const Shape flat{K_DIM};
    const Config::Value absent = Config::parse_json("{}", "<test>");
    const Config::Value longer = Config::parse_json(R"({"length": 9})", "<test>");
    const Config::Value shorter = Config::parse_json(R"({"length": 4})", "<test>");
    const Config::Value unknown = Config::parse_json(R"({"rows": 9})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), absent,
                                                       "layer 1"));
    const auto* wrapped = dynamic_cast<const NN::ModuleNode<PositionEmbedding>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().length, K_LENGTH);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, input);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> long_node,
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), longer,
                                                       "layer 1"));
    const auto* long_wrapped =
        dynamic_cast<const NN::ModuleNode<PositionEmbedding>*>(long_node.get());
    ASSERT_NE(long_wrapped, nullptr);
    EXPECT_EQ(long_wrapped->layer().length, 9);

    SA_EXPECT_FAILURE(
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), shorter,
                                                       "layer 1"),
        StatusCode::InvalidDescription, "layer 1", "'length' is 4", "5 positions");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), unknown,
                                                       "layer 1"),
        StatusCode::InvalidDescription, "layer 1", "'rows'");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&flat, 1), absent,
                                                       "layer 1"),
        StatusCode::InvalidDescription, "layer 1", "rank-2 [T, C] input");
}

namespace
{
    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
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

/** @brief Tests that one position gathers the row of the fed position and adds it to the slice. */
TEST(Unit_BlockPositionEmbedding, TheIncrementalFormAddsTheRowOfTheFedPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{K_LENGTH, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(count_op(graph, Ops::K_GATHER_ROWS.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
    EXPECT_EQ(graph.value(block.weight.id_in(graph)).shape, (Shape{K_LENGTH, K_DIM}));

    ASSERT_EQ(fed.leaves().size(), 1u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_POSITION_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_TRUE(carried.values().empty());
}

/** @brief Tests that a window longer than the table and a longer slice are each refused. */
TEST(Unit_BlockPositionEmbedding, TheIncrementalFormRefusesAWindowItsTableCannotServe)
{
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
        PositionEmbedding block(K_LENGTH, K_DIM);
        NN::FedLeafList fed;
        NN::CarriedValueList carried;
        const Trace trace = Trace::carrying(builder, fed, carried);

        EXPECT_EQ(block.forward_incremental(trace, x, Shape{K_LENGTH + 1, K_DIM}), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "6 positions",
                          "5 rows");
        EXPECT_TRUE(fed.leaves().empty());
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 2, K_DIM}, "x");
        PositionEmbedding block(K_LENGTH, K_DIM);
        NN::FedLeafList fed;
        NN::CarriedValueList carried;
        const Trace trace = Trace::carrying(builder, fed, carried);

        EXPECT_EQ(block.forward_incremental(trace, x, Shape{K_LENGTH, K_DIM}), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "one position",
                          "[1, 2, 6]");
    }
}

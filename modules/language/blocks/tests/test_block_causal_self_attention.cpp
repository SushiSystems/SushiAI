/**************************************************************************/
/* test_block_causal_self_attention.cpp                                   */
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
 * @file test_block_causal_self_attention.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of CausalSelfAttention.
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
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::AttentionPositions;
using Language::Blocks::CausalSelfAttention;
using NN::Trace;

namespace
{
    /** @brief Holds the width every case attends over. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count every case splits K_DIM into. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the nodes a head adds besides its mask. */
    constexpr std::size_t K_NODES_PER_HEAD = 14;

    /** @brief Holds the nodes a trace adds besides its heads: four leaves and eleven others. */
    constexpr std::size_t K_NODES_AROUND_HEADS = 15;

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

    /** @brief Traces @p block over @p x, expects @p x back, and returns what was recorded. */
    Status refused(CausalSelfAttention& block, const Graph::GraphBuilder& builder,
                   Graph::ValueID x)
    {
        const Trace trace = Trace::evaluation(builder);
        EXPECT_EQ(block.forward(trace, x), x);
        return trace.status();
    }
}

/** @brief Tests that a causal trace adds 15 nodes around its heads and 15 per head. */
TEST(Unit_BlockCausalSelfAttention, ACausalTraceMasksEveryHead)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.node_count(),
              nodes_before + K_NODES_AROUND_HEADS + K_HEADS * (K_NODES_PER_HEAD + 1));
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 2u);
}

/** @brief Tests that each mask reads a [N, T, T] product scaled by 1 / sqrt(dim / heads). */
TEST(Unit_BlockCausalSelfAttention, EachMaskReadsTheScaledScoresOfItsHead)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 5, K_DIM}, "x")));

    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op != Ops::K_CAUSAL_MASK.id)
            continue;

        EXPECT_EQ(graph.value(node.inputs[0]).shape, (Shape{2, 5, 5}));
        EXPECT_LT(node.attributes.values[0], -1.0e8);

        const Graph::Node& scaled = graph.node(graph.value(node.inputs[0]).producer);
        ASSERT_EQ(scaled.op, Ops::K_SCALE.id);
        EXPECT_DOUBLE_EQ(scaled.attributes.values[0], 0.5);
        EXPECT_EQ(graph.node(graph.value(scaled.inputs[0]).producer).op, Ops::K_MATMUL.id);
    }
}

/** @brief Tests that a block without a mask adds no CAUSAL_MASK and one node less per head. */
TEST(Unit_BlockCausalSelfAttention, ATraceWithoutAMaskEmitsNone)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::None);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    static_cast<void>(block.forward(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
    EXPECT_EQ(graph.node_count(),
              nodes_before + K_NODES_AROUND_HEADS + K_HEADS * K_NODES_PER_HEAD);
}

/** @brief Tests that qkv is [3 * dim, dim], proj [dim, dim], and the walk names the four. */
TEST(Unit_BlockCausalSelfAttention, TheWalkYieldsTheTwoProjections)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 5, K_DIM}, "x")));

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
    NN::walk_state(block, "attn", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"attn.qkv.weight", "attn.qkv.bias",
                                               "attn.proj.weight", "attn.proj.bias"}));
    EXPECT_EQ(block.qkv.weight.shape(), (Shape{3 * K_DIM, K_DIM}));
    EXPECT_EQ(block.proj.weight.shape(), (Shape{K_DIM, K_DIM}));
}

/** @brief Tests that a rank-2 input, another width and an integer input fail and emit nothing. */
TEST(Unit_BlockCausalSelfAttention, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 2}, "wide");
    const Graph::ValueID whole = builder.input(Shape{2, 5, K_DIM}, "whole", DType::Int32);
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused(block, builder, flat), StatusCode::InvalidDescription,
                      "CausalSelfAttention", "rank-3 [N, T, 8] input", "[5, 8]");
    SA_EXPECT_FAILURE(refused(block, builder, wide), StatusCode::InvalidDescription,
                      "CausalSelfAttention", "[2, 5, 10]");
    SA_EXPECT_FAILURE(refused(block, builder, whole), StatusCode::InvalidDescription,
                      "half, float32 or float64 input", "int32");

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.qkv.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that a head count that does not divide the width, and no width, each fail. */
TEST(Unit_BlockCausalSelfAttention, ABlockDescribedWrongFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention uneven(K_DIM, 3, AttentionMask::Causal);
    CausalSelfAttention empty{};

    SA_EXPECT_FAILURE(refused(uneven, builder, x), StatusCode::InvalidDescription,
                      "head count divides", "dim=8 heads=3");
    SA_EXPECT_FAILURE(refused(empty, builder, x), StatusCode::InvalidDescription,
                      "dim=0 heads=0");
}

/** @brief Tests that the factory takes dim from the input and reads "heads" and "causal". */
TEST(Unit_BlockCausalSelfAttention, TheFactoryReadsTheHeadsAndTheMask)
{
    const Shape input{5, K_DIM};
    const Config::Value plain = Config::parse_json(R"({"heads": 4})", "<test>");
    const Config::Value open = Config::parse_json(R"({"heads": 2, "causal": false})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), plain,
                                                          "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> encoder,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), open,
                                                          "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(node.get());
    const auto* wrapped_open =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(encoder.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_open, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().heads, 4);
    EXPECT_EQ(wrapped->layer().mask, AttentionMask::Causal);
    EXPECT_EQ(wrapped_open->layer().mask, AttentionMask::None);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
}

/** @brief Tests that the factory refuses a stray key, no heads, uneven heads and a bad input. */
TEST(Unit_BlockCausalSelfAttention, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{5, K_DIM};
    const Shape map{4, 4, K_DIM};
    const Config::Value good = Config::parse_json(R"({"heads": 2})", "<test>");
    const Config::Value stray = Config::parse_json(R"({"heads": 2, "dropout": 0.1})", "<test>");
    const Config::Value absent = Config::parse_json("{}", "<test>");
    const Config::Value uneven = Config::parse_json(R"({"heads": 3})", "<test>");

    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), stray,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'dropout'");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), absent,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'heads'", "absent");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), uneven,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "dim=8 heads=3");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&map, 1), good,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "rank-2 [T, C] input", "[4, 4, 8]");
}

namespace
{
    /** @brief Holds T, the rows of the caches of every incremental case. */
    constexpr int64_t K_WINDOW = 5;

    /** @brief Returns the per-sample [T, dim] shape the block reads in its ordinary form. */
    Shape window_sample()
    {
        return Shape{K_WINDOW, K_DIM};
    }
} // namespace

/** @brief Tests that one position writes two rows and attends over the cache with no mask node. */
TEST(Unit_BlockCausalSelfAttention, TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(count_nodes(graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 4u);
}

/** @brief Tests that the block declares the position, the mask and two carried [T, dim] leaves. */
TEST(Unit_BlockCausalSelfAttention, TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    static_cast<void>(block.forward_incremental(trace, x, window_sample()));

    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_POSITION_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ATTENTION_MASK_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[1].value).shape, (Shape{1, K_WINDOW}));

    ASSERT_EQ(carried.values().size(), 2u);
    for (const NN::CarriedValue& kept : carried.values())
    {
        EXPECT_EQ(graph.value(kept.leaf).shape, (Shape{K_WINDOW, K_DIM}));
        EXPECT_EQ(graph.value(kept.next).shape, (Shape{K_WINDOW, K_DIM}));
        EXPECT_EQ(graph.value(kept.leaf).lifetime, Graph::Lifetime::Input);
        EXPECT_EQ(graph.node(graph.value(kept.next).producer).op, Ops::K_ADD.id);
    }
}

/** @brief Tests that two blocks of one trace share the fed leaves and carry four values. */
TEST(Unit_BlockCausalSelfAttention, TwoBlocksOfOneTraceShareTheirFedLeaves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention first(K_DIM, K_HEADS, AttentionMask::Causal);
    CausalSelfAttention second(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID between = first.forward_incremental(trace, x, window_sample());
    static_cast<void>(second.forward_incremental(trace, between, window_sample()));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(carried.values().size(), 4u);
}

/** @brief Tests that the incremental form declares the state the ordinary form declares. */
TEST(Unit_BlockCausalSelfAttention, TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    static_cast<void>(block.forward_incremental(trace, x, window_sample()));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(NN::parameter_count(block), 4u);
    EXPECT_EQ(NN::buffer_count(block), 0u);
}

/** @brief Tests that an unmasked block, a longer slice and a plain trace are each refused. */
TEST(Unit_BlockCausalSelfAttention, TheIncrementalFormRefusesWhatItCannotCache)
{
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
        CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::None);
        NN::FedLeafList fed;
        NN::CarriedValueList carried;
        const Trace trace = Trace::carrying(builder, fed, carried);

        EXPECT_EQ(block.forward_incremental(trace, x, window_sample()), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "no mask",
                          "no incremental form");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 3, K_DIM}, "x");
        CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
        NN::FedLeafList fed;
        NN::CarriedValueList carried;
        const Trace trace = Trace::carrying(builder, fed, carried);

        EXPECT_EQ(block.forward_incremental(trace, x, window_sample()), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "one position",
                          "[1, 3, 8]");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
        CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
        const Trace trace = Trace::evaluation(builder);

        EXPECT_EQ(block.forward_incremental(trace, x, window_sample()), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'position'",
                          "carrying trace");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
        CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
        NN::FedLeafList fed;
        NN::CarriedValueList carried;
        const Trace trace = Trace::carrying(builder, fed, carried);

        EXPECT_EQ(block.forward_incremental(trace, x, Shape{K_WINDOW}), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "per-sample",
                          "[5]");
    }
}

namespace
{
    /** @brief Holds the nodes rotary positions add to a trace: two tables and 21 others. */
    constexpr std::size_t K_ROTARY_NODES = 23;

    /** @brief Holds the pairs of a head of K_DIM / K_HEADS channels. */
    constexpr int64_t K_HEAD_PAIRS = 2;

    /** @brief Returns a causal block of K_HEADS heads over K_DIM with rotary positions. */
    CausalSelfAttention rotary_block()
    {
        return CausalSelfAttention(K_DIM, K_HEADS, AttentionMask::Causal,
                                   AttentionPositions::Rotary);
    }
} // namespace

/** @brief Tests that rotary positions turn the queries and the keys once, for every head. */
TEST(Unit_BlockCausalSelfAttention, ARotaryTraceTurnsTheQueriesAndTheKeysOnce)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();
    const std::size_t nodes_before = graph.node_count();

    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.node_count(),
              nodes_before + K_NODES_AROUND_HEADS + K_HEADS * (K_NODES_PER_HEAD + 1) +
                  K_ROTARY_NODES);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 2u);

    // One table of cosines and one of sines, a row per position and a column per pair.
    ASSERT_EQ(tables.tables().size(), 2u);
    for (const NN::HostTable& table : tables.tables())
    {
        EXPECT_EQ(graph.value(table.leaf).shape, (Shape{5, K_HEAD_PAIRS}));
        EXPECT_EQ(graph.value(table.leaf).lifetime, Graph::Lifetime::Buffer);
        EXPECT_FALSE(graph.value(table.leaf).requires_grad);
        EXPECT_EQ(table.values.size(), static_cast<std::size_t>(5 * K_HEAD_PAIRS));
    }
    EXPECT_DOUBLE_EQ(tables.tables()[0].values[0], 1.0);
    EXPECT_DOUBLE_EQ(tables.tables()[1].values[0], 0.0);
}

/** @brief Tests that the walk of a rotary block yields the two projections and no table. */
TEST(Unit_BlockCausalSelfAttention, ARotaryBlockHoldsTheStateOfALearnedOne)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    CausalSelfAttention block = rotary_block();
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, 5, K_DIM}, "x")));
    SA_EXPECT_OK(trace.status());

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
    NN::walk_state(block, "attn", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"attn.qkv.weight", "attn.qkv.bias",
                                               "attn.proj.weight", "attn.proj.bias"}));
}

/** @brief Tests that a trace no step fills tables for and an odd head are each refused. */
TEST(Unit_BlockCausalSelfAttention, ARotaryTraceRefusesWhatItCannotTurn)
{
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
        CausalSelfAttention block = rotary_block();

        SA_EXPECT_FAILURE(refused(block, builder, x), StatusCode::InvalidDescription,
                          "'rotary_cosine'", "collect_tables");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 5, 6}, "x");
        CausalSelfAttention odd(6, 2, AttentionMask::Causal, AttentionPositions::Rotary);

        SA_EXPECT_FAILURE(refused(odd, builder, x), StatusCode::InvalidDescription,
                          "rotary positions", "3 channels");
    }
}

/** @brief Tests that the factory reads "positions" and refuses a word or a head that misfits. */
TEST(Unit_BlockCausalSelfAttention, TheFactoryReadsThePositions)
{
    const Shape input{5, K_DIM};
    const Shape odd{5, 6};
    const Config::Value plain = Config::parse_json(R"({"heads": 2})", "<test>");
    const Config::Value rotary =
        Config::parse_json(R"({"heads": 2, "positions": "rotary"})", "<test>");
    const Config::Value unknown =
        Config::parse_json(R"({"heads": 2, "positions": "sinusoidal"})", "<test>");
    const Config::Value number = Config::parse_json(R"({"heads": 2, "positions": 1})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> learned,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), plain,
                                                          "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> turned,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), rotary,
                                                          "layer 0"));

    const auto* wrapped =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(learned.get());
    const auto* wrapped_turned =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(turned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_turned, nullptr);
    EXPECT_EQ(wrapped->layer().positions, AttentionPositions::Learned);
    EXPECT_EQ(wrapped_turned->layer().positions, AttentionPositions::Rotary);

    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), unknown,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'positions'", "'sinusoidal'", "'learned'",
        "'rotary'");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), number,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'positions'", "must be a string");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&odd, 1), rotary,
                                                          "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "rotary positions", "3 channels");
}

/** @brief Tests that one rotary position puts a row in place of what each ring holds. */
TEST(Unit_BlockCausalSelfAttention, TheRotaryIncrementalFormReplacesARowOfEachRing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));

    // One scatter marks the row and one per ring places it; a select takes the place of the add.
    EXPECT_EQ(count_nodes(graph, Ops::K_SCATTER_ADD_ROWS.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SELECT.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);

    ASSERT_EQ(carried.values().size(), 2u);
    for (const NN::CarriedValue& kept : carried.values())
    {
        EXPECT_EQ(graph.value(kept.leaf).shape, (Shape{K_WINDOW, K_DIM}));
        const Graph::Node& written = graph.node(graph.value(kept.next).producer);
        ASSERT_EQ(written.op, Ops::K_SELECT.id);
        EXPECT_EQ(written.inputs[2], kept.leaf);
    }
}

/** @brief Tests that a rotary position is fed its slot, the mask and its angles, no position. */
TEST(Unit_BlockCausalSelfAttention, TheRotaryIncrementalFormIsFedASlotAndTheAngles)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();
    CausalSelfAttention second = rotary_block();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());
    static_cast<void>(second.forward_incremental(trace, y, window_sample()));

    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(fed.leaves().size(), 4u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_CACHE_SLOT_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ATTENTION_MASK_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[1].value).shape, (Shape{1, K_WINDOW}));
    EXPECT_EQ(fed.leaves()[2].name, Language::Blocks::K_ROTARY_COSINE_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[2].value).shape, (Shape{1, K_HEAD_PAIRS}));
    EXPECT_EQ(fed.leaves()[3].name, Language::Blocks::K_ROTARY_SINE_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[3].value).shape, (Shape{1, K_HEAD_PAIRS}));
    EXPECT_EQ(carried.values().size(), 4u);
}

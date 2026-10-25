/**************************************************************************/
/* test_causal_self_attention.cpp                                         */
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
 * @file test_causal_self_attention.cpp
 * @brief Tests causal self-attention: its trace, its mask and gradients on an engine, and a
 *        transformer block over it.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::AttentionPositions;
using Language::Blocks::CausalSelfAttention;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;
using Language::Blocks::TransformerBlock;

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
TEST(Medium_BlockCausalSelfAttention, ACausalTraceMasksEveryHead)
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
TEST(Medium_BlockCausalSelfAttention, EachMaskReadsTheScaledScoresOfItsHead)
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
TEST(Medium_BlockCausalSelfAttention, ATraceWithoutAMaskEmitsNone)
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
TEST(Medium_BlockCausalSelfAttention, TheWalkYieldsTheTwoProjections)
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
TEST(Medium_BlockCausalSelfAttention, AnInputThatDoesNotFitFails)
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
TEST(Medium_BlockCausalSelfAttention, ABlockDescribedWrongFails)
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
TEST(Medium_BlockCausalSelfAttention, TheFactoryReadsTheHeadsAndTheMask)
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
TEST(Medium_BlockCausalSelfAttention, TheFactoryRefusesARowThatDoesNotFit)
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
TEST(Medium_BlockCausalSelfAttention, TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow)
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
TEST(Medium_BlockCausalSelfAttention, TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues)
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
TEST(Medium_BlockCausalSelfAttention, TwoBlocksOfOneTraceShareTheirFedLeaves)
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
TEST(Medium_BlockCausalSelfAttention, TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm)
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
TEST(Medium_BlockCausalSelfAttention, TheIncrementalFormRefusesWhatItCannotCache)
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
TEST(Medium_BlockCausalSelfAttention, ARotaryTraceTurnsTheQueriesAndTheKeysOnce)
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
TEST(Medium_BlockCausalSelfAttention, ARotaryBlockHoldsTheStateOfALearnedOne)
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
TEST(Medium_BlockCausalSelfAttention, ARotaryTraceRefusesWhatItCannotTurn)
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
TEST(Medium_BlockCausalSelfAttention, TheFactoryReadsThePositions)
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
TEST(Medium_BlockCausalSelfAttention, TheRotaryIncrementalFormReplacesARowOfEachRing)
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
TEST(Medium_BlockCausalSelfAttention, TheRotaryIncrementalFormIsFedASlotAndTheAngles)
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

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_AttentionGradient : public SaTest::EngineTest
    {

    };

    /** @brief Bounds the distance of a FLOAT64 gradient from its closed form on the host. */
    constexpr double K_CLOSED_FORM_TOLERANCE = 1e-12;

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr int64_t K_GRADIENT_LENGTH = 4;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr int64_t K_GRADIENT_DIM = 4;

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_gradient_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++found;
        }

        return found;
    }

    /** @brief Names the leaf a probe differentiates at and the [N, C] value its loss reads. */
    struct Subject
    {
        ValueID variable = Graph::K_NO_VALUE; /**< The leaf the gradient is taken at. */
        ValueID logits = Graph::K_NO_VALUE;   /**< The [N, C] value the cross-entropy reads. */
    };

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     *
     * The loss is the cross-entropy of the logits the build function names, against targets
     * whose row r names class r mod C. Every other leaf is filled from the host stream.
     */
    class Probe
    {
        public:
            /** @brief Builds the graph, differentiates it at its variable and compiles it. */
            template<typename BuildFn>
            Probe(SushiBLAS::Engine& engine, std::uint64_t seed, BuildFn&& build)
                : pool_(engine)
            {
                const GraphBuilder builder(graph_);
                const Subject subject = build(builder);
                variable_ = subject.variable;

                const Shape logit_shape = graph_.value(subject.logits).shape;
                targets_ = builder.input(logit_shape, "targets", DType::Float64);
                loss_ = builder.nll_loss(builder.log_softmax(subject.logits), targets_);

                const std::vector<ValueID> wrt = {variable_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_other_leaves(seed, logit_shape);

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();
            }

            /** @brief Returns the graph the probe built, with its gradient nodes. */
            const Graph::Graph& graph() const noexcept
            {
                return graph_;
            }

            /** @brief Returns how many elements the variable holds. */
            std::size_t variable_size() const
            {
                return static_cast<std::size_t>(graph_.value(variable_).shape.num_elements());
            }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            /** @brief Writes the one-hot targets, and noise into each leaf but the variable. */
            void fill_other_leaves(std::uint64_t seed, const Shape& logit_shape)
            {
                std::vector<double> one_hot(
                    static_cast<std::size_t>(logit_shape.num_elements()), 0.0);
                for (int64_t row = 0; row < logit_shape[0]; ++row)
                    one_hot[static_cast<std::size_t>(row * logit_shape[1]
                                                     + row % logit_shape[1])] = 1.0;
                write_tensor(pool_.tensor(targets_), one_hot);

                for (ValueID id = 0; id < graph_.value_count(); ++id)
                {
                    const Graph::Lifetime lifetime = graph_.value(id).lifetime;
                    const bool is_leaf = lifetime == Graph::Lifetime::Parameter
                                      || lifetime == Graph::Lifetime::Input;
                    if (!is_leaf || id == variable_ || id == targets_)
                        continue;

                    const std::size_t count =
                        static_cast<std::size_t>(graph_.value(id).shape.num_elements());
                    write_tensor(pool_.tensor(id), noise(count, seed + id));
                }
            }

            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID variable_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Runs the full-Jacobian check of @p probe at a point drawn from @p seed. */
    void expect_gradient_matches(Probe& probe, std::uint64_t seed)
    {
        std::vector<double> point = noise(probe.variable_size(), seed);
        const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at)
            {
                return probe.value(at);
            },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            {
                probe.gradient(at, out);
            },
            point);

        EXPECT_TRUE(result.passed) << result.to_string();
    }

    /** @brief Names which leaf of an attention probe the gradient is taken at. */
    enum class AttentionVariable : std::uint8_t
    {
        Input,            /**< The [1, K_GRADIENT_LENGTH, K_GRADIENT_DIM] sequence the block reads. */
        ProjectionWeight, /**< The weight of qkv, which every score and every value reads. */
    };

    /**
     * @brief Traces @p block over a FLOAT64 [1, K_GRADIENT_LENGTH, K_GRADIENT_DIM] sequence and names its
     *        [K_GRADIENT_LENGTH, K_GRADIENT_DIM] output rows as the logits.
     */
    Subject attention_subject(const GraphBuilder& builder, CausalSelfAttention& block,
                              AttentionVariable variable)
    {
        const Shape sequence{1, K_GRADIENT_LENGTH, K_GRADIENT_DIM};
        const ValueID x = variable == AttentionVariable::Input
            ? builder.parameter(sequence, "x", DType::Float64)
            : builder.input(sequence, "x", DType::Float64);

        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        EXPECT_FALSE(builder.graph().has_refusal());

        const ValueID at = variable == AttentionVariable::Input
            ? x
            : block.qkv.weight.id_in(builder.graph());
        return Subject{at, builder.reshape(y, Shape{K_GRADIENT_LENGTH, K_GRADIENT_DIM})};
    }

    /** @brief Checks the gradient of a block with @p mask at @p variable. */
    void expect_attention_gradient_matches(SushiBLAS::Engine& engine, AttentionMask mask,
                                           AttentionVariable variable, std::uint64_t seed)
    {
        CausalSelfAttention block(K_GRADIENT_DIM, K_HEADS, mask);
        Probe probe(engine, seed, [&block, variable](const GraphBuilder& builder)
        {
            return attention_subject(builder, block, variable);
        });

        // Each head's softmax is one LOG_SOFTMAX, and none of them feeds a cross-entropy.
        EXPECT_EQ(count_gradient_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id),
                  static_cast<std::size_t>(K_HEADS));
        ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, seed + 1000));
    }
}

/**
 * @brief Tests the unfused rule against its closed form, with targets that are not one-hot.
 *
 * A SCALE by one sits between LOG_SOFTMAX and NLL_LOSS, so the fused rule does not apply and
 * the gradient is (softmax(x) * rowsum(Y) - Y) / N, which the host computes.
 */
TEST_F(Medium_AttentionGradient, AnUnfusedLogSoftmaxMatchesItsClosedForm)
{
    constexpr std::size_t K_ROWS = 3;
    constexpr std::size_t K_CLASSES = 4;
    const Shape shape{static_cast<int64_t>(K_ROWS), static_cast<int64_t>(K_CLASSES)};

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.parameter(shape, "x", DType::Float64);
    const ValueID targets = builder.input(shape, "targets", DType::Float64);
    const ValueID detoured = builder.scale(builder.log_softmax(x), 1.0);
    const ValueID loss = builder.nll_loss(detoured, targets);

    const std::vector<ValueID> wrt = {x};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);
    ASSERT_EQ(count_gradient_nodes(graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);

    // The weights are positive and their rows do not sum to one.
    const std::vector<double> values = noise(K_ROWS * K_CLASSES, 7101);
    std::vector<double> weights = noise(K_ROWS * K_CLASSES, 7102);
    for (double& weight : weights)
        weight = std::abs(weight) + 0.25;

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(targets), weights);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());

    const std::vector<double> got = read_tensor(pool.tensor(gradients[0]));
    ASSERT_EQ(got.size(), K_ROWS * K_CLASSES);
    for (std::size_t row = 0; row < K_ROWS; ++row)
    {
        double total = 0.0;
        double weight_sum = 0.0;
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            total += std::exp(values[row * K_CLASSES + column]);
            weight_sum += weights[row * K_CLASSES + column];
        }

        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            const std::size_t i = row * K_CLASSES + column;
            const double expected = (std::exp(values[i]) / total * weight_sum - weights[i])
                                  / static_cast<double>(K_ROWS);
            EXPECT_NEAR(got[i], expected, K_CLOSED_FORM_TOLERANCE)
                << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests the gradient of a [3, 4] softmax that a matrix product reads. */
TEST_F(Medium_AttentionGradient, ASoftmaxGradientMatchesCentralDifferences)
{
    Probe probe(*engine, 7200, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{3, 4}, "x", DType::Float64);
        const ValueID mix = builder.input(Shape{4, 3}, "mix", DType::Float64);
        return Subject{x, builder.matmul(builder.softmax(x), mix)};
    });

    EXPECT_EQ(count_gradient_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7201));
}

/** @brief Tests the gradient of a softmax over the last axis of a [2, 3, 4] value. */
TEST_F(Medium_AttentionGradient, ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes)
{
    Probe probe(*engine, 7300, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{2, 3, 4}, "x", DType::Float64);
        return Subject{x, builder.reshape(builder.softmax(x), Shape{6, 4})};
    });

    EXPECT_EQ(count_gradient_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7301));
}

/** @brief Tests the gradient of a causal attention block at the sequence it reads. */
TEST_F(Medium_AttentionGradient, ACausalAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::Causal, AttentionVariable::Input, 7400));
}

/** @brief Tests the gradient of a causal attention block at the weight of its qkv. */
TEST_F(Medium_AttentionGradient, ACausalAttentionDifferentiatesAtItsProjectionWeight)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::Causal, AttentionVariable::ProjectionWeight, 7500));
}

/** @brief Tests the gradient of an attention block without the mask at the sequence it reads. */
TEST_F(Medium_AttentionGradient, AnUnmaskedAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::None, AttentionVariable::Input, 7600));
}

namespace
{
    /** @brief Provides the Engine every causal case records on. */
    class Medium_CausalAttention : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a value that must be the same number in both runs. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Holds the value the mask cases write above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr std::size_t K_MASK_LENGTH = 5;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr std::size_t K_MASK_DIM = 8;

    /** @brief Holds the first position whose input the block cases change. */
    constexpr std::size_t K_CHANGED_POSITION = 3;

    /** @brief Returns @p scores, [matrices, length, length], with the fill above each diagonal. */
    std::vector<double> masked_on_host(std::vector<double> scores, std::size_t length,
                                       double fill)
    {
        for (std::size_t i = 0; i < scores.size(); ++i)
        {
            const std::size_t row = (i / length) % length;
            const std::size_t column = i % length;
            if (column > row)
                scores[i] = fill;
        }

        return scores;
    }

    /** @brief Runs the mask over a [2, 4, 4] value of @p dtype and compares with the host. */
    void expect_mask_matches_host(SushiBLAS::Engine& engine, DType dtype)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID scores = builder.input(Shape{2, 4, 4}, "scores", dtype);
        const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
        SA_EXPECT_OK(masked);
        ASSERT_TRUE(masked.has_value());

        // Dyadic values survive the FLOAT32 round trip, so the kept entries compare exactly.
        std::vector<double> values(2 * 4 * 4);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = 0.25 * static_cast<double>(i) - 3.0;

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(scores), values);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(masked.value()));
        const std::vector<double> expected = masked_on_host(values, 4, K_FILL);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], K_EXACT_TOLERANCE) << "at element " << i;
    }

    /** @brief Holds what two runs of one block gave: before and after a later input changed. */
    struct TwoRuns
    {
        std::vector<double> before; /**< The [K_MASK_LENGTH, K_MASK_DIM] output of the first input. */
        std::vector<double> after;  /**< The output once positions from 3 on were changed. */
    };

    /**
     * @brief Traces @p block over a [1, K_MASK_LENGTH, K_MASK_DIM] input, fills every parameter from
     *        the host stream, and runs it on an input and on one changed from position 3 on.
     */
    template<typename Block>
    void run_twice(SushiBLAS::Engine& engine, Block& block, TwoRuns& runs)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = builder.input(
            Shape{1, static_cast<int64_t>(K_MASK_LENGTH), static_cast<int64_t>(K_MASK_DIM)}, "x");
        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        ASSERT_FALSE(graph.has_refusal());

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        for (ValueID id = 0; id < graph.value_count(); ++id)
        {
            if (graph.value(id).lifetime != Graph::Lifetime::Parameter)
                continue;

            const std::size_t count =
                static_cast<std::size_t>(graph.value(id).shape.num_elements());
            write_tensor(pool.tensor(id), noise(count, 6100 + id));
        }

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));

        std::vector<double> input = noise(K_MASK_LENGTH * K_MASK_DIM, 6001);
        write_tensor(pool.tensor(x), input);
        SA_EXPECT_OK(step->run());
        runs.before = read_tensor(pool.tensor(y));

        const std::vector<double> later = noise(K_MASK_LENGTH * K_MASK_DIM, 6002);
        for (std::size_t i = K_CHANGED_POSITION * K_MASK_DIM; i < input.size(); ++i)
            input[i] = later[i];

        write_tensor(pool.tensor(x), input);
        SA_EXPECT_OK(step->run());
        runs.after = read_tensor(pool.tensor(y));
    }

    /** @brief Counts the elements of positions [first, last) at which the two runs differ. */
    std::size_t changed_elements(const TwoRuns& runs, std::size_t first, std::size_t last)
    {
        std::size_t changed = 0;
        for (std::size_t i = first * K_MASK_DIM; i < last * K_MASK_DIM; ++i)
        {
            if (runs.before[i] != runs.after[i])
                ++changed;
        }

        return changed;
    }

    /** @brief Expects the two runs to hold K_MASK_LENGTH * K_MASK_DIM finite elements each. */
    void expect_whole_and_finite(const TwoRuns& runs)
    {
        ASSERT_EQ(runs.before.size(), K_MASK_LENGTH * K_MASK_DIM);
        ASSERT_EQ(runs.after.size(), K_MASK_LENGTH * K_MASK_DIM);
        for (const double value : runs.before)
            ASSERT_TRUE(std::isfinite(value));
    }
}

/** @brief Tests that the mask keeps the lower triangle and writes the fill above, in FLOAT32. */
TEST_F(Medium_CausalAttention, TheMaskReplacesTheEntriesAboveTheDiagonal)
{
    ASSERT_NO_FATAL_FAILURE(expect_mask_matches_host(*engine, DType::Float32));
}

/** @brief Tests the same in FLOAT64, where the mask vectors are converted from FLOAT32. */
TEST_F(Medium_CausalAttention, TheMaskReplacesTheEntriesAboveTheDiagonalInFloat64)
{
    ASSERT_NO_FATAL_FAILURE(expect_mask_matches_host(*engine, DType::Float64));
}

/**
 * @brief Tests that the gradient of the scores is the incoming gradient on and below the
 *        diagonal and zero above it.
 *
 * The loss is the cross-entropy of the masked [4, 4] rows, so the incoming gradient is
 * (softmax - target) / 4, which the host computes from the masked scores.
 */
TEST_F(Medium_CausalAttention, TheMaskPassesItsGradientThroughTheKeptEntriesAlone)
{
    constexpr std::size_t K_SIDE = 4;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.parameter(Shape{4, 4}, "scores", DType::Float64);
    const ValueID targets = builder.input(Shape{4, 4}, "targets", DType::Float64);
    const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
    ASSERT_TRUE(masked.has_value());
    const ValueID loss = builder.nll_loss(builder.log_softmax(masked.value()), targets);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);

    // Row r targets class 0, which every row keeps.
    const std::vector<double> values = noise(K_SIDE * K_SIDE, 6201);
    std::vector<double> one_hot(K_SIDE * K_SIDE, 0.0);
    for (std::size_t row = 0; row < K_SIDE; ++row)
        one_hot[row * K_SIDE] = 1.0;

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(scores), values);
    write_tensor(pool.tensor(targets), one_hot);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    const std::vector<double> got = read_tensor(pool.tensor(gradients[0]));
    ASSERT_EQ(got.size(), K_SIDE * K_SIDE);
    for (std::size_t row = 0; row < K_SIDE; ++row)
    {
        double total = 0.0;
        for (std::size_t column = 0; column <= row; ++column)
            total += std::exp(values[row * K_SIDE + column]);

        for (std::size_t column = 0; column < K_SIDE; ++column)
        {
            const std::size_t i = row * K_SIDE + column;
            const double kept = (std::exp(values[i]) / total - one_hot[i]) / 4.0;
            const double expected = column <= row ? kept : 0.0;
            const double bound = column <= row ? 1e-12 : K_EXACT_TOLERANCE;
            EXPECT_NEAR(got[i], expected, bound) << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests that positions 0 to 2 of a masked attention keep their output exactly. */
TEST_F(Medium_CausalAttention, AnEarlierPositionIgnoresALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, AttentionMask::Causal);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_MASK_LENGTH), 0u)
        << "the changed positions gave the output they gave before";
}

/** @brief Tests that without the mask an earlier position does read the later input. */
TEST_F(Medium_CausalAttention, WithoutTheMaskAnEarlierPositionReadsALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, AttentionMask::None);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_GT(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
}

/** @brief Tests that a whole causal transformer block keeps the same property. */
TEST_F(Medium_CausalAttention, ATransformerBlockKeepsEarlierPositions)
{
    TransformerBlock block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, 4 * static_cast<int64_t>(K_MASK_DIM),
                           AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_MASK_LENGTH), 0u);
}

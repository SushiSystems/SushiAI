/**************************************************************************/
/* test_expert_transformer_block.cpp                                      */
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
 * @file test_expert_transformer_block.cpp
 * @brief Tests the trace, the state, the incremental form and the row of a transformer block whose
 *        feed-forward is a mixture of experts.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/expert_transformer_block.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::ExpertTrace;
using Language::Blocks::ExpertTransformerBlock;
using Language::Blocks::MoEFeedForward;
using NN::Trace;

namespace
{
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of every case. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the hidden width of one expert. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds the number of experts of every case. */
    constexpr int64_t K_EXPERTS = 4;

    /** @brief Holds the nodes of a causal two-head block: 53 around a mixture of 31. */
    constexpr std::size_t K_TRACE_NODES = 84;

    /** @brief Returns a causal block of K_EXPERTS experts that keeps @p top_k of them. */
    ExpertTransformerBlock make_block(int64_t top_k = 0)
    {
        return ExpertTransformerBlock(K_HEADS,
                                      MoEFeedForward(K_DIM, K_HIDDEN, K_EXPERTS, top_k),
                                      AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON);
    }

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

    /** @brief Builds the node of a "transformer_block" row over a [5, K_DIM] sample. */
    Result<std::unique_ptr<NN::Node>> make_row(const char* text)
    {
        const Shape input{5, K_DIM};
        return Language::Blocks::make_transformer_block_node(
            NN::ShapeSpan(&input, 1), Config::parse_json(text, "<test>"), "layer 3");
    }
}

/** @brief Tests that a trace adds the attention, then the mixture, each to what it read. */
TEST(Unit_BlockExpertTransformerBlock, ATraceAddsANormalisedAttentionThenANormalisedMixture)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    ExpertTransformerBlock block = make_block();
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(traced.balance).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_TRACE_NODES);
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_UPSAMPLE2D.id), 1u);

    // The last sum reads the first sum, and the first sum reads the block's input.
    const Graph::Node& last = graph.node(graph.value(traced.output).producer);
    ASSERT_EQ(last.op, Ops::K_ADD.id);
    const Graph::Node& first = graph.node(graph.value(last.inputs[0]).producer);
    ASSERT_EQ(first.op, Ops::K_ADD.id);
    EXPECT_EQ(first.inputs[0], x);
}

/** @brief Tests that the walk names the fourteen parameters under the four children. */
TEST(Unit_BlockExpertTransformerBlock, TheWalkYieldsTheFourChildren)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ExpertTransformerBlock block = make_block();
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
    NN::walk_state(block, "h", visitor);

    EXPECT_EQ(names,
              (std::vector<std::string>{
                  "h.norm1.weight", "h.norm1.bias", "h.attn.qkv.weight", "h.attn.qkv.bias",
                  "h.attn.proj.weight", "h.attn.proj.bias", "h.norm2.weight", "h.norm2.bias",
                  "h.ff.router.weight", "h.ff.router.bias", "h.ff.expand.weight",
                  "h.ff.expand.bias", "h.ff.project.weight", "h.ff.project.bias"}));
}

/** @brief Tests that an input of another width fails once, with the first child's message. */
TEST(Unit_BlockExpertTransformerBlock, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    ExpertTransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(trace, wide), wide);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "LayerNorm(8)",
                      "[2, 5, 9]");
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 0u);
    EXPECT_FALSE(block.ff.router.weight.declared());
}

/** @brief Tests that "experts" turns the row into a node of three ports, and its absence not. */
TEST(Unit_BlockExpertTransformerBlock, TheTransformerRowWithExpertsHasThreePorts)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> plain, make_row(R"({"heads": 2})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> none,
                            make_row(R"({"heads": 2, "experts": 0})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> routed,
                            make_row(R"({"heads": 2, "experts": 4, "top_k": 1})"));

    EXPECT_EQ(plain->ports().size(), 1u);
    EXPECT_EQ(none->ports().size(), 1u);
    ASSERT_EQ(routed->ports().size(), 3u);
    EXPECT_EQ(routed->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(routed->ports()[0].shape, (Shape{5, K_DIM}));
    EXPECT_EQ(routed->ports()[1].name, Language::Blocks::K_BALANCE_PORT);
    EXPECT_EQ(routed->ports()[2].name, Language::Blocks::K_USAGE_PORT);
    EXPECT_EQ(routed->ports()[2].shape, (Shape{K_EXPERTS}));

    // The hidden width a row leaves out is four times the width, per expert.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    const Trace trace = Trace::evaluation(builder);
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(routed->forward(trace, NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(graph.value(outputs[2]).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 1u);
    const bool has_stacked_hidden = std::any_of(
        graph.nodes().begin(), graph.nodes().end(), [&graph](const Graph::Node& node)
    {
        return node.op == Ops::K_GELU.id
            && graph.value(node.outputs[0]).shape == Shape{10, K_EXPERTS * 4 * K_DIM};
    });
    EXPECT_TRUE(has_stacked_hidden);
}

/** @brief Tests that the row refuses a top_k without experts and one above their count. */
TEST(Unit_BlockExpertTransformerBlock, TheTransformerRowRefusesCountsThatDoNotFit)
{
    SA_EXPECT_FAILURE(make_row(R"({"heads": 2, "top_k": 1})"), StatusCode::InvalidDescription,
                      "layer 3", "'top_k'", "no 'experts'");
    SA_EXPECT_FAILURE(make_row(R"({"heads": 2, "experts": 2, "top_k": 3})"),
                      StatusCode::InvalidDescription, "layer 3", "top_k=3");
    SA_EXPECT_FAILURE(make_row(R"({"heads": 2, "experts": -1})"),
                      StatusCode::InvalidDescription, "layer 3", "'experts'", "zero or more");
}

/** @brief Tests that one position carries the two caches and routes through the experts. */
TEST(Unit_BlockExpertTransformerBlock, TheIncrementalFormTracesEachChildOverOnePosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    ExpertTransformerBlock block = make_block(1);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced,
                            block.trace_experts_incremental(trace, x, Shape{5, K_DIM}));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 1u);
    EXPECT_EQ(carried.values().size(), 2u);

    // The node form appends the same three values through forward_incremental.
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            make_row(R"({"heads": 2, "experts": 4})"));
    Graph::Graph node_graph;
    const Graph::GraphBuilder node_builder(node_graph);
    const Graph::ValueID slice = node_builder.input(Shape{1, 1, K_DIM}, "x");
    NN::FedLeafList node_fed;
    NN::CarriedValueList node_carried;
    const Trace node_trace = Trace::carrying(node_builder, node_fed, node_carried);
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward_incremental(node_trace, NN::ValueSpan(&slice, 1), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(node_graph.value(outputs[0]).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(node_carried.values().size(), 2u);
}

/** @brief Tests that the block and its row hand the positions to the attention. */
TEST(Unit_BlockExpertTransformerBlock, TheBlockAndItsRowPassThePositionsToTheAttention)
{
    using Language::Blocks::AttentionPositions;

    EXPECT_EQ(make_block().attn.positions, AttentionPositions::Learned);
    const ExpertTransformerBlock turned(K_HEADS, MoEFeedForward(K_DIM, K_HIDDEN, K_EXPERTS, 0),
                                        AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON,
                                        AttentionPositions::Rotary);
    EXPECT_EQ(turned.attn.positions, AttentionPositions::Rotary);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> routed,
        make_row(R"({"heads": 2, "experts": 4, "positions": "rotary"})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(routed->forward(trace, NN::ValueSpan(&x, 1), outputs));

    // The one attention of the block declares its cosines and its sines.
    EXPECT_EQ(tables.tables().size(), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 2u);

    SA_EXPECT_FAILURE(make_row(R"({"heads": 8, "experts": 4, "positions": "rotary"})"),
                      StatusCode::InvalidDescription, "layer 3", "rotary positions",
                      "1 channels");
}

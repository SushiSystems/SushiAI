/**************************************************************************/
/* test_transformer_block.cpp                                             */
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
 * @file test_transformer_block.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of TransformerBlock.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::TransformerBlock;
using NN::Trace;

namespace
{
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of every case. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the width of the feed-forward of every case. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds the nodes a causal two-head trace adds: 34 around the heads, 15 per head. */
    constexpr std::size_t K_TRACE_NODES = 64;

    /** @brief Returns a block over the three widths above with the mask given. */
    TransformerBlock make_block(AttentionMask mask = AttentionMask::Causal)
    {
        return TransformerBlock(K_DIM, K_HEADS, K_HIDDEN, mask, NN::K_LAYER_NORM_EPSILON);
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
}

/** @brief Tests that a trace is two normalised branches, each added to what it read. */
TEST(Small_BlockTransformerBlock, ATraceAddsANormalisedAttentionThenANormalisedFeedForward)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    TransformerBlock block = make_block();
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_TRACE_NODES);
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_GELU.id), 1u);

    // The last sum reads the first sum, and the first sum reads the block's input.
    const Graph::Node& last = graph.node(graph.value(y).producer);
    ASSERT_EQ(last.op, Ops::K_ADD.id);
    const Graph::Node& first = graph.node(graph.value(last.inputs[0]).producer);
    ASSERT_EQ(first.op, Ops::K_ADD.id);
    EXPECT_EQ(first.inputs[0], x);

    // Each normalisation reads what its branch is added to: the pre-norm form.
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op == Ops::K_LAYER_NORM.id)
            EXPECT_TRUE(node.inputs[0] == x || node.inputs[0] == first.outputs[0]);
    }
}

/** @brief Tests that the mask option reaches the attention: no CAUSAL_MASK without it. */
TEST(Small_BlockTransformerBlock, ABlockWithoutAMaskEmitsNone)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    TransformerBlock block = make_block(AttentionMask::None);

    const Trace trace = Trace::evaluation(builder);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, 5, K_DIM}, "x")));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u);
}

/** @brief Tests that the walk names the twelve parameters under the four children, in order. */
TEST(Small_BlockTransformerBlock, TheWalkYieldsTheFourChildren)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    TransformerBlock block = make_block();
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
                  "h.ff.expand.weight", "h.ff.expand.bias", "h.ff.project.weight",
                  "h.ff.project.bias"}));
}

/** @brief Tests that an input of another width fails once, with the first child's message. */
TEST(Small_BlockTransformerBlock, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    TransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(trace, wide), wide);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "LayerNorm(8)",
                      "[2, 5, 9]");
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 0u);
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that the factory reads its keys, and takes four times the width as hidden. */
TEST(Small_BlockTransformerBlock, TheFactoryReadsItsKeysOverTheirDefaults)
{
    const Shape input{5, K_DIM};
    const Config::Value plain = Config::parse_json(R"({"heads": 2})", "<test>");
    const Config::Value stated = Config::parse_json(
        R"({"heads": 4, "hidden": 16, "causal": false, "ln_epsilon": 0.001})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), plain,
                                                      "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> tuned,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), stated,
                                                      "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(node.get());
    const auto* wrapped_tuned =
        dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(tuned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_tuned, nullptr);
    EXPECT_EQ(wrapped->layer().attn.heads, 2);
    EXPECT_EQ(wrapped->layer().ff.hidden, 4 * K_DIM);
    EXPECT_EQ(wrapped->layer().attn.mask, AttentionMask::Causal);
    EXPECT_EQ(wrapped->layer().norm1.epsilon, NN::K_LAYER_NORM_EPSILON);
    EXPECT_EQ(wrapped_tuned->layer().attn.heads, 4);
    EXPECT_EQ(wrapped_tuned->layer().ff.hidden, 16);
    EXPECT_EQ(wrapped_tuned->layer().attn.mask, AttentionMask::None);
    EXPECT_EQ(wrapped_tuned->layer().norm2.epsilon, 0.001);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
}

/** @brief Tests that the factory refuses a stray key, no heads and heads that do not divide. */
TEST(Small_BlockTransformerBlock, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{5, K_DIM};
    const Config::Value stray = Config::parse_json(R"({"heads": 2, "depth": 3})", "<test>");
    const Config::Value absent = Config::parse_json(R"({"hidden": 16})", "<test>");
    const Config::Value uneven = Config::parse_json(R"({"heads": 5})", "<test>");

    SA_EXPECT_FAILURE(
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), stray,
                                                      "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'depth'");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), absent,
                                                      "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'heads'", "absent");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), uneven,
                                                      "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "dim=8 heads=5");
}

/** @brief Tests that one position of a block normalises twice and carries the two caches. */
TEST(Small_BlockTransformerBlock, TheIncrementalFormTracesEachChildOverOnePosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    TransformerBlock block(K_DIM, K_HEADS, K_HIDDEN, AttentionMask::Causal,
                           NN::K_LAYER_NORM_EPSILON);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{5, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_GELU.id), 1u);
    EXPECT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(carried.values().size(), 2u);
}

/** @brief Tests that the incremental form declares the parameters the ordinary form declares. */
TEST(Small_BlockTransformerBlock, BothFormsDeclareTheSameParameters)
{
    TransformerBlock whole(K_DIM, K_HEADS, K_HIDDEN, AttentionMask::Causal,
                           NN::K_LAYER_NORM_EPSILON);
    TransformerBlock sliced(K_DIM, K_HEADS, K_HIDDEN, AttentionMask::Causal,
                            NN::K_LAYER_NORM_EPSILON);

    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    static_cast<void>(whole.forward(Trace::evaluation(whole_builder),
                                    whole_builder.input(Shape{1, 5, K_DIM}, "x")));

    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    static_cast<void>(sliced.forward_incremental(
        trace, sliced_builder.input(Shape{1, 1, K_DIM}, "x"), Shape{5, K_DIM}));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(NN::parameter_count(sliced), NN::parameter_count(whole));
    EXPECT_EQ(NN::buffer_count(sliced), 0u);
}

/** @brief Tests that a block without a mask fails the trace through its attention. */
TEST(Small_BlockTransformerBlock, AnUnmaskedBlockHasNoIncrementalForm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    TransformerBlock block(K_DIM, K_HEADS, K_HIDDEN, AttentionMask::None,
                           NN::K_LAYER_NORM_EPSILON);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    EXPECT_EQ(block.forward_incremental(trace, x, Shape{5, K_DIM}), x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "no mask");
}

/** @brief Tests that the factory hands "positions" to the attention and refuses an odd head. */
TEST(Small_BlockTransformerBlock, TheFactoryPassesThePositionsToItsAttention)
{
    const Shape input{5, K_DIM};
    const Shape odd{5, 6};
    const Config::Value plain = Config::parse_json(R"({"heads": 2})", "<test>");
    const Config::Value rotary =
        Config::parse_json(R"({"heads": 2, "positions": "rotary"})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> learned,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), plain,
                                                      "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> turned,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), rotary,
                                                      "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(learned.get());
    const auto* wrapped_turned =
        dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(turned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_turned, nullptr);
    EXPECT_EQ(wrapped->layer().attn.positions,
              Language::Blocks::AttentionPositions::Learned);
    EXPECT_EQ(wrapped_turned->layer().attn.positions,
              Language::Blocks::AttentionPositions::Rotary);

    SA_EXPECT_FAILURE(
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&odd, 1), rotary,
                                                      "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "rotary positions", "3 channels");
}

/**************************************************************************/
/* test_moe_feed_forward.cpp                                              */
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
 * @file test_moe_feed_forward.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of MoEFeedForward.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::ExpertTrace;
using Language::Blocks::MoEFeedForward;
using NN::Trace;

namespace
{
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the width between the two linear layers of one expert. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Holds the number of experts of every case. */
    constexpr int64_t K_EXPERTS = 4;

    /** @brief Holds the nodes a trace that keeps every expert adds, six of them leaves. */
    constexpr std::size_t K_DENSE_NODES = 31;

    /** @brief Holds the nodes the gate of the kept scores adds before its first level. */
    constexpr std::size_t K_TOP_NODES = 7;

    /** @brief Holds the nodes each level of kept scores adds. */
    constexpr std::size_t K_LEVEL_NODES = 4;

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

    /** @brief Returns the names the state walk of @p block yields under @p prefix. */
    template<typename Walked>
    std::vector<std::string> state_names(Walked& block, const std::string& prefix)
    {
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
        NN::walk_state(block, prefix, visitor);

        return names;
    }
}

/** @brief Tests that a trace gates every expert, and yields its usage and its balance. */
TEST(Small_BlockMoeFeedForward, ATraceMixesEveryExpertByItsGate)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 0);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(traced.balance).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_DENSE_NODES);

    // One softmax gates the experts and feeds the balance; no window picks a score.
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_UPSAMPLE2D.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_GELU.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 5u);

    // The result is the sum of the gated projections and of the gated biases.
    const Graph::Node& restored = graph.node(graph.value(traced.output).producer);
    ASSERT_EQ(restored.op, Ops::K_RESHAPE.id);
    const Graph::Node& mixed = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(mixed.op, Ops::K_ADD.id);
    const Graph::Node& projected = graph.node(graph.value(mixed.inputs[0]).producer);
    ASSERT_EQ(projected.op, Ops::K_MATMUL.id);
    const Graph::Node& gated = graph.node(graph.value(projected.inputs[0]).producer);
    ASSERT_EQ(gated.op, Ops::K_MUL.id);
    EXPECT_EQ(graph.value(gated.outputs[0]).shape, (Shape{10, K_EXPERTS * K_HIDDEN}));
}

/** @brief Tests that each kept level adds one row window, one comparison and their glue. */
TEST(Small_BlockMoeFeedForward, KeepingTheTopScoresAddsOneWindowPerLevel)
{
    for (const int64_t top_k : {int64_t{1}, int64_t{2}, K_EXPERTS})
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
        MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, top_k);
        const std::size_t nodes_before = graph.node_count();
        const std::size_t levels = static_cast<std::size_t>(top_k);

        const Trace trace = Trace::evaluation(builder);
        SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

        SA_EXPECT_OK(trace.status());
        EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM})) << top_k;
        EXPECT_EQ(graph.node_count(),
                  nodes_before + K_DENSE_NODES + K_TOP_NODES + K_LEVEL_NODES * levels)
            << top_k;
        EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), levels) << top_k;
        EXPECT_EQ(count_nodes(graph, Ops::K_LESS_EQUAL.id), levels) << top_k;
        EXPECT_EQ(count_nodes(graph, Ops::K_SELECT.id), levels) << top_k;

        // The balance reads the softmax of every score, the gate that of the kept ones.
        EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u) << top_k;
        const Graph::Node& usage = graph.node(graph.value(traced.usage).producer);
        ASSERT_EQ(usage.op, Ops::K_RESHAPE.id);
    }
}

/** @brief Tests that the walk names the router, then the two stacked layers of the experts. */
TEST(Small_BlockMoeFeedForward, TheWalkYieldsTheRouterAndTheStackedLayers)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 0);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 5, K_DIM}, "x")));

    EXPECT_EQ(state_names(block, "ff"),
              (std::vector<std::string>{"ff.router.weight", "ff.router.bias",
                                        "ff.expand.weight", "ff.expand.bias",
                                        "ff.project.weight", "ff.project.bias"}));
    EXPECT_EQ(block.router.weight.shape(), (Shape{K_EXPERTS, K_DIM}));
    EXPECT_EQ(block.expand.weight.shape(), (Shape{K_EXPERTS * K_HIDDEN, K_DIM}));
    EXPECT_EQ(block.project.weight.shape(), (Shape{K_DIM, K_EXPERTS * K_HIDDEN}));
    EXPECT_EQ(block.project.bias.shape(), (Shape{K_EXPERTS, K_DIM}));
    EXPECT_EQ(block.project.bias.role(), NN::ParameterRole::Bias);
}

/**
 * @brief Tests that an input of another rank, a block of no width and a top_k above the
 *        experts each fail and add no node.
 */
TEST(Small_BlockMoeFeedForward, AnInputOrADescriptionThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 0);
    MoEFeedForward empty{};
    MoEFeedForward greedy(K_DIM, K_HIDDEN, K_EXPERTS, K_EXPERTS + 1);
    const std::size_t nodes_before = graph.node_count();

    const Trace first = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(first, flat), flat);
    SA_EXPECT_FAILURE(first.status(), StatusCode::InvalidDescription, "MoEFeedForward",
                      "rank-3 [N, T, 6] input", "[5, 6]");

    const Trace second = Trace::evaluation(builder);
    EXPECT_EQ(empty.forward(second, x), x);
    SA_EXPECT_FAILURE(second.status(), StatusCode::InvalidDescription, "MoEFeedForward",
                      "at least one expert", "dim=0 hidden=0 experts=0");

    const Trace third = Trace::evaluation(builder);
    SA_EXPECT_FAILURE(greedy.trace_experts(third, x), StatusCode::InvalidDescription,
                      "1 to 4 of them", "top_k=5");
    SA_EXPECT_OK(third.status());

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.router.weight.declared());
}

/** @brief Tests that the factory reads its keys and builds a node of three ports. */
TEST(Small_BlockMoeFeedForward, TheFactoryBuildsANodeOfThreePorts)
{
    const Shape input{5, K_DIM};
    const Config::Value good =
        Config::parse_json(R"({"experts": 4, "hidden": 8, "top_k": 2})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_moe_feed_forward_node(NN::ShapeSpan(&input, 1), good,
                                                     "layer 0"));

    EXPECT_EQ(node->input_count(), 1u);
    ASSERT_EQ(node->ports().size(), 3u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
    EXPECT_EQ(node->ports()[1].name, Language::Blocks::K_BALANCE_PORT);
    EXPECT_EQ(node->ports()[1].shape, (Shape{1}));
    EXPECT_EQ(node->ports()[2].name, Language::Blocks::K_USAGE_PORT);
    EXPECT_EQ(node->ports()[2].shape, (Shape{K_EXPERTS}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    const Trace trace = Trace::evaluation(builder);
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(trace, NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(outputs[1]).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(outputs[2]).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 2u);
    EXPECT_EQ(state_names(*node, "3").size(), 6u);

    // A refused input leaves the outputs as they were.
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    const Trace refused = Trace::evaluation(builder);
    SA_EXPECT_FAILURE(node->forward(refused, NN::ValueSpan(&flat, 1), outputs),
                      StatusCode::InvalidDescription, "MoEFeedForward", "[5, 6]");
    EXPECT_EQ(outputs.size(), 3u);
}

/** @brief Tests that the factory refuses a key it lacks, a stray key and a count out of range. */
TEST(Small_BlockMoeFeedForward, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{5, K_DIM};
    const auto make = [&input](const char* text)
    {
        return Language::Blocks::make_moe_feed_forward_node(
            NN::ShapeSpan(&input, 1), Config::parse_json(text, "<test>"), "layer 3");
    };

    SA_EXPECT_FAILURE(make(R"({"hidden": 8})"), StatusCode::InvalidDescription, "layer 3",
                      "'experts'", "absent");
    SA_EXPECT_FAILURE(make(R"({"experts": 4})"), StatusCode::InvalidDescription, "layer 3",
                      "'hidden'", "absent");
    SA_EXPECT_FAILURE(make(R"({"experts": 4, "hidden": 8, "noise": 0.1})"),
                      StatusCode::InvalidDescription, "layer 3", "'noise'");
    SA_EXPECT_FAILURE(make(R"({"experts": 4, "hidden": 8, "top_k": 5})"),
                      StatusCode::InvalidDescription, "layer 3", "top_k=5");
    SA_EXPECT_FAILURE(make(R"({"experts": 4, "hidden": 8, "top_k": -1})"),
                      StatusCode::InvalidDescription, "layer 3", "'top_k'", "zero or more");
    SA_EXPECT_FAILURE(make(R"({"experts": 4, "hidden": 8, "top_k": 0.5})"),
                      StatusCode::InvalidDescription, "layer 3", "'top_k'", "whole number");
}

/** @brief Tests that the incremental form of one position is the ordinary trace of it. */
TEST(Small_BlockMoeFeedForward, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    MoEFeedForward whole(K_DIM, K_HIDDEN, K_EXPERTS, 1);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    static_cast<void>(whole.forward(Trace::evaluation(whole_builder),
                                    whole_builder.input(Shape{1, 1, K_DIM}, "x")));

    MoEFeedForward sliced(K_DIM, K_HIDDEN, K_EXPERTS, 1);
    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    const Graph::ValueID y = sliced.forward_incremental(
        trace, sliced_builder.input(Shape{1, 1, K_DIM}, "x"), Shape{5, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(sliced_graph.value(y).shape, (Shape{1, 1, K_DIM}));
    ASSERT_EQ(sliced_graph.node_count(), whole_graph.node_count());
    for (std::size_t at = 0; at < whole_graph.node_count(); ++at)
        EXPECT_EQ(sliced_graph.nodes()[at].op, whole_graph.nodes()[at].op) << "node " << at;

    EXPECT_TRUE(fed.leaves().empty());
    EXPECT_TRUE(carried.values().empty());
}

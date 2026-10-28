/**************************************************************************/
/* test_feed_forward.cpp                                                  */
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
 * @file test_feed_forward.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of FeedForward.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::FeedForward;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the width between the two linear layers. */
    constexpr int64_t K_HIDDEN = 24;

    /** @brief Holds the nodes one trace adds: four leaves and seven others. */
    constexpr std::size_t K_TRACE_NODES = 11;
}

/** @brief Tests that a trace is reshape, linear, GELU, linear, reshape, and keeps the shape. */
TEST(Small_BlockFeedForward, ATraceMapsTheRowsThroughBothLayers)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    FeedForward block(K_DIM, K_HIDDEN);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_TRACE_NODES);

    const Graph::Node& restored = graph.node(graph.value(y).producer);
    ASSERT_EQ(restored.op, Ops::K_RESHAPE.id);
    const Graph::Node& biased = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(biased.op, Ops::K_BIAS_ADD.id);
    const Graph::Node& projected = graph.node(graph.value(biased.inputs[0]).producer);
    ASSERT_EQ(projected.op, Ops::K_MATMUL.id);
    const Graph::Node& active = graph.node(graph.value(projected.inputs[0]).producer);
    ASSERT_EQ(active.op, Ops::K_GELU.id);
    EXPECT_EQ(graph.value(active.outputs[0]).shape, (Shape{10, K_HIDDEN}));
}

/** @brief Tests that the walk names the two linear layers, expand before project. */
TEST(Small_BlockFeedForward, TheWalkYieldsTheTwoLinearLayers)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    FeedForward block(K_DIM, K_HIDDEN);
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
    NN::walk_state(block, "ff", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"ff.expand.weight", "ff.expand.bias",
                                               "ff.project.weight", "ff.project.bias"}));
    EXPECT_EQ(block.expand.weight.shape(), (Shape{K_HIDDEN, K_DIM}));
    EXPECT_EQ(block.project.weight.shape(), (Shape{K_DIM, K_HIDDEN}));
}

/** @brief Tests that an input of another rank or width, and a block of no width, each fail. */
TEST(Small_BlockFeedForward, AnInputOrADescriptionThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    FeedForward block(K_DIM, K_HIDDEN);
    FeedForward empty{};
    const std::size_t nodes_before = graph.node_count();

    const Trace first = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(first, flat), flat);
    SA_EXPECT_FAILURE(first.status(), StatusCode::InvalidDescription, "FeedForward",
                      "rank-3 [N, T, 6] input", "[5, 6]");

    const Trace second = Trace::evaluation(builder);
    EXPECT_EQ(empty.forward(second, x), x);
    SA_EXPECT_FAILURE(second.status(), StatusCode::InvalidDescription, "positive widths",
                      "dim=0 hidden=0");

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.expand.weight.declared());
}

/** @brief Tests that the factory reads "hidden" and refuses its absence and a stray key. */
TEST(Small_BlockFeedForward, TheFactoryReadsTheHiddenWidth)
{
    const Shape input{5, K_DIM};
    const Config::Value good = require_value(Config::parse_json(R"({"hidden": 24})", "<test>"));
    const Config::Value absent = require_value(Config::parse_json("{}", "<test>"));
    const Config::Value stray =
        require_value(Config::parse_json(R"({"hidden": 24, "bias": true})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_feed_forward_node(NN::ShapeSpan(&input, 1), good, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<FeedForward>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().hidden, K_HIDDEN);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_feed_forward_node(NN::ShapeSpan(&input, 1), absent, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'hidden'", "absent");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_feed_forward_node(NN::ShapeSpan(&input, 1), stray, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'bias'");
}

/** @brief Tests that the incremental form of one position is the ordinary trace of it. */
TEST(Small_BlockFeedForward, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    FeedForward whole(K_DIM, K_HIDDEN);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    static_cast<void>(whole.forward(Trace::evaluation(whole_builder),
                                    whole_builder.input(Shape{1, 1, K_DIM}, "x")));

    FeedForward sliced(K_DIM, K_HIDDEN);
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

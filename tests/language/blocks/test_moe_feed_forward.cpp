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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_moe_feed_forward.cpp
 * @brief Tests the trace, the kept scores, the state and the node factory of MoEFeedForward.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::ExpertTrace;
using Language::Blocks::MoEFeedForward;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the width between the two linear layers of one expert. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Holds the number of experts of every case. */
    constexpr int64_t K_EXPERTS = 4;

    /** @brief Holds the positions of the sequence the incremental case compares against. */
    constexpr int64_t K_POSITIONS = 3;

    /** @brief Holds the position the incremental case traces alone. */
    constexpr std::size_t K_LAST_POSITION = 2;

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

    /** @brief Returns the node the row {"experts": 4, "hidden": 8, "top_k": 2} builds. */
    std::unique_ptr<NN::Node> make_node()
    {
        const Shape input{5, K_DIM};
        return require_value(Language::Blocks::make_moe_feed_forward_node(
            NN::ShapeSpan(&input, 1),
            require_value(Config::parse_json(R"({"experts": 4, "hidden": 8, "top_k": 2})",
                                             "<test>")),
            "layer 0"));
    }

    /** @brief Runs one count of kept experts. */
    class Small_BlockMoeFeedForwardTopK : public ::testing::TestWithParam<int64_t>
    {

    };

    /** @brief Holds one factory row the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused factory row. */
    class Small_BlockMoeFeedForwardFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Names a factory row by its name field. */
    std::string row_name(const ::testing::TestParamInfo<FactoryRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace yields the output, the balance and the usage in their shapes. */
TEST(Small_BlockMoeFeedForward, ATraceMixesEveryExpertByItsGate)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 0);

    const Trace trace = Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(traced.balance).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
}

/** @brief Tests that a block keeping the top scores keeps the sequence shape, one row per count. */
TEST_P(Small_BlockMoeFeedForwardTopK, KeepingTheTopScoresKeepsTheSequenceShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, GetParam());

    const Trace trace = Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM}));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockMoeFeedForwardTopK,
                         ::testing::Values(int64_t{1}, int64_t{2}, K_EXPERTS));

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

/** @brief Tests that an input of another rank fails and leaves the graph as it was. */
TEST(Small_BlockMoeFeedForward, AnInputOfAnotherRankIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 0);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, flat);

    EXPECT_EQ(y, flat);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "MoEFeedForward",
                      "rank-3 [N, T, 6] input", "[5, 6]");
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.router.weight.declared());
}

/** @brief Tests that a block of no width fails the trace and names its three counts. */
TEST(Small_BlockMoeFeedForward, ABlockOfNoWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward empty{};

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = empty.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "MoEFeedForward",
                      "at least one expert", "dim=0 hidden=0 experts=0");
}

/** @brief Tests that trace_experts refuses a top_k above the experts and adds no node. */
TEST(Small_BlockMoeFeedForward, ATopKAboveTheExpertsIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    MoEFeedForward greedy(K_DIM, K_HIDDEN, K_EXPERTS, K_EXPERTS + 1);
    const std::size_t nodes_before = graph.node_count();
    const Trace trace = Trace::evaluation(builder);

    const Result<ExpertTrace> traced = greedy.trace_experts(trace, x);

    SA_EXPECT_FAILURE(traced, StatusCode::InvalidDescription, "1 to 4 of them", "top_k=5");
    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that the factory builds a node of the output, balance and usage ports. */
TEST(Small_BlockMoeFeedForward, TheFactoryBuildsANodeOfThreePorts)
{
    const std::unique_ptr<NN::Node> node = make_node();

    EXPECT_EQ(node->input_count(), 1u);
    ASSERT_EQ(node->ports().size(), 3u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
    EXPECT_EQ(node->ports()[1].name, Language::Blocks::K_BALANCE_PORT);
    EXPECT_EQ(node->ports()[1].shape, (Shape{1}));
    EXPECT_EQ(node->ports()[2].name, Language::Blocks::K_USAGE_PORT);
    EXPECT_EQ(node->ports()[2].shape, (Shape{K_EXPERTS}));
}

/** @brief Tests that the node writes the output, the balance and the usage in their shapes. */
TEST(Small_BlockMoeFeedForward, TheNodeWritesThreeOutputsOfTheirShapes)
{
    const std::unique_ptr<NN::Node> node = make_node();
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
}

/** @brief Tests that the node leaves its outputs as they were when it refuses an input. */
TEST(Small_BlockMoeFeedForward, ARefusedInputLeavesTheOutputsAsTheyWere)
{
    const std::unique_ptr<NN::Node> node = make_node();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));
    const std::vector<Graph::ValueID> before = outputs;
    const Graph::ValueID flat = builder.input(Shape{5, K_DIM}, "flat");
    const Trace refused = Trace::evaluation(builder);

    const Status outcome = node->forward(refused, NN::ValueSpan(&flat, 1), outputs);

    SA_EXPECT_FAILURE(outcome, StatusCode::InvalidDescription, "MoEFeedForward", "[5, 6]");
    EXPECT_EQ(outputs, before);
}

/** @brief Tests that the factory refuses a row whose keys or counts do not fit. */
TEST_P(Small_BlockMoeFeedForwardFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Shape input{5, K_DIM};
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(Language::Blocks::make_moe_feed_forward_node(NN::ShapeSpan(&input, 1),
                                                                   attributes, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockMoeFeedForwardFactory,
    ::testing::Values(
        FactoryRow{"NoExperts", R"({"hidden": 8})", "'experts'", "absent"},
        FactoryRow{"NoHidden", R"({"experts": 4})", "'hidden'", "absent"},
        FactoryRow{"StrayKey", R"({"experts": 4, "hidden": 8, "noise": 0.1})", "'noise'", ""},
        FactoryRow{"TopKAboveExperts", R"({"experts": 4, "hidden": 8, "top_k": 5})", "top_k=5",
                   ""},
        FactoryRow{"NegativeTopK", R"({"experts": 4, "hidden": 8, "top_k": -1})", "'top_k'",
                   "zero or more"},
        FactoryRow{"FractionalTopK", R"({"experts": 4, "hidden": 8, "top_k": 0.5})", "'top_k'",
                   "whole number"}),
    row_name);

/**
 * @brief Tests that the incremental output of one position is the ordinary output at it.
 *
 * Both graphs keep one expert and run on the host evaluator over the same state and rows, so
 * the row of the position is computed by the same operations in the same order and compares
 * exactly.
 */
TEST(Small_BlockMoeFeedForward, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    const std::vector<double> rows = SaTest::eighths(K_POSITIONS * K_DIM, 0);
    MoEFeedForward whole(K_DIM, K_HIDDEN, K_EXPERTS, 1);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    const Graph::ValueID sequence = whole_builder.input(Shape{1, K_POSITIONS, K_DIM}, "x");
    const Graph::ValueID whole_y = whole.forward(Trace::evaluation(whole_builder), sequence);
    Graph::HostEvaluator whole_run(whole_graph);
    SA_EXPECT_OK(whole_run.write(sequence, rows));
    SA_EXPECT_OK(SaTest::write_state(whole_run, whole_graph, whole));
    SA_EXPECT_OK(whole_run.run());
    MoEFeedForward sliced(K_DIM, K_HIDDEN, K_EXPERTS, 1);
    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    const Graph::ValueID position = sliced_builder.input(Shape{1, 1, K_DIM}, "x");
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    const Graph::ValueID sliced_y =
        sliced.forward_incremental(trace, position, Shape{K_POSITIONS, K_DIM});
    Graph::HostEvaluator sliced_run(sliced_graph);
    SA_EXPECT_OK(sliced_run.write(position, SaTest::row_of(rows, K_LAST_POSITION, K_DIM)));
    SA_EXPECT_OK(SaTest::write_state(sliced_run, sliced_graph, sliced));

    SA_EXPECT_OK(sliced_run.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> ordinary, whole_run.read(whole_y));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> incremental, sliced_run.read(sliced_y));
    EXPECT_EQ(incremental, SaTest::row_of(ordinary, K_LAST_POSITION, K_DIM));
}

/** @brief Tests that the incremental form declares no fed leaf and carries no value. */
TEST(Small_BlockMoeFeedForward, TheIncrementalFormFeedsAndCarriesNothing)
{
    MoEFeedForward block(K_DIM, K_HIDDEN, K_EXPERTS, 1);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    const Graph::ValueID y = block.forward_incremental(
        trace, builder.input(Shape{1, 1, K_DIM}, "x"), Shape{K_POSITIONS, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_TRUE(fed.leaves().empty());
    EXPECT_TRUE(carried.values().empty());
}

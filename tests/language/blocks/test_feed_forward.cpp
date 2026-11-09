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
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::FeedForward;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the width between the two linear layers. */
    constexpr int64_t K_HIDDEN = 24;

    /** @brief Holds the positions of the sequence the incremental case compares against. */
    constexpr int64_t K_POSITIONS = 3;

    /** @brief Holds the position the incremental case traces alone. */
    constexpr std::size_t K_LAST_POSITION = 2;

    /** @brief Holds one input the trace refuses and the shape its message names. */
    struct InputRow
    {
        std::string name;  /**< The row's name. */
        Shape shape;       /**< The input's shape. */
        std::string named; /**< The shape the message names. */
    };

    /** @brief Runs one refused input. */
    class Small_BlockFeedForwardInput : public ::testing::TestWithParam<InputRow>
    {

    };

    /** @brief Holds one factory row the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        std::string key;    /**< The key the message names. */
        std::string reason; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused factory row. */
    class Small_BlockFeedForwardFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace keeps the [N, T, dim] shape and records no failure. */
TEST(Small_BlockFeedForward, ATraceKeepsTheSequenceShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    FeedForward block(K_DIM, K_HIDDEN);

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    EXPECT_FALSE(graph.has_refusal());
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

/** @brief Tests that an input of another rank or width fails and leaves the graph as it was. */
TEST_P(Small_BlockFeedForwardInput, AnInputThatDoesNotFitIsRefused)
{
    const InputRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.shape, "x");
    FeedForward block(K_DIM, K_HIDDEN);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "FeedForward",
                      "rank-3 [N, T, 6] input", row.named);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.expand.weight.declared());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockFeedForwardInput,
                         ::testing::Values(InputRow{"RankTwo", Shape{5, K_DIM}, "[5, 6]"},
                                           InputRow{"WidthSeven", Shape{2, 5, 7}, "[2, 5, 7]"}),
                         row_name<InputRow>);

/** @brief Tests that a block of no width fails the trace and names both widths. */
TEST(Small_BlockFeedForward, ABlockOfNoWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    FeedForward empty{};
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = empty.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "positive widths",
                      "dim=0 hidden=0");
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that the factory reads "hidden" into a one-port node of the input's shape. */
TEST(Small_BlockFeedForward, TheFactoryReadsTheHiddenWidth)
{
    const Shape input{5, K_DIM};
    const Config::Value good = require_value(Config::parse_json(R"({"hidden": 24})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_feed_forward_node(NN::ShapeSpan(&input, 1), good, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<FeedForward>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().hidden, K_HIDDEN);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
}

/** @brief Tests that the factory refuses a row without "hidden" and a row with a stray key. */
TEST_P(Small_BlockFeedForwardFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Shape input{5, K_DIM};
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_feed_forward_node(NN::ShapeSpan(&input, 1), attributes, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.key, row.reason);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockFeedForwardFactory,
    ::testing::Values(FactoryRow{"NoHidden", "{}", "'hidden'", "absent"},
                      FactoryRow{"StrayKey", R"({"hidden": 24, "bias": true})", "'bias'", ""}),
    row_name<FactoryRow>);

/**
 * @brief Tests that the incremental output of one position is the ordinary output at it.
 *
 * Both graphs run on the host evaluator over the same state and the same rows, so the row of
 * the position is computed by the same operations in the same order and compares exactly.
 */
TEST(Small_BlockFeedForward, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    const std::vector<double> rows = SaTest::eighths(K_POSITIONS * K_DIM, 0);
    FeedForward whole(K_DIM, K_HIDDEN);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    const Graph::ValueID sequence = whole_builder.input(Shape{1, K_POSITIONS, K_DIM}, "x");
    const Graph::ValueID whole_y = whole.forward(Trace::evaluation(whole_builder), sequence);
    Graph::HostEvaluator whole_run(whole_graph);
    SA_EXPECT_OK(whole_run.write(sequence, rows));
    SA_EXPECT_OK(SaTest::write_state(whole_run, whole_graph, whole));
    SA_EXPECT_OK(whole_run.run());
    FeedForward sliced(K_DIM, K_HIDDEN);
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
TEST(Small_BlockFeedForward, TheIncrementalFormFeedsAndCarriesNothing)
{
    FeedForward block(K_DIM, K_HIDDEN);
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

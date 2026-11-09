/**************************************************************************/
/* test_concat.cpp                                                        */
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
 * @file test_concat.cpp
 * @brief Tests the Concat node's shape check, its port and the node it traces.
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
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::Concat;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Builds a Concat over @p shapes, one per input. */
    Result<std::unique_ptr<Concat>> create_over(const std::vector<Shape>& shapes)
    {
        return Concat::create(NN::ShapeSpan(shapes.data(), shapes.size()));
    }

    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueID>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }
} // namespace

/** @brief Tests that the port keeps H and W and carries the sum of the inputs' channels. */
TEST(Small_Concat, CreateSumsTheChannelsOfItsInputs)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 5}, Shape{8, 6, 1}}));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 9}));
}

/** @brief Tests that one input is accepted and its shape is the port's. */
TEST(Small_Concat, CreateAcceptsOneInput)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node, create_over({Shape{8, 6, 3}}));

    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Holds input shapes that Concat refuses and the substrings of the refusal. */
struct RefusedInputsRow
{
    std::string name;          /**< The row's name. */
    std::vector<Shape> shapes; /**< The input shapes given to create. */
    std::string first;         /**< A substring the message holds. */
    std::string second;        /**< A second substring the message holds. */
};

/** @brief Runs the refusal over each set of inputs create cannot join. */
class Small_ConcatRefused : public ::testing::TestWithParam<RefusedInputsRow>
{
};

/** @brief Tests that create refuses the inputs and names the offending one. */
TEST_P(Small_ConcatRefused, CreateRefusesTheInputs)
{
    const RefusedInputsRow& row = GetParam();

    SA_EXPECT_FAILURE(create_over(row.shapes), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ConcatRefused,
    ::testing::Values(
        RefusedInputsRow{"UnequalHeight", {Shape{8, 6, 3}, Shape{4, 6, 3}}, "concat input 1",
                         "H and W"},
        RefusedInputsRow{"UnequalWidth", {Shape{8, 6, 3}, Shape{8, 6, 3}, Shape{8, 12, 3}},
                         "concat input 2", "H and W"},
        RefusedInputsRow{"ASecondInputThatIsNotAFeatureMap", {Shape{8, 6, 3}, Shape{144}},
                         "concat input 1", "rank-3"},
        RefusedInputsRow{"AFirstInputThatIsNotAFeatureMap", {Shape{144}, Shape{8, 6, 3}},
                         "concat input 0", "rank-3"}),
    [](const ::testing::TestParamInfo<RefusedInputsRow>& info) { return info.param.name; });

/** @brief Tests that an empty input list is refused. */
TEST(Small_Concat, CreateRejectsNoInput)
{
    SA_EXPECT_FAILURE(create_over({}), StatusCode::InvalidDescription, "at least one input");
}

/** @brief Tests that forward emits one CONCAT_CHANNELS reading every input in order. */
TEST(Small_Concat, ForwardEmitsOneConcatChannelsReadingEveryInputInOrder)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 5}, "b");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 5}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({b, a}), outputs));

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& emitted = graph.nodes().back();
    EXPECT_EQ(emitted.op, Ops::K_CONCAT_CHANNELS.id);
    EXPECT_EQ(emitted.inputs, (std::vector<Graph::ValueID>{b, a}));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(emitted.outputs.size(), 1u);
    EXPECT_EQ(emitted.outputs.front(), outputs[0]);
}

/** @brief Tests that the traced value is the port's shape with the batch axis in front. */
TEST(Small_Concat, ForwardResultIsThePortShapeWithABatchAxis)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 5}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 5}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({a, b}), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 8, 6, 8}));
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 8}));
}

/** @brief Tests that forward refuses an input count other than the one it was built over. */
TEST(Small_Concat, ForwardRejectsAnInputCountItWasNotBuiltOver)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({a}), outputs),
                      StatusCode::InvalidDescription, "input count is 2", "given is 1");
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a node's input count is the number of shapes it was built over. */
TEST(Small_Concat, TheInputCountIsTheNumberOfShapesItWasBuiltOver)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> one, create_over({Shape{8, 6, 3}}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Concat> three,
        create_over({Shape{8, 6, 3}, Shape{8, 6, 5}, Shape{8, 6, 1}}));

    EXPECT_EQ(one->input_count(), 1u);
    EXPECT_EQ(three->input_count(), 3u);
}

/** @brief Tests that traced values the emitter refuses come back as a failure, not a throw. */
TEST(Small_Concat, ForwardReturnsTheFailureOfValuesThatDoNotJoin)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID shorter = builder.input(Shape{2, 4, 6, 3}, "shorter");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;

    const Status joined =
        node->forward(Trace::evaluation(builder), span_of({a, shorter}), outputs);

    EXPECT_FALSE(joined.is_ok());
    EXPECT_EQ(joined.code(), StatusCode::InvalidDescription);
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that the factory builds a Concat over its shapes from a row with no key. */
TEST(Small_Concat, TheFactoryBuildsANodeFromARowWithNoKey)
{
    const Shape shapes[] = {Shape{8, 6, 3}, Shape{8, 6, 5}};
    const Config::Value none = require_value(Config::parse_json("{}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_concat_node(NN::ShapeSpan(shapes, 2), none, "layer 2 'concat'"));

    EXPECT_NE(dynamic_cast<const Concat*>(node.get()), nullptr);
    EXPECT_EQ(node->input_count(), 2u);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 8}));
}

/** @brief Tests that the factory refuses a key and names the row. */
TEST(Small_Concat, TheFactoryRefusesAKey)
{
    const Shape joined[] = {Shape{8, 6, 3}, Shape{8, 6, 5}};
    const Config::Value stray = require_value(Config::parse_json(R"({"axis": 3})", "<test>"));

    SA_EXPECT_FAILURE(NN::make_concat_node(NN::ShapeSpan(joined, 2), stray, "layer 2 'concat'"),
                      StatusCode::InvalidDescription, "layer 2 'concat'", "'axis'", "no key");
}

/** @brief Tests that the factory refuses shapes that do not join and names the row. */
TEST(Small_Concat, TheFactoryRefusesShapesThatDoNotJoin)
{
    const Shape apart[] = {Shape{8, 6, 3}, Shape{4, 6, 5}};
    const Config::Value none = require_value(Config::parse_json("{}", "<test>"));

    SA_EXPECT_FAILURE(NN::make_concat_node(NN::ShapeSpan(apart, 2), none, "layer 2 'concat'"),
                      StatusCode::InvalidDescription, "layer 2 'concat'", "H and W");
}

/** @brief Tests that the state walk visits no parameter and no buffer. */
TEST(Small_Concat, TheStateWalkVisitsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node, create_over({Shape{8, 6, 3}}));

    std::size_t visits = 0;
    const auto on_parameter = [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    };
    const auto on_buffer = [&visits](const std::string&, NN::Buffer&)
    {
        ++visits;
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    node->walk_state("merge", visitor);

    EXPECT_EQ(visits, 0u);
}

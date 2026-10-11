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
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using NN::Concat;
using NN::Trace;

namespace
{
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
TEST(Unit_Concat, CreateSumsTheChannelsOfItsInputs)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 5}, Shape{8, 6, 1}}));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 9}));
}

/** @brief Tests that one input is accepted and its shape is the port's. */
TEST(Unit_Concat, CreateAcceptsOneInput)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node, create_over({Shape{8, 6, 3}}));

    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Tests that inputs of unequal height are refused and the offending input is named. */
TEST(Unit_Concat, CreateRejectsUnequalHeight)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{4, 6, 3}}),
                      StatusCode::InvalidDescription, "concat input 1", "H and W");
}

/** @brief Tests that inputs of unequal width are refused and the offending input is named. */
TEST(Unit_Concat, CreateRejectsUnequalWidth)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{8, 6, 3}, Shape{8, 12, 3}}),
                      StatusCode::InvalidDescription, "concat input 2", "H and W");
}

/** @brief Tests that an input that is not rank 3 is refused and named. */
TEST(Unit_Concat, CreateRejectsAnInputThatIsNotAFeatureMap)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{144}}),
                      StatusCode::InvalidDescription, "concat input 1", "rank-3");
}

/** @brief Tests that a first input that is not rank 3 is refused and named as input 0. */
TEST(Unit_Concat, CreateRejectsAFirstInputThatIsNotAFeatureMap)
{
    SA_EXPECT_FAILURE(create_over({Shape{144}, Shape{8, 6, 3}}),
                      StatusCode::InvalidDescription, "concat input 0", "rank-3");
}

/** @brief Tests that an empty input list is refused. */
TEST(Unit_Concat, CreateRejectsNoInput)
{
    SA_EXPECT_FAILURE(create_over({}), StatusCode::InvalidDescription, "at least one input");
}

/** @brief Tests that forward emits one CONCAT_CHANNELS reading every input in order. */
TEST(Unit_Concat, ForwardEmitsOneConcatChannelsReadingEveryInputInOrder)
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
TEST(Unit_Concat, ForwardResultIsThePortShapeWithABatchAxis)
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
TEST(Unit_Concat, ForwardRejectsAnInputCountItWasNotBuiltOver)
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
TEST(Unit_Concat, TheInputCountIsTheNumberOfShapesItWasBuiltOver)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> one, create_over({Shape{8, 6, 3}}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Concat> three,
        create_over({Shape{8, 6, 3}, Shape{8, 6, 5}, Shape{8, 6, 1}}));

    EXPECT_EQ(one->input_count(), 1u);
    EXPECT_EQ(three->input_count(), 3u);
}

/** @brief Tests that traced values the emitter refuses come back as a failure, not a throw. */
TEST(Unit_Concat, ForwardReturnsTheFailureOfValuesThatDoNotJoin)
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
TEST(Unit_Concat, TheFactoryBuildsANodeFromARowWithNoKey)
{
    const Shape shapes[] = {Shape{8, 6, 3}, Shape{8, 6, 5}};
    const Config::Value none = Config::parse_json("{}", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_concat_node(NN::ShapeSpan(shapes, 2), none, "layer 2 'concat'"));

    EXPECT_NE(dynamic_cast<const Concat*>(node.get()), nullptr);
    EXPECT_EQ(node->input_count(), 2u);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 8}));
}

/** @brief Tests that the factory refuses a key and names the row when the shapes do not join. */
TEST(Unit_Concat, TheFactoryRefusesAKeyAndShapesThatDoNotJoin)
{
    const Shape joined[] = {Shape{8, 6, 3}, Shape{8, 6, 5}};
    const Shape apart[] = {Shape{8, 6, 3}, Shape{4, 6, 5}};
    const Config::Value none = Config::parse_json("{}", "<test>");
    const Config::Value stray = Config::parse_json(R"({"axis": 3})", "<test>");

    SA_EXPECT_FAILURE(NN::make_concat_node(NN::ShapeSpan(joined, 2), stray, "layer 2 'concat'"),
                      StatusCode::InvalidDescription, "layer 2 'concat'", "'axis'", "no key");
    SA_EXPECT_FAILURE(NN::make_concat_node(NN::ShapeSpan(apart, 2), none, "layer 2 'concat'"),
                      StatusCode::InvalidDescription, "layer 2 'concat'", "H and W");
}

/** @brief Tests that the state walk visits nothing after a trace. */
TEST(Unit_Concat, TheStateWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Concat> node, create_over({Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({a}), outputs));

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

/**************************************************************************/
/* test_add.cpp                                                           */
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
 * @file test_add.cpp
 * @brief Tests the Add node's shape check, its port and the node it traces.
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
#include <SushiAI/nn/add.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::Add;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Builds an Add over @p shapes, one per input. */
    Result<std::unique_ptr<Add>> create_over(const std::vector<Shape>& shapes)
    {
        return Add::create(NN::ShapeSpan(shapes.data(), shapes.size()));
    }

    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueID>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }
} // namespace

/** @brief Tests that the port is named as the single output and carries the shared shape. */
TEST(Small_Add, CreateKeepsTheSharedShapeAsThePort)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Tests that one input is refused and the count is named. */
TEST(Small_Add, CreateRejectsOneInput)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}}), StatusCode::InvalidDescription,
                      "takes 2 inputs", "was given 1");
}

/** @brief Tests that three inputs are refused and the count is named. */
TEST(Small_Add, CreateRejectsThreeInputs)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{8, 6, 3}, Shape{8, 6, 3}}),
                      StatusCode::InvalidDescription, "takes 2 inputs", "was given 3");
}

/** @brief Tests that an empty input list is refused as a count of zero. */
TEST(Small_Add, CreateRejectsNoInput)
{
    SA_EXPECT_FAILURE(create_over({}), StatusCode::InvalidDescription, "takes 2 inputs",
                      "was given 0");
}

/** @brief Tests that inputs differing in channels are refused as unequal shapes. */
TEST(Small_Add, CreateRejectsUnequalChannels)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{8, 6, 5}}),
                      StatusCode::InvalidDescription, "share one shape");
}

/** @brief Tests that inputs differing in height are refused as unequal shapes. */
TEST(Small_Add, CreateRejectsUnequalHeight)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{4, 6, 3}}),
                      StatusCode::InvalidDescription, "share one shape");
}

/** @brief Tests that inputs differing in width are refused as unequal shapes. */
TEST(Small_Add, CreateRejectsUnequalWidth)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{8, 12, 3}}),
                      StatusCode::InvalidDescription, "share one shape");
}

/** @brief Tests that a second input that is not rank 3 is refused and named. */
TEST(Small_Add, CreateRejectsAnInputThatIsNotAFeatureMap)
{
    SA_EXPECT_FAILURE(create_over({Shape{8, 6, 3}, Shape{144}}), StatusCode::InvalidDescription,
                      "add input 1", "rank-3");
}

/** @brief Tests that a first input that is not rank 3 is refused and named as input 0. */
TEST(Small_Add, CreateRejectsAFirstInputThatIsNotAFeatureMap)
{
    SA_EXPECT_FAILURE(create_over({Shape{144}, Shape{144}}), StatusCode::InvalidDescription,
                      "add input 0", "rank-3");
}

/** @brief Tests that the node reports two inputs, whatever it was built over. */
TEST(Small_Add, TheInputCountIsTwo)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));

    EXPECT_EQ(node->input_count(), 2u);
}

/** @brief Tests that forward emits one K_ADD reading both inputs in the order given. */
TEST(Small_Add, ForwardEmitsOneAddReadingBothInputsInOrder)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 3}, "b");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({b, a}), outputs));

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& emitted = graph.nodes().back();
    EXPECT_EQ(emitted.op, Ops::K_ADD.id);
    EXPECT_EQ(emitted.inputs, (std::vector<Graph::ValueID>{b, a}));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(emitted.outputs.size(), 1u);
    EXPECT_EQ(emitted.outputs.front(), outputs[0]);
}

/** @brief Tests that the traced value is the port's shape with the batch axis in front. */
TEST(Small_Add, ForwardResultIsThePortShapeWithABatchAxis)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 3}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({a, b}), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 8, 6, 3}));
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Tests that forward refuses an input count other than two and emits nothing. */
TEST(Small_Add, ForwardRejectsAnInputCountItWasNotBuiltOver)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({a}), outputs),
                      StatusCode::InvalidDescription, "input count is 2", "given is 1");
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that values of unequal shape come back as a failure and emit nothing. */
TEST(Small_Add, ForwardReturnsTheFailureOfValuesThatDoNotAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID wider = builder.input(Shape{2, 8, 6, 5}, "wider");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;

    const Status summed = node->forward(Trace::evaluation(builder), span_of({a, wider}), outputs);

    EXPECT_FALSE(summed.is_ok());
    EXPECT_EQ(summed.code(), StatusCode::InvalidDescription);
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that the factory builds an Add over its shapes from a row with no key. */
TEST(Small_Add, TheFactoryBuildsANodeFromARowWithNoKey)
{
    const Shape shapes[] = {Shape{8, 6, 3}, Shape{8, 6, 3}};
    const Config::Value none = require_value(Config::parse_json("{}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_add_node(NN::ShapeSpan(shapes, 2), none, "layer 3 'add'"));

    EXPECT_NE(dynamic_cast<const Add*>(node.get()), nullptr);
    EXPECT_EQ(node->input_count(), 2u);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Tests that the factory refuses any key and shapes that do not match. */
TEST(Small_Add, TheFactoryRefusesAKeyAndShapesThatDoNotMatch)
{
    const Shape matched[] = {Shape{8, 6, 3}, Shape{8, 6, 3}};
    const Shape unmatched[] = {Shape{8, 6, 3}, Shape{8, 6, 5}};
    const Config::Value stray = require_value(Config::parse_json(R"({"alpha": 2})", "<test>"));
    const Config::Value none = require_value(Config::parse_json("{}", "<test>"));

    SA_EXPECT_FAILURE(NN::make_add_node(NN::ShapeSpan(matched, 2), stray, "layer 3 'add'"),
                      StatusCode::InvalidDescription, "layer 3 'add'", "'alpha'", "no key");
    SA_EXPECT_FAILURE(NN::make_add_node(NN::ShapeSpan(unmatched, 2), none, "layer 3 'add'"),
                      StatusCode::InvalidDescription, "layer 3 'add'", "share one shape");
}

/** @brief Tests that the state walk visits nothing after a trace. */
TEST(Small_Add, TheStateWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 3}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Add> node,
                            create_over({Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({a, b}), outputs));

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
    node->walk_state("sum", visitor);

    EXPECT_EQ(visits, 0u);
}

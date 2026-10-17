/**************************************************************************/
/* test_dropout.cpp                                                       */
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
 * @file test_dropout.cpp
 * @brief Tests the node Dropout emits in training, its pass-through elsewhere, and its factory.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::Dropout;
using NN::Trace;

static_assert(NN::ModalModule<Dropout> && !NN::Module<Dropout> && !NN::TraceModule<Dropout>,
              "Dropout has the modal form alone");

namespace
{
    /** @brief Holds the seed the training traces of these cases state. */
    constexpr std::uint64_t K_TRACE_SEED = 5;

    /** @brief Returns the node that wrote @p value. */
    const Graph::Node& producer_of(const Graph::Graph& graph, Graph::ValueID value)
    {
        return graph.node(graph.value(value).producer);
    }
}

/** @brief Tests that the training form emits one DROPOUT node with the rate and site seed 0. */
TEST(Small_Dropout, TheTrainingFormEmitsOneDropoutNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, 4}, "x");
    Dropout layer{};
    layer.rate = 0.25;
    const std::size_t nodes_before = graph.node_count();

    const Trace training = Trace::training(builder, NN::discard_updates(), K_TRACE_SEED);
    const Graph::ValueID y = NN::trace_layer(layer, training, x);

    SA_EXPECT_OK(training.status());
    ASSERT_NE(y, x);
    ASSERT_EQ(graph.node_count(), nodes_before + 1u);
    const Graph::Node& node = producer_of(graph, y);
    EXPECT_EQ(node.op, Ops::K_DROPOUT.id);
    EXPECT_EQ(node.inputs, (std::vector<Graph::ValueID>{x}));
    EXPECT_EQ(node.attributes.values[Graph::K_DROPOUT_RATE_SLOT], 0.25);
    EXPECT_EQ(Graph::dropout_seed(node.attributes),
              Graph::dropout_site_seed(K_TRACE_SEED, 0));
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 3, 4}));
    EXPECT_EQ(NN::parameter_count(layer), 0u);
    EXPECT_EQ(NN::buffer_count(layer), 0u);
}

/** @brief Tests that the evaluation form returns its input and adds no node and no value. */
TEST(Small_Dropout, TheEvaluationFormReturnsItsInputAndEmitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, 4}, "x");
    Dropout layer{};
    layer.rate = 0.25;
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    const Trace evaluation = Trace::evaluation(builder);

    EXPECT_EQ(NN::trace_layer(layer, evaluation, x), x);
    SA_EXPECT_OK(evaluation.status());
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}

/** @brief Tests that a rate of zero emits nothing in training and claims no site. */
TEST(Small_Dropout, ARateOfZeroEmitsNothingInTraining)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    Dropout layer{};
    const std::size_t nodes_before = graph.node_count();

    const Trace training = Trace::training(builder, NN::discard_updates(), K_TRACE_SEED);

    EXPECT_EQ(NN::trace_layer(layer, training, x), x);
    SA_EXPECT_OK(training.status());
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(training.claim_random_site(), 0u);
}

/** @brief Tests that two sites of a trace get seeds 0 and 1 of its seed, on every trace of it. */
TEST(Small_Dropout, EachSiteOfATraceGetsItsOwnSeed)
{
    Dropout layer{};
    layer.rate = 0.5;

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
        const Trace training = Trace::training(builder, NN::discard_updates(), K_TRACE_SEED);

        const Graph::ValueID first = NN::trace_layer(layer, training, x);
        const Graph::ValueID second = NN::trace_layer(layer, training, first);

        SA_EXPECT_OK(training.status());
        EXPECT_EQ(Graph::dropout_seed(producer_of(graph, first).attributes),
                  Graph::dropout_site_seed(K_TRACE_SEED, 0));
        EXPECT_EQ(Graph::dropout_seed(producer_of(graph, second).attributes),
                  Graph::dropout_site_seed(K_TRACE_SEED, 1));
    }
}

/** @brief Tests that a training trace given no seed has seed 0, and an evaluation trace too. */
TEST(Small_Dropout, ATraceWithoutAStatedSeedHasSeedZero)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    EXPECT_EQ(Trace::training(builder, NN::discard_updates()).seed(), 0u);
    EXPECT_EQ(Trace::training(builder, NN::discard_updates(), K_TRACE_SEED).seed(),
              K_TRACE_SEED);
    EXPECT_EQ(Trace::evaluation(builder).seed(), 0u);
}

/** @brief Tests that a rate of one, a negative rate and a NaN are each refused by both forms. */
TEST(Small_Dropout, ARateOutsideItsIntervalFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    for (const double rate : {1.0, -0.1, std::numeric_limits<double>::quiet_NaN()})
    {
        Dropout layer{};
        layer.rate = rate;

        const Trace training = Trace::training(builder, NN::discard_updates());
        const Trace evaluation = Trace::evaluation(builder);
        EXPECT_EQ(layer.trace_training(training, x), x);
        EXPECT_EQ(layer.trace_evaluation(evaluation, x), x);

        SA_EXPECT_FAILURE(training.status(), StatusCode::InvalidDescription, "rate in [0, 1)");
        SA_EXPECT_FAILURE(evaluation.status(), StatusCode::InvalidDescription, "rate in [0, 1)");
    }

    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that output_shape returns the per-sample shape it is given. */
TEST(Small_Dropout, OutputShapeIsTheInputShape)
{
    const Dropout layer{};

    EXPECT_EQ(layer.output_shape(Shape{6, 4}), (Shape{6, 4}));
}

/** @brief Tests that the factory reads "rate" and builds a node that keeps its input shape. */
TEST(Small_Dropout, TheFactoryReadsTheRate)
{
    const Shape input{6, 4};
    const Config::Value attributes = Config::parse_json(R"({"rate": 0.1})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_dropout_node(NN::ShapeSpan(&input, 1), attributes, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<Dropout>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().rate, 0.1);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{6, 4}));
}

/** @brief Tests that the factory refuses no rate, a rate of one, a stray key and two inputs. */
TEST(Small_Dropout, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{6, 4};
    const Shape pair[] = {Shape{4}, Shape{4}};
    const Config::Value good = Config::parse_json(R"({"rate": 0.1})", "<test>");
    const Config::Value absent = Config::parse_json("{}", "<test>");
    const Config::Value whole = Config::parse_json(R"({"rate": 1.0})", "<test>");
    const Config::Value stray = Config::parse_json(R"({"rate": 0.1, "seed": 3})", "<test>");

    SA_EXPECT_FAILURE(NN::make_dropout_node(NN::ShapeSpan(&input, 1), absent, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'rate'", "absent");
    SA_EXPECT_FAILURE(NN::make_dropout_node(NN::ShapeSpan(&input, 1), whole, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'rate'", "[0, 1)");
    SA_EXPECT_FAILURE(NN::make_dropout_node(NN::ShapeSpan(&input, 1), stray, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'seed'");
    SA_EXPECT_FAILURE(NN::make_dropout_node(NN::ShapeSpan(pair, 2), good, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "reads one input", "given 2");
}

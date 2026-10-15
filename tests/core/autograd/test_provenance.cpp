/**************************************************************************/
/* test_provenance.cpp                                                    */
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
 * @file test_provenance.cpp
 * @brief Tests the forward origin that differentiation records for every node it appends.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::NodeID;
using Graph::ValueID;

namespace
{
    /** @brief Holds a two-layer perceptron with a cross-entropy loss and the values tests name. */
    struct TwoLinearGraph
    {
        /** @brief Builds the forward graph. */
        TwoLinearGraph()
        {
            const GraphBuilder builder(graph);

            const ValueID x = builder.input(Shape{8, 6}, "x");
            const ValueID targets = builder.input(Shape{8, 3}, "targets");

            first_weight = builder.parameter(Shape{6, 4}, "first_weight");
            first_bias = builder.parameter(Shape{4}, "first_bias");
            first_product = builder.matmul(x, first_weight);
            const ValueID hidden = builder.relu(builder.bias_add(first_product, first_bias));

            second_weight = builder.parameter(Shape{4, 3}, "second_weight");
            second_bias = builder.parameter(Shape{3}, "second_bias");
            const ValueID logits =
                builder.bias_add(builder.matmul(hidden, second_weight), second_bias);

            loss = builder.nll_loss(builder.log_softmax(logits), targets);
        }

        /** @brief Returns the four parameters, the first layer's weight first. */
        std::vector<ValueID> parameters() const
        {
            return {first_weight, first_bias, second_weight, second_bias};
        }

        /** @brief Holds the graph under test. */
        Graph::Graph graph;

        /** @brief Holds the first layer's weight. */
        ValueID first_weight = K_NO_VALUE;

        /** @brief Holds the first layer's bias. */
        ValueID first_bias = K_NO_VALUE;

        /** @brief Holds the result of the first layer's MATMUL. */
        ValueID first_product = K_NO_VALUE;

        /** @brief Holds the second layer's weight. */
        ValueID second_weight = K_NO_VALUE;

        /** @brief Holds the second layer's bias. */
        ValueID second_bias = K_NO_VALUE;

        /** @brief Holds the scalar loss. */
        ValueID loss = K_NO_VALUE;
    };

    /** @brief Differentiates @p loss with respect to @p wrt, recording into @p provenance. */
    std::vector<ValueID> differentiate(Graph::Graph& graph, ValueID loss,
                                       const std::vector<ValueID>& wrt,
                                       Autograd::Provenance* provenance)
    {
        return Autograd::differentiate(graph, loss,
                                       SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()),
                                       K_NO_VALUE, provenance);
    }

    /** @brief Returns the ids of the nodes of @p op at or after node @p from. */
    std::vector<NodeID> nodes_of(const Graph::Graph& graph, OpID op, std::size_t from)
    {
        std::vector<NodeID> found;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            if (graph.nodes()[i].op == op)
                found.push_back(static_cast<NodeID>(i));
        return found;
    }
}

/** @brief Tests that every appended node has exactly one origin and that it is a forward node. */
TEST(Small_Provenance, EveryBackwardNodeHasAForwardOrigin)
{
    TwoLinearGraph model;
    const std::size_t forward_nodes = model.graph.node_count();

    Autograd::Provenance provenance;
    (void)differentiate(model.graph, model.loss, model.parameters(), &provenance);

    ASSERT_GT(model.graph.node_count(), forward_nodes);
    EXPECT_EQ(provenance.forward_node_count, forward_nodes);
    ASSERT_EQ(provenance.origin.size(), model.graph.node_count() - forward_nodes);

    for (std::size_t i = 0; i < provenance.origin.size(); ++i)
        EXPECT_LT(provenance.origin[i], forward_nodes) << "appended node " << i;
}

/** @brief Tests that the first layer's weight gradient is recorded under the first MATMUL. */
TEST(Small_Provenance, ARuleEmitsUnderTheNodeItDifferentiates)
{
    TwoLinearGraph model;

    Autograd::Provenance provenance;
    const std::vector<ValueID> gradients =
        differentiate(model.graph, model.loss, model.parameters(), &provenance);

    ASSERT_NE(gradients.at(0), K_NO_VALUE);
    const NodeID emitted = model.graph.value(gradients[0]).producer;
    ASSERT_GE(emitted, provenance.forward_node_count);

    const NodeID origin = provenance.origin_of(emitted);
    EXPECT_EQ(model.graph.node(origin).op, Ops::K_MATMUL.id);
    EXPECT_EQ(origin, model.graph.value(model.first_product).producer);
}

/** @brief Tests that the ACCUMULATE summing a twice-read value is recorded under its producer. */
TEST(Small_Provenance, AFanOutAccumulateBelongsToTheProducer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    const ValueID shared = builder.relu(x);
    const ValueID sum = builder.add(builder.tanh(shared), builder.sigmoid(shared));
    const ValueID loss = builder.nll_loss(sum, targets);

    Autograd::Provenance provenance;
    (void)differentiate(graph, loss, {x}, &provenance);

    const std::vector<NodeID> accumulates =
        nodes_of(graph, Ops::K_ACCUMULATE.id, provenance.forward_node_count);
    ASSERT_EQ(accumulates.size(), 1u);

    const NodeID origin = provenance.origin_of(accumulates.front());
    EXPECT_EQ(graph.node(origin).op, Ops::K_RELU.id);
    EXPECT_EQ(origin, graph.value(shared).producer);
}

/** @brief Tests that the ACCUMULATE summing a twice-read parameter is recorded under its LEAF. */
TEST(Small_Provenance, AParameterReadTwiceAccumulatesUnderItsLeaf)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // targets comes first so that the LEAF of x is not node 0, where the reverse walk ends.
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID x = builder.parameter(Shape{4, 3}, "x");

    const ValueID sum = builder.add(builder.relu(x), builder.tanh(x));
    const ValueID loss = builder.nll_loss(sum, targets);

    Autograd::Provenance provenance;
    const std::vector<ValueID> gradients = differentiate(graph, loss, {x}, &provenance);

    ASSERT_NE(gradients.at(0), K_NO_VALUE);
    const NodeID closing = graph.value(gradients[0]).producer;
    ASSERT_EQ(graph.node(closing).op, Ops::K_ACCUMULATE.id);

    const NodeID origin = provenance.origin_of(closing);
    EXPECT_NE(origin, 0u);
    EXPECT_EQ(graph.node(origin).op, Ops::K_LEAF.id);
    EXPECT_EQ(origin, graph.value(x).producer);
}

/** @brief Tests that a Provenance used for a second run holds that run's origins alone. */
TEST(Small_Provenance, AReusedProvenanceHoldsOnlyTheLatestRun)
{
    Graph::Graph earlier;
    const GraphBuilder builder(earlier);
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID loss =
        builder.nll_loss(builder.add(builder.relu(x), builder.tanh(x)), targets);

    Autograd::Provenance reused;
    (void)differentiate(earlier, loss, {x}, &reused);
    ASSERT_FALSE(reused.origin.empty());

    TwoLinearGraph second;
    TwoLinearGraph reference;
    Autograd::Provenance fresh;
    (void)differentiate(second.graph, second.loss, second.parameters(), &reused);
    (void)differentiate(reference.graph, reference.loss, reference.parameters(), &fresh);

    EXPECT_EQ(reused.forward_node_count, fresh.forward_node_count);
    EXPECT_EQ(reused.origin, fresh.origin);
}

/** @brief Tests that a requested value no node produces is refused when its sum needs an origin. */
TEST(Small_Provenance, ARequestedValueWithoutAProducerIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID x =
        graph.add_value(Shape{4, 3}, DType::Float32, Graph::Lifetime::Parameter, "x");
    graph.value(x).requires_grad = true;

    const ValueID sum = builder.add(builder.relu(x), builder.tanh(x));
    const ValueID loss = builder.nll_loss(sum, targets);

    Autograd::Provenance provenance;
    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}, &provenance),
                    "provenance", "producer node");
}

/** @brief Tests that a null provenance yields the same nodes and gradients as a recorded run. */
TEST(Small_Provenance, PassingNoProvenanceChangesNothing)
{
    TwoLinearGraph recorded;
    TwoLinearGraph plain;

    Autograd::Provenance provenance;
    const std::vector<ValueID> recorded_gradients =
        differentiate(recorded.graph, recorded.loss, recorded.parameters(), &provenance);
    const std::vector<ValueID> plain_gradients =
        differentiate(plain.graph, plain.loss, plain.parameters(), nullptr);

    EXPECT_EQ(recorded_gradients, plain_gradients);
    ASSERT_EQ(recorded.graph.node_count(), plain.graph.node_count());

    for (std::size_t i = 0; i < plain.graph.node_count(); ++i)
    {
        const Graph::Node& with = recorded.graph.nodes()[i];
        const Graph::Node& without = plain.graph.nodes()[i];
        EXPECT_EQ(with.op, without.op) << "node " << i;
        EXPECT_EQ(with.inputs, without.inputs) << "node " << i;
        EXPECT_EQ(with.outputs, without.outputs) << "node " << i;
    }
}

/**************************************************************************/
/* test_layers.cpp                                                        */
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
 * @file test_layers.cpp
 * @brief Tests the built-in layers: their backward topology, the loss root they feed and the shape
 *        errors they name.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;
using Graph::NodeID;

namespace
{
    /** @brief The op names of nodes [from, end), for a readable expectation. */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }

    /** @brief How many nodes of @p op the whole graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

} // namespace

TEST(Small_Layers, TwoLayerMLPProducesTheExpectedBackwardTopology)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    const ValueID logits = model.forward(builder, x);
    const ValueID loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::size_t forward_nodes = graph.node_count();

    std::vector<ValueID> wrt;
    NN::for_each_parameter(model, [&wrt, &graph](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id_in(graph));
    });
    ASSERT_EQ(wrt.size(), 4u);

    const std::vector<ValueID> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    // Cross-entropy differentiates as one fused rule; see tests/layers/nn/README.md.
    EXPECT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.exp",
        "sushiai.sub",
        "sushiai.scale",
        "sushiai.sum_rows",         // d(bias) of the output layer
        "sushiai.matmul",           // d(hidden activation)
        "sushiai.matmul",           // d(weight) of the output layer
        "sushiai.relu_backward",
        "sushiai.sum_rows",         // d(bias) of the hidden layer
        "sushiai.matmul",           // d(weight) of the hidden layer
    }));

    // No LOG_SOFTMAX gradient: the fused rule absorbed that node.
    EXPECT_EQ(count_op(graph, Ops::K_LOG_SOFTMAX.id), 1u);

    // One contributor per value emits no ACCUMULATE; see tests/layers/nn/README.md.
    EXPECT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 0u);

    ASSERT_EQ(grads.size(), 4u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], K_NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
        EXPECT_EQ(graph.value(grads[i]).lifetime, Graph::Lifetime::Gradient);
    }
}

TEST(Small_Layers, TheInputIsNotDifferentiatedWhenItDoesNotRequireOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Linear layer{6, 3};
    const ValueID logits = layer.forward(builder, x);
    const ValueID loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::vector<ValueID> wrt = {layer.weight.id_in(graph)};
    (void)Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    // The input gradient's MATMUL is skipped; see tests/layers/nn/README.md.
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 2u);
}

namespace
{

    /** @brief Differentiates @p loss with respect to @p wrt. */
    std::vector<ValueID> differentiate(Graph::Graph& graph, ValueID loss,
                                       const std::vector<ValueID>& wrt,
                                       ValueID seed_factor = K_NO_VALUE,
                                       Autograd::Provenance* provenance = nullptr)
    {
        return Autograd::differentiate(graph, loss,
                                       SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()),
                                       seed_factor, provenance);
    }

} // namespace

/** @brief Tests that a fused cross-entropy root emits the nodes it emitted before, with no seed. */
TEST(Small_Layers, AFusedCrossEntropyRootEmitsNoSeedNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    const ValueID logits = model.forward(builder, x);
    const ValueID loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);
    const std::size_t forward_nodes = graph.node_count();

    std::vector<ValueID> wrt;
    NN::for_each_parameter(model, [&wrt, &graph](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id_in(graph));
    });

    (void)differentiate(graph, loss, wrt);

    EXPECT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.exp",
        "sushiai.sub",
        "sushiai.scale",
        "sushiai.sum_rows",
        "sushiai.matmul",
        "sushiai.matmul",
        "sushiai.relu_backward",
        "sushiai.sum_rows",
        "sushiai.matmul",
    }));
    EXPECT_EQ(count_op(graph, Ops::K_ZERO.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_CONSTANT.id), 0u);
}

TEST(Small_Layers, LinearNamesItsOwnDimensionsWhenTheInputDoesNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 5}, "x");

    NN::Linear layer{6, 4};
    SA_EXPECT_ERROR((void)layer.forward(builder, x),
                    "Linear(6, 4)", "[8, 5]", "inner extent must be 6");

    NN::Linear rank_checker{5, 4};
    const ValueID vector = builder.input(Shape{5}, "v");
    SA_EXPECT_ERROR((void)rank_checker.forward(builder, vector),
                    "Linear needs a rank-2 input", "[5]");

    NN::Linear empty{0, 4};
    SA_EXPECT_ERROR((void)empty.forward(builder, x),
                    "Linear needs positive feature counts", "in=0", "out=4");
}

/**************************************************************************/
/* test_loss_roots.cpp                                                    */
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
 * @file test_loss_roots.cpp
 * @brief Tests the explicit seed of a loss root and the gradient barriers a root can sit behind.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::NodeID;
using Graph::ValueID;

namespace
{
    /** @brief Bounds the error allowed on an attribute that a rule copies from the forward node. */
    constexpr double K_ATTRIBUTE_TOLERANCE = 1e-12;

    /** @brief Holds the factor of the SCALE term in WeightedSum. */
    constexpr double K_TERM_WEIGHT = 2.0;

    /** @brief Holds the factor of the SCALE that ends a loss in the tests that need one. */
    constexpr double K_ROOT_WEIGHT = 0.5;

    /** @brief Returns the operation names of nodes [from, end). */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }

    /** @brief Counts the nodes of @p op in the whole graph. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

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

    /** @brief Holds the loss K_TERM_WEIGHT * a + b * c over three one-element parameters. */
    struct WeightedSum
    {
        /** @brief Builds the forward graph. */
        WeightedSum()
        {
            const GraphBuilder builder(graph);

            a = builder.parameter(Shape{1}, "a");
            b = builder.parameter(Shape{1}, "b");
            c = builder.parameter(Shape{1}, "c");
            scaled = builder.scale(a, K_TERM_WEIGHT);
            product = builder.mul(b, c);
            sum = builder.add(scaled, product);
        }

        /** @brief Returns the three parameters in declaration order. */
        std::vector<ValueID> parameters() const
        {
            return {a, b, c};
        }

        Graph::Graph graph;          /**< @brief Holds the graph under test. */
        ValueID a = K_NO_VALUE;        /**< @brief Holds the operand of the SCALE term. */
        ValueID b = K_NO_VALUE;        /**< @brief Holds the first operand of the MUL term. */
        ValueID c = K_NO_VALUE;        /**< @brief Holds the second operand of the MUL term. */
        ValueID scaled = K_NO_VALUE;   /**< @brief Holds the SCALE term. */
        ValueID product = K_NO_VALUE;  /**< @brief Holds the MUL term. */
        ValueID sum = K_NO_VALUE;      /**< @brief Holds the ADD of the two terms. */
    };
}

/** @brief Tests that a loss ending in an ADD of two terms gives every parameter a gradient. */
TEST(Unit_LossRoots, ASumOfTwoTermsEndingInAddReachesEveryParameter)
{
    WeightedSum model;
    const std::size_t forward_nodes = model.graph.node_count();

    const std::vector<ValueID> gradients =
        differentiate(model.graph, model.sum, model.parameters());

    ASSERT_EQ(op_names(model.graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.mul",
        "sushiai.mul",
        "sushiai.scale",
    }));

    const ValueID seed = model.graph.nodes()[forward_nodes].outputs.at(0);
    const Graph::Node& db = model.graph.nodes()[forward_nodes + 1];
    const Graph::Node& dc = model.graph.nodes()[forward_nodes + 2];
    const Graph::Node& da = model.graph.nodes()[forward_nodes + 3];

    EXPECT_EQ(db.inputs, (std::vector<ValueID>{seed, model.c}));
    EXPECT_EQ(dc.inputs, (std::vector<ValueID>{seed, model.b}));
    EXPECT_EQ(da.inputs, (std::vector<ValueID>{seed}));
    EXPECT_NEAR(da.attributes.get(0), K_TERM_WEIGHT, K_ATTRIBUTE_TOLERANCE);

    ASSERT_EQ(gradients.size(), 3u);
    EXPECT_EQ(gradients[0], da.outputs.at(0));
    EXPECT_EQ(gradients[1], db.outputs.at(0));
    EXPECT_EQ(gradients[2], dc.outputs.at(0));

    for (ValueID gradient : gradients)
        EXPECT_EQ(model.graph.value(gradient).lifetime, Graph::Lifetime::Gradient);
}

/** @brief Tests that a loss ending in a SCALE hands the scaled seed to both terms under it. */
TEST(Unit_LossRoots, ASumOfTwoTermsEndingInScaleReachesEveryParameter)
{
    WeightedSum model;
    const GraphBuilder builder(model.graph);
    const ValueID loss = builder.scale(model.sum, K_ROOT_WEIGHT);
    const std::size_t forward_nodes = model.graph.node_count();

    const std::vector<ValueID> gradients = differentiate(model.graph, loss, model.parameters());

    ASSERT_EQ(op_names(model.graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.scale",
        "sushiai.mul",
        "sushiai.mul",
        "sushiai.scale",
    }));

    const ValueID seed = model.graph.nodes()[forward_nodes].outputs.at(0);
    const Graph::Node& through_root = model.graph.nodes()[forward_nodes + 1];
    const ValueID sum_gradient = through_root.outputs.at(0);

    EXPECT_EQ(through_root.inputs, (std::vector<ValueID>{seed}));
    EXPECT_NEAR(through_root.attributes.get(0), K_ROOT_WEIGHT, K_ATTRIBUTE_TOLERANCE);
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 2].inputs,
              (std::vector<ValueID>{sum_gradient, model.c}));
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 3].inputs,
              (std::vector<ValueID>{sum_gradient, model.b}));
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 4].inputs,
              (std::vector<ValueID>{sum_gradient}));

    ASSERT_EQ(gradients.size(), 3u);
    for (std::size_t i = 0; i < gradients.size(); ++i)
        EXPECT_NE(gradients[i], K_NO_VALUE) << "parameter " << i << " received no gradient";
}

/** @brief Tests that the seed is one CONSTANT of 1 in the root's shape, element type and layout. */
TEST(Unit_LossRoots, TheSeedIsAConstantOneShapedAndTypedLikeTheRoot)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1, 1, 1, 1}, "x", DType::Float64);
    const ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
    const ValueID loss = builder.scale(nchw, K_ROOT_WEIGHT);
    const std::size_t forward_nodes = graph.node_count();

    (void)differentiate(graph, loss, {x});

    ASSERT_GE(graph.node_count(), forward_nodes + 1);
    const Graph::Node& one = graph.nodes()[forward_nodes];

    ASSERT_EQ(one.op, Ops::K_CONSTANT.id);
    EXPECT_TRUE(one.inputs.empty());
    EXPECT_NEAR(one.attributes.get(0), 1.0, K_ATTRIBUTE_TOLERANCE);
    EXPECT_EQ(count_op(graph, Ops::K_CONSTANT.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_ZERO.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_EXP.id), 0u);

    const Graph::Value& seed = graph.value(one.outputs.at(0));
    EXPECT_EQ(seed.shape, (Shape{1, 1, 1, 1}));
    EXPECT_EQ(seed.dtype, DType::Float64);
    EXPECT_EQ(seed.layout, Graph::TensorLayout::NCHW);
}

/** @brief Tests that a seed factor multiplies the seed in one DYNAMIC_SCALE every rule reads. */
TEST(Unit_LossRoots, TheSeedFactorMultipliesTheSeedOnce)
{
    WeightedSum model;
    const GraphBuilder builder(model.graph);
    const ValueID factor = builder.input(Shape{1}, "factor");
    const std::size_t forward_nodes = model.graph.node_count();

    (void)differentiate(model.graph, model.sum, model.parameters(), factor);

    ASSERT_EQ(op_names(model.graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.dynamic_scale",
        "sushiai.mul",
        "sushiai.mul",
        "sushiai.scale",
    }));

    const ValueID one = model.graph.nodes()[forward_nodes].outputs.at(0);
    const Graph::Node& scaling = model.graph.nodes()[forward_nodes + 1];
    const ValueID seed = scaling.outputs.at(0);

    EXPECT_EQ(scaling.inputs, (std::vector<ValueID>{one, factor}));
    EXPECT_EQ(count_op(model.graph, Ops::K_DYNAMIC_SCALE.id), 1u);
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 2].inputs,
              (std::vector<ValueID>{seed, model.c}));
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 3].inputs,
              (std::vector<ValueID>{seed, model.b}));
    EXPECT_EQ(model.graph.nodes()[forward_nodes + 4].inputs, (std::vector<ValueID>{seed}));
}

/** @brief Tests that a root of more than one element throws and names its shape. */
TEST(Unit_LossRoots, ANonScalarRootThrowsNamingItsShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.add(x, x);

    SA_EXPECT_ERROR((void)differentiate(graph, y, {x}),
                    "needs a scalar to differentiate", "[4, 3]");
}

/** @brief Tests that a root that is a leaf throws, names the leaf and appends nothing. */
TEST(Unit_LossRoots, ALeafRootThrowsNamingTheLeaf)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID leaf = builder.parameter(Shape{1}, "leaf");
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, leaf, {leaf}), "it is a leaf", "sushiai.leaf");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a root no parameter reaches throws, names the cause and appends nothing. */
TEST(Unit_LossRoots, ARootThatNeedsNoGradientThrowsNamingTheCause)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID p = builder.input(Shape{1}, "p");
    const ValueID q = builder.input(Shape{1}, "q");
    const ValueID orphan = builder.parameter(Shape{1}, "orphan");
    const ValueID loss = builder.add(p, q);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {orphan}),
                    "requires no gradient", "sushiai.add");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a root whose producer has no rule throws, names it and appends nothing. */
TEST(Unit_LossRoots, ARootWhoseProducerHasNoRuleThrowsNamingTheOperation)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID factor = builder.input(Shape{1}, "factor");
    const ValueID loss = builder.dynamic_scale(x, factor);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "no backward rule", "sushiai.dynamic_scale");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a root that is not its producer's first result throws and names it. */
TEST(Unit_LossRoots, ARootThatIsNotTheFirstResultThrowsNamingTheProducer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1, 1, 1, 1}, "x");
    const Graph::PoolResult pooled = builder.maxpool2d(x, Graph::ConvGeometry{});
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, pooled.indices, {x}),
                    "not the first result", "sushiai.maxpool2d");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a root of an integer element type throws and names the type. */
TEST(Unit_LossRoots, ARootThatIsNotRealFloatingThrowsNamingItsType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x", DType::Int32);
    const ValueID loss = builder.scale(x, K_ROOT_WEIGHT);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "element type is int32", "sushiai.scale");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a fused cross-entropy root emits the nodes it emitted before, with no seed. */
TEST(Unit_LossRoots, AFusedCrossEntropyRootEmitsNoSeedNode)
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

/** @brief Tests that an NLL_LOSS root outside the fused pair emits its SCALE and no seed node. */
TEST(Unit_LossRoots, AnUnfusedNLLLossRootEmitsNoSeedNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID loss = builder.nll_loss(builder.relu(x), targets);
    const std::size_t forward_nodes = graph.node_count();

    (void)differentiate(graph, loss, {x});

    ASSERT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.relu_backward",
    }));
    EXPECT_EQ(graph.nodes()[forward_nodes].inputs, (std::vector<ValueID>{targets}));
}

/** @brief Tests that an NLL_LOSS root with a seed factor applies it in one DYNAMIC_SCALE. */
TEST(Unit_LossRoots, AnNLLLossRootAppliesTheSeedFactorInItsOwnRule)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID factor = builder.input(Shape{1}, "factor");
    const ValueID loss = builder.nll_loss(builder.log_softmax(x), targets);
    const std::size_t forward_nodes = graph.node_count();

    const std::vector<ValueID> gradients = differentiate(graph, loss, {x}, factor);

    ASSERT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.exp",
        "sushiai.sub",
        "sushiai.scale",
        "sushiai.dynamic_scale",
    }));
    EXPECT_EQ(gradients.at(0), graph.nodes()[forward_nodes + 3].outputs.at(0));
}

/** @brief Tests that the emitter writes one STOP_GRADIENT node that copies its operand's form. */
TEST(Unit_LossRoots, StopGradientEmitsAnIdentityShapedLikeItsOperand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{2, 3, 4, 5}, "x", DType::Float64);
    const ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
    const ValueID stopped = builder.stop_gradient(nchw);

    const Graph::Node& node = graph.node(graph.value(stopped).producer);
    EXPECT_EQ(node.op, Ops::K_STOP_GRADIENT.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{nchw}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{stopped}));

    EXPECT_EQ(graph.value(stopped).shape, graph.value(nchw).shape);
    EXPECT_EQ(graph.value(stopped).dtype, DType::Float64);
    EXPECT_EQ(graph.value(stopped).layout, Graph::TensorLayout::NCHW);
    EXPECT_NO_THROW(graph.finalise());
}

/** @brief Tests that STOP_GRADIENT is declared a gradient barrier and not differentiable. */
TEST(Unit_LossRoots, StopGradientIsDeclaredAGradientBarrierAndNotDifferentiable)
{
    EXPECT_TRUE(Ops::K_STOP_GRADIENT.blocks_gradient);
    EXPECT_FALSE(Ops::K_STOP_GRADIENT.differentiable);
    EXPECT_TRUE(Ops::is_gradient_barrier(Ops::K_STOP_GRADIENT.id));
    EXPECT_FALSE(Ops::is_gradient_barrier(Ops::K_RESHAPE.id));
    EXPECT_EQ(Ops::name_of(Ops::K_STOP_GRADIENT.id), "sushiai.stop_gradient");
}

/** @brief Tests that finalise() marks neither a STOP_GRADIENT result nor its consumers. */
TEST(Unit_LossRoots, RequiresGradDoesNotCrossStopGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID stopped = builder.stop_gradient(x);
    const ValueID downstream = builder.scale(stopped, K_TERM_WEIGHT);
    const ValueID direct = builder.scale(x, K_TERM_WEIGHT);

    graph.finalise();

    EXPECT_TRUE(graph.value(x).requires_grad);
    EXPECT_FALSE(graph.value(stopped).requires_grad);
    EXPECT_FALSE(graph.value(downstream).requires_grad);
    EXPECT_TRUE(graph.value(direct).requires_grad);
}

/** @brief Tests that a parameter read only through STOP_GRADIENT gets no gradient and no throw. */
TEST(Unit_LossRoots, StopGradientBlocksAParameterReachableOnlyThroughIt)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID blocked = builder.parameter(Shape{1}, "blocked");
    const ValueID open = builder.parameter(Shape{1}, "open");
    const ValueID stopped = builder.stop_gradient(builder.scale(blocked, K_TERM_WEIGHT));
    const ValueID loss = builder.mul(stopped, open);
    const std::size_t forward_nodes = graph.node_count();

    std::vector<ValueID> gradients;
    ASSERT_NO_THROW(gradients = differentiate(graph, loss, {blocked, open}));

    ASSERT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.mul",
    }));

    const ValueID seed = graph.nodes()[forward_nodes].outputs.at(0);
    const Graph::Node& d_open = graph.nodes()[forward_nodes + 1];

    EXPECT_EQ(gradients.at(0), K_NO_VALUE);
    EXPECT_EQ(gradients.at(1), d_open.outputs.at(0));
    EXPECT_EQ(d_open.inputs, (std::vector<ValueID>{seed, stopped}));
}

/** @brief Tests that a value read directly and through STOP_GRADIENT gets one gradient only. */
TEST(Unit_LossRoots, AValueReadDirectlyAndThroughStopGradientKeepsTheDirectGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID stopped = builder.stop_gradient(x);
    const ValueID loss = builder.mul(stopped, x);
    const std::size_t forward_nodes = graph.node_count();

    const std::vector<ValueID> gradients = differentiate(graph, loss, {x});

    ASSERT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.mul",
    }));

    const ValueID seed = graph.nodes()[forward_nodes].outputs.at(0);
    const Graph::Node& dx = graph.nodes()[forward_nodes + 1];

    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{seed, stopped}));
    EXPECT_EQ(gradients.at(0), dx.outputs.at(0));
    EXPECT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 0u);
}

/** @brief Tests that a loss every path to which crosses STOP_GRADIENT throws and names it. */
TEST(Unit_LossRoots, ALossReachableOnlyThroughStopGradientThrowsNamingTheBarrier)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID loss = builder.scale(builder.stop_gradient(x), K_ROOT_WEIGHT);

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "requires no gradient", "sushiai.scale");
}

/** @brief Tests that a root that is a STOP_GRADIENT result throws for needing no gradient. */
TEST(Unit_LossRoots, AStopGradientRootThrowsForRequiringNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID loss = builder.stop_gradient(x);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "requires no gradient", "sushiai.stop_gradient");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a gradient contributed to a STOP_GRADIENT result is dropped, not thrown on. */
TEST(Unit_LossRoots, AContributionToAStopGradientResultIsDroppedWithoutAskingForARule)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID blocked = builder.parameter(Shape{1}, "blocked");
    const ValueID open = builder.parameter(Shape{1}, "open");
    const ValueID stopped = builder.stop_gradient(builder.scale(blocked, K_TERM_WEIGHT));
    const ValueID loss = builder.mul(stopped, open);
    const std::size_t forward_nodes = graph.node_count();

    // finalise() only ever sets the flag, so a flag set by hand survives it.
    graph.value(stopped).requires_grad = true;

    Autograd::Provenance provenance;
    std::vector<ValueID> gradients;
    ASSERT_NO_THROW(gradients = differentiate(graph, loss, {blocked, open}, K_NO_VALUE,
                                              &provenance));

    ASSERT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.constant",
        "sushiai.mul",
        "sushiai.mul",
    }));
    EXPECT_EQ(gradients.at(0), K_NO_VALUE);
    EXPECT_NE(gradients.at(1), K_NO_VALUE);

    const NodeID barrier = graph.value(stopped).producer;
    EXPECT_EQ(std::count(provenance.origin.begin(), provenance.origin.end(), barrier), 0);
}

/** @brief Tests that the seed nodes CONSTANT and DYNAMIC_SCALE originate at the root producer. */
TEST(Unit_LossRoots, TheSeedNodesHaveTheRootProducerAsOrigin)
{
    WeightedSum model;
    const GraphBuilder builder(model.graph);
    const ValueID factor = builder.input(Shape{1}, "factor");
    const std::size_t forward_nodes = model.graph.node_count();

    Autograd::Provenance provenance;
    (void)differentiate(model.graph, model.sum, model.parameters(), factor, &provenance);

    ASSERT_EQ(provenance.forward_node_count, forward_nodes);
    ASSERT_EQ(provenance.origin.size(), model.graph.node_count() - forward_nodes);
    ASSERT_EQ(provenance.origin.size(), 5u);

    const NodeID root_producer = model.graph.value(model.sum).producer;
    const NodeID product_producer = model.graph.value(model.product).producer;
    const NodeID scaled_producer = model.graph.value(model.scaled).producer;
    const NodeID first = static_cast<NodeID>(forward_nodes);

    EXPECT_EQ(model.graph.node(root_producer).op, Ops::K_ADD.id);
    EXPECT_EQ(provenance.origin_of(first), root_producer);
    EXPECT_EQ(provenance.origin_of(first + 1), root_producer);
    EXPECT_EQ(provenance.origin_of(first + 2), product_producer);
    EXPECT_EQ(provenance.origin_of(first + 3), product_producer);
    EXPECT_EQ(provenance.origin_of(first + 4), scaled_producer);
}

/** @brief Tests that a root a CONSTANT produces throws for requiring no gradient. */
TEST(Unit_LossRoots, AConstantRootThrowsForRequiringNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID loss = builder.constant_like(x, K_ROOT_WEIGHT);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "requires no gradient", "sushiai.constant");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that a root a comparison of a parameter produces throws for needing no gradient. */
TEST(Unit_LossRoots, AComparisonRootThrowsForRequiringNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID bound = builder.input(Shape{1}, "bound");
    const ValueID loss = builder.less_equal(x, bound);
    const std::size_t forward_nodes = graph.node_count();

    SA_EXPECT_ERROR((void)differentiate(graph, loss, {x}),
                    "requires no gradient", "sushiai.less_equal");
    EXPECT_EQ(graph.node_count(), forward_nodes);
}

/** @brief Tests that finalise() marks no barrier's result and marks what bypasses the barrier. */
TEST(Unit_LossRoots, RequiresGradCrossesNoGradientBarrier)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID bound = builder.input(Shape{1}, "bound");
    const ValueID mask = builder.less_equal(x, bound);
    const ValueID one = builder.constant_like(x, 1.0);
    const ValueID zero = builder.zeros_like(x, Shape{1});
    const ValueID masked = builder.mul(mask, one);
    const ValueID mixed = builder.mul(mask, x);

    graph.finalise();

    EXPECT_FALSE(graph.value(mask).requires_grad);
    EXPECT_FALSE(graph.value(one).requires_grad);
    EXPECT_FALSE(graph.value(zero).requires_grad);
    EXPECT_FALSE(graph.value(masked).requires_grad);
    EXPECT_TRUE(graph.value(mixed).requires_grad);
}

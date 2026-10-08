/**************************************************************************/
/* test_fusion.cpp                                                        */
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
 * @file test_fusion.cpp
 * @brief Tests the matches and the refusals of the static fusion pass on synthetic graphs.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::DeclinedFusion;
using Graph::FusedForm;
using Graph::FusionDecline;
using Graph::FusionOptions;
using Graph::FusionReport;
using Graph::GraphBuilder;
using Graph::K_NO_NODE;
using Graph::NodeId;
using Graph::ValueId;

namespace
{
    constexpr int64_t K_ROWS = 4;
    constexpr int64_t K_IN_FEATURES = 3;
    constexpr int64_t K_OUT_FEATURES = 5;

    /**
     * @struct Layer
     * @brief One traced affine layer, with the ids a test wants to name.
     */
    struct Layer
    {
        ValueId x = Graph::K_NO_VALUE;
        ValueId weight = Graph::K_NO_VALUE;
        ValueId bias = Graph::K_NO_VALUE;
        ValueId product = Graph::K_NO_VALUE;   /**< The MATMUL's result. */
        ValueId biased = Graph::K_NO_VALUE;    /**< The BIAS_ADD's result. */
    };

    /** @brief Emits x -> matmul -> bias_add, the shape every case below builds on. */
    Layer trace_linear(const GraphBuilder& builder)
    {
        Layer layer;
        layer.x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x");
        layer.weight = builder.parameter(Shape{K_OUT_FEATURES, K_IN_FEATURES}, "w");
        layer.bias = builder.parameter(Shape{K_OUT_FEATURES}, "b");
        layer.product = builder.matmul(layer.x, layer.weight, false, true);
        layer.biased = builder.bias_add(layer.product, layer.bias);
        return layer;
    }

    /** @brief How many nodes of @p op the graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        std::size_t total = 0;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == op) ++total;
        return total;
    }

    /** @brief Whether the report holds a decline with @p reason. */
    bool declined_for(const FusionReport& report, FusionDecline reason)
    {
        const auto declines = report.declined();
        return std::any_of(declines.begin(), declines.end(),
                           [reason](const DeclinedFusion& d) { return d.reason == reason; });
    }

    class Unit_Fusion : public ::testing::Test {};
}

TEST_F(Unit_Fusion, EveryFormRoundTripsThroughItsOperationIdentity)
{
    // Every form round-trips through its operation identity; see tests/functional/unit/README.md.
    constexpr FusedForm K_ALL[] = {
        FusedForm::GemmBias, FusedForm::GemmBiasRelu,
        FusedForm::GemmBiasSigmoid, FusedForm::GemmBiasTanh,
        FusedForm::GemmBiasGelu,
        FusedForm::ExpSubScale,
    };

    for (FusedForm form : K_ALL)
    {
        const OpID op = Graph::fused_form_op(form);
        EXPECT_NE(Ops::name_of(op), "sushiai.unknown")
            << Graph::fused_form_name(form) << " has no declared identity";
        EXPECT_EQ(Graph::fused_form_of(op), form)
            << Graph::fused_form_name(form) << " does not round-trip";
    }

    EXPECT_EQ(Graph::fused_form_of(Ops::K_MATMUL.id), FusedForm::None);
    EXPECT_EQ(Graph::fused_form_of(Ops::K_RELU.id), FusedForm::None);
}

TEST_F(Unit_Fusion, AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts)
{
    // RELU's gradient reads its output, so it survives fusion; see tests/functional/unit/README.md.
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_RELU.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_SIGMOID.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_TANH.id));

    // GELU's gradient reads the pre-activation; see tests/functional/unit/README.md.
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::K_GELU.id));

    // Only the entry that spills folds GELU; see tests/functional/unit/README.md.
    EXPECT_TRUE(Graph::catalog_folds_activation(Ops::K_GELU.id));
    EXPECT_TRUE(Graph::fused_form_spills(FusedForm::GemmBiasGelu));

    for (FusedForm form : {FusedForm::GemmBias, FusedForm::GemmBiasRelu,
                           FusedForm::GemmBiasSigmoid, FusedForm::GemmBiasTanh,
                           FusedForm::ExpSubScale})
        EXPECT_FALSE(Graph::fused_form_spills(form))
            << Graph::fused_form_name(form) << " grew a second output without a reason";

    // Not an activation at all, so neither question applies to it.
    EXPECT_FALSE(Graph::catalog_folds_activation(Ops::K_MATMUL.id));
    EXPECT_FALSE(Graph::catalog_folds_activation(Ops::K_EXP.id));

    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GemmBias), Ops::K_LEAF.id);
    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GemmBiasRelu), Ops::K_RELU.id);
    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GemmBiasGelu), Ops::K_GELU.id);
}

TEST_F(Unit_Fusion, AMatmulAndABiasAddBecomeOneNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const std::vector<ValueId> observed{layer.biased};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    ASSERT_EQ(report.count(FusedForm::GemmBias), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_GEMM_BIAS.id), 1u);

    // The fused node writes the value the caller named; see tests/functional/unit/README.md.
    ASSERT_EQ(report.fused().size(), 1u);
    EXPECT_EQ(report.fused()[0].result, layer.biased);
    ASSERT_NE(graph.value(layer.biased).producer, K_NO_NODE);

    const Graph::Node& fused = graph.node(graph.value(layer.biased).producer);
    ASSERT_EQ(fused.inputs.size(), 3u);
    EXPECT_EQ(fused.inputs[0], layer.x);
    EXPECT_EQ(fused.inputs[1], layer.weight);
    EXPECT_EQ(fused.inputs[2], layer.bias);

    // The transposition flags survive the fusion; see tests/functional/unit/README.md.
    EXPECT_FALSE(fused.attributes.flag(0));
    EXPECT_TRUE(fused.attributes.flag(1));

    // The intermediate is nobody's output now.
    EXPECT_EQ(graph.value(layer.product).producer, K_NO_NODE);
}

TEST_F(Unit_Fusion, TheLongestMatchWinsOverTheShorterOneInsideIt)
{
    // The matcher walks roots in reverse; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.relu(layer.biased);

    const std::vector<ValueId> observed{activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasRelu), 1u);
    EXPECT_EQ(report.count(FusedForm::GemmBias), 0u);
    EXPECT_EQ(graph.node_count(), report.nodes_before() - 2u);
}

TEST_F(Unit_Fusion, TheCrossEntropyGradientsElementwiseTailBecomesOnePass)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId log_probs = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "log_probs");
    const ValueId targets = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "targets");
    const ValueId probabilities = builder.exp(log_probs);
    const ValueId difference = builder.sub(probabilities, targets);
    const ValueId scaled = builder.scale(difference, 0.25);

    const std::vector<ValueId> observed{scaled};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    ASSERT_EQ(report.count(FusedForm::ExpSubScale), 1u);

    const Graph::Node& fused = graph.node(graph.value(scaled).producer);
    ASSERT_EQ(fused.inputs.size(), 2u);
    EXPECT_EQ(fused.inputs[0], log_probs);
    EXPECT_EQ(fused.inputs[1], targets);
    EXPECT_DOUBLE_EQ(fused.attributes.get(0), 0.25);
    EXPECT_EQ(graph.value(probabilities).producer, K_NO_NODE);
    EXPECT_EQ(graph.value(difference).producer, K_NO_NODE);
}

TEST_F(Unit_Fusion, ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch)
{
    // The entry's link position refuses swapped operands; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "a");
    const ValueId b = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "b");
    const ValueId exponent = builder.exp(a);
    const ValueId difference = builder.sub(b, exponent);
    const ValueId scaled = builder.scale(difference, 0.5);

    const std::vector<ValueId> observed{scaled};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::ExpSubScale), 0u);
    EXPECT_EQ(report.nodes_after(), report.nodes_before());
}

TEST_F(Unit_Fusion, AnIntermediateWithASecondConsumerIsNotFusedAway)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    // The MATMUL result has a second reader; see tests/functional/unit/README.md.
    const ValueId also = builder.scale(layer.product, 2.0);

    const std::vector<ValueId> observed{layer.biased, also};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_TRUE(report.fused().empty());
    EXPECT_TRUE(declined_for(report, FusionDecline::ValueHasOtherConsumers));
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 1u);
}

TEST_F(Unit_Fusion, AnIntermediateTheCallerReadsByIdIsNotFusedAway)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.relu(layer.biased);

    // The training loop reads `biased` after every replay; see tests/functional/unit/README.md.
    const std::vector<ValueId> observed{layer.biased, activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasRelu), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::ValueIsObserved));

    // The shorter form rooted at the bias_add still fuses; see tests/functional/unit/README.md.
    EXPECT_EQ(report.count(FusedForm::GemmBias), 1u);
}

TEST_F(Unit_Fusion, AParameterIsProtectedWithoutBeingListed)
{
    // A persistent value is protected without being listed; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "a");
    const ValueId b = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "b");
    const ValueId exponent = builder.exp(a);

    // The exp's result is forced into a persistent class; see tests/functional/unit/README.md.
    graph.value(exponent).lifetime = Graph::Lifetime::OptimizerState;

    const ValueId difference = builder.sub(exponent, b);
    (void)builder.scale(difference, 0.5);

    const FusionReport report = Graph::fuse(graph, FusionOptions{});

    EXPECT_EQ(report.count(FusedForm::ExpSubScale), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::ValueIsObserved));
}

TEST_F(Unit_Fusion, AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills)
{
    // The fused node keeps writing the pre-activation; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const ValueId activated = builder.gelu(layer.biased);
    const ValueId upstream = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "dy");
    const ValueId gradient =
        builder.activation_backward(Ops::K_GELU_BACKWARD.id, upstream, layer.biased);

    const std::vector<ValueId> observed{activated, gradient};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    ASSERT_EQ(report.count(FusedForm::GemmBiasGelu), 1u) << report.to_string();
    EXPECT_EQ(count_op(graph, Ops::K_GELU.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_BIAS_ADD.id), 0u);

    // The pre-activation is the fused node's second output; see tests/functional/unit/README.md.
    ASSERT_EQ(report.fused().size(), 1u);
    EXPECT_EQ(report.fused()[0].result, activated);
    EXPECT_EQ(report.fused()[0].spilled, layer.biased);

    const NodeId producer = graph.value(layer.biased).producer;
    ASSERT_NE(producer, K_NO_NODE) << "the pre-activation has no producer - the spill was lost";
    EXPECT_EQ(producer, graph.value(activated).producer)
        << "the spill and the result must come from the same node";

    const Graph::Node& fused = graph.node(producer);
    ASSERT_EQ(fused.outputs.size(), 2u);
    EXPECT_EQ(fused.outputs[0], activated);
    EXPECT_EQ(fused.outputs[1], layer.biased);

    // Only the MATMUL's result is reported as eliminated; see tests/functional/unit/README.md.
    EXPECT_EQ(report.fused()[0].eliminated, std::vector<ValueId>{layer.product});
    EXPECT_EQ(graph.value(layer.product).producer, K_NO_NODE);

    // And gelu_backward still reads the pre-activation, not the result.
    const Graph::Node& backward = graph.node(graph.value(gradient).producer);
    ASSERT_EQ(backward.inputs.size(), 2u);
    EXPECT_EQ(backward.inputs[1], layer.biased);
}

TEST_F(Unit_Fusion, ASpilledPreActivationReadBeforeTheFusedNodeIsRefused)
{
    // A reader before the fused node would break the order; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    // Emitted *before* the GELU, unlike the backward node in the case above.
    const ValueId early = builder.scale(layer.biased, 2.0);
    const ValueId activated = builder.gelu(layer.biased);

    const std::vector<ValueId> observed{early, activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasGelu), 0u) << report.to_string();
    EXPECT_TRUE(declined_for(report, FusionDecline::SpillReadTooEarly));

    // The shorter form still fuses after the refusal; see tests/functional/unit/README.md.
    EXPECT_EQ(report.count(FusedForm::GemmBias), 1u);
    EXPECT_EQ(count_op(graph, Ops::K_GELU.id), 1u);

    // The rewritten graph stays topological; see tests/functional/unit/README.md.
    for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
        for (ValueId in : graph.node(id).inputs)
        {
            const NodeId producer = graph.value(in).producer;
            if (producer == K_NO_NODE) continue;
            EXPECT_LT(producer, id) << "node " << id << " reads value " << in;
        }
}

TEST_F(Unit_Fusion, ASpilledPreActivationMayBeReadFromOutsideTheGraph)
{
    // A spilled pre-activation may be named as observed; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.gelu(layer.biased);

    const std::vector<ValueId> observed{layer.biased, activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasGelu), 1u) << report.to_string();
    EXPECT_FALSE(declined_for(report, FusionDecline::ValueIsObserved));
    EXPECT_NE(graph.value(layer.biased).producer, K_NO_NODE);
}

TEST_F(Unit_Fusion, TheGeluFoldStillObeysEveryOtherRefusal)
{
    // A spill licenses one interior value and no other; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const ValueId also = builder.scale(layer.product, 2.0);
    const ValueId activated = builder.gelu(layer.biased);

    const std::vector<ValueId> observed{also, activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_TRUE(report.fused().empty()) << report.to_string();
    EXPECT_TRUE(declined_for(report, FusionDecline::ValueHasOtherConsumers));
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 1u);
}

TEST_F(Unit_Fusion, DisablingTheGeluFormFallsBackToTheShorterOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.gelu(layer.biased);

    const std::vector<ValueId> observed{activated};
    const std::vector<FusedForm> off{FusedForm::GemmBiasGelu};

    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());
    options.disabled = SushiRuntime::span<const FusedForm>(off.data(), off.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasGelu), 0u);
    EXPECT_EQ(report.count(FusedForm::GemmBias), 1u);
    EXPECT_TRUE(declined_for(report, FusionDecline::Disabled));

    // The GELU left outside is not reported as unfoldable; see tests/functional/unit/README.md.
    EXPECT_FALSE(declined_for(report, FusionDecline::GradientNeedsPreActivation));
    EXPECT_FALSE(declined_for(report, FusionDecline::NoCatalogEntry));
}

TEST_F(Unit_Fusion, ARecoverableActivationNextToAFusedGemmIsFoldedNotReported)
{
    // ReLU's gradient reads its own result, so it folds; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const ValueId activated = builder.relu(layer.biased);
    const ValueId upstream = builder.input(Shape{K_ROWS, K_OUT_FEATURES}, "dy");
    const ValueId gradient =
        builder.activation_backward(Ops::K_RELU_BACKWARD.id, upstream, activated);

    const std::vector<ValueId> observed{gradient};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasRelu), 1u);
    EXPECT_FALSE(declined_for(report, FusionDecline::GradientNeedsPreActivation));
}

TEST_F(Unit_Fusion, DisablingThePassLeavesTheGraphExactlyAsItWas)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    (void)builder.relu(layer.biased);

    const std::size_t nodes = graph.node_count();
    const std::size_t values = graph.value_count();

    FusionOptions options;
    options.enabled = false;

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_FALSE(report.enabled());
    EXPECT_TRUE(report.fused().empty());
    EXPECT_EQ(graph.node_count(), nodes);
    EXPECT_EQ(graph.value_count(), values);
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 1u);
}

TEST_F(Unit_Fusion, DisablingOneFormFallsBackToTheShorterOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.relu(layer.biased);

    const std::vector<ValueId> observed{activated};
    const std::vector<FusedForm> off{FusedForm::GemmBiasRelu};

    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());
    options.disabled = SushiRuntime::span<const FusedForm>(off.data(), off.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GemmBiasRelu), 0u);
    EXPECT_EQ(report.count(FusedForm::GemmBias), 1u);
    EXPECT_TRUE(declined_for(report, FusionDecline::Disabled));
}

TEST_F(Unit_Fusion, ValueIdsAndTheirDescriptionsSurviveTheRewrite)
{
    // Value ids and descriptions survive the rewrite; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    (void)builder.relu(layer.biased);

    std::vector<Shape> shapes;
    std::vector<Graph::Lifetime> lifetimes;
    std::vector<bool> needs_grad;
    for (const Graph::Value& value : graph.values())
    {
        shapes.push_back(value.shape);
        lifetimes.push_back(value.lifetime);
        needs_grad.push_back(value.requires_grad);
    }

    (void)Graph::fuse(graph, FusionOptions{});

    ASSERT_EQ(graph.value_count(), shapes.size());
    for (ValueId id = 0; id < static_cast<ValueId>(graph.value_count()); ++id)
    {
        EXPECT_EQ(graph.value(id).shape, shapes[id]) << "value " << id;
        EXPECT_EQ(graph.value(id).lifetime, lifetimes[id]) << "value " << id;
        EXPECT_EQ(graph.value(id).requires_grad, needs_grad[id]) << "value " << id;
    }
}

TEST_F(Unit_Fusion, TheRewrittenGraphIsStillInTopologicalOrder)
{
    // The fused node sits at the root's position; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const Layer first = trace_linear(builder);
    const ValueId hidden = builder.relu(first.biased);
    const ValueId weight = builder.parameter(Shape{K_OUT_FEATURES, K_OUT_FEATURES}, "w2");
    const ValueId bias = builder.parameter(Shape{K_OUT_FEATURES}, "b2");
    const ValueId product = builder.matmul(hidden, weight, false, true);
    (void)builder.bias_add(product, bias);

    (void)Graph::fuse(graph, FusionOptions{});

    for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
        for (ValueId in : graph.node(id).inputs)
        {
            const NodeId producer = graph.value(in).producer;
            if (producer == K_NO_NODE) continue;
            EXPECT_LT(producer, id) << "node " << id << " reads value " << in
                                    << ", which node " << producer << " writes later";
        }
}

TEST_F(Unit_Fusion, TheReportNamesEverySubtreeItCollapsed)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    (void)builder.relu(layer.biased);

    const FusionReport report = Graph::fuse(graph, FusionOptions{});
    const std::string text = report.to_string();

    EXPECT_NE(text.find("gemm+bias+relu"), std::string::npos) << text;
    EXPECT_NE(text.find("dropping value"), std::string::npos) << text;
    EXPECT_EQ(report.nodes_after(), graph.node_count());
}

/** @brief Tests that fuse keeps each state's leaf, a buffer's lifetime and a refusal mark. */
TEST_F(Unit_Fusion, StateLeavesAndTheRefusalMarkSurviveTheRewrite)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x");
    const ValueId weight = builder.parameter(Shape{K_OUT_FEATURES, K_IN_FEATURES}, "w");
    const ValueId shift = builder.buffer(Shape{K_OUT_FEATURES}, "running_shift");
    (void)builder.relu(builder.bias_add(builder.matmul(x, weight, false, true), shift));

    graph.bind_state_leaf(Graph::StateId{21}, weight);
    graph.bind_state_leaf(Graph::StateId{22}, shift);
    graph.mark_refused();
    const auto bound_before = graph.state_leaves();

    const FusionReport report = Graph::fuse(graph, FusionOptions{});

    // The pass must have rebuilt the graph, or the lookups below prove nothing.
    ASSERT_EQ(report.fused().size(), 1u);
    ASSERT_EQ(count_op(graph, Ops::K_GEMM_BIAS_RELU.id), 1u);

    EXPECT_EQ(graph.state_leaf(Graph::StateId{21}), weight);
    EXPECT_EQ(graph.state_leaf(Graph::StateId{22}), shift);
    EXPECT_EQ(graph.state_leaves(), bound_before);
    EXPECT_EQ(graph.value(shift).lifetime, Graph::Lifetime::Buffer);
    EXPECT_FALSE(graph.value(shift).requires_grad);
    EXPECT_TRUE(graph.value(weight).requires_grad);
    EXPECT_TRUE(graph.has_refusal());
}

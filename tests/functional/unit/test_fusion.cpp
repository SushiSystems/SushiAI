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
using Graph::NO_NODE;
using Graph::NodeId;
using Graph::ValueId;

namespace
{
    constexpr int64_t ROWS = 4;
    constexpr int64_t IN_FEATURES = 3;
    constexpr int64_t OUT_FEATURES = 5;

    /**
     * @struct Layer
     * @brief One traced affine layer, with the ids a test wants to name.
     */
    struct Layer
    {
        ValueId x = Graph::NO_VALUE;
        ValueId weight = Graph::NO_VALUE;
        ValueId bias = Graph::NO_VALUE;
        ValueId product = Graph::NO_VALUE;   /**< The MATMUL's result. */
        ValueId biased = Graph::NO_VALUE;    /**< The BIAS_ADD's result. */
    };

    /** @brief Emits x -> matmul -> bias_add, the shape every case below builds on. */
    Layer trace_linear(const GraphBuilder& builder)
    {
        Layer layer;
        layer.x = builder.input(Shape{ROWS, IN_FEATURES}, "x");
        layer.weight = builder.parameter(Shape{OUT_FEATURES, IN_FEATURES}, "w");
        layer.bias = builder.parameter(Shape{OUT_FEATURES}, "b");
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
    constexpr FusedForm ALL[] = {
        FusedForm::GEMM_BIAS, FusedForm::GEMM_BIAS_RELU,
        FusedForm::GEMM_BIAS_SIGMOID, FusedForm::GEMM_BIAS_TANH,
        FusedForm::GEMM_BIAS_GELU,
        FusedForm::EXP_SUB_SCALE,
    };

    for (FusedForm form : ALL)
    {
        const OpID op = Graph::fused_form_op(form);
        EXPECT_NE(Ops::name_of(op), "sushiai.unknown")
            << Graph::fused_form_name(form) << " has no declared identity";
        EXPECT_EQ(Graph::fused_form_of(op), form)
            << Graph::fused_form_name(form) << " does not round-trip";
    }

    EXPECT_EQ(Graph::fused_form_of(Ops::MATMUL.id), FusedForm::NONE);
    EXPECT_EQ(Graph::fused_form_of(Ops::RELU.id), FusedForm::NONE);
}

TEST_F(Unit_Fusion, AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts)
{
    // RELU's gradient reads its output, so it survives fusion; see tests/functional/unit/README.md.
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::RELU.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::SIGMOID.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::TANH.id));

    // GELU's gradient reads the pre-activation; see tests/functional/unit/README.md.
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::GELU.id));

    // Only the entry that spills folds GELU; see tests/functional/unit/README.md.
    EXPECT_TRUE(Graph::catalog_folds_activation(Ops::GELU.id));
    EXPECT_TRUE(Graph::fused_form_spills(FusedForm::GEMM_BIAS_GELU));

    for (FusedForm form : {FusedForm::GEMM_BIAS, FusedForm::GEMM_BIAS_RELU,
                           FusedForm::GEMM_BIAS_SIGMOID, FusedForm::GEMM_BIAS_TANH,
                           FusedForm::EXP_SUB_SCALE})
        EXPECT_FALSE(Graph::fused_form_spills(form))
            << Graph::fused_form_name(form) << " grew a second output without a reason";

    // Not an activation at all, so neither question applies to it.
    EXPECT_FALSE(Graph::catalog_folds_activation(Ops::MATMUL.id));
    EXPECT_FALSE(Graph::catalog_folds_activation(Ops::EXP.id));

    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GEMM_BIAS), Ops::LEAF.id);
    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GEMM_BIAS_RELU), Ops::RELU.id);
    EXPECT_EQ(Graph::fused_form_activation(FusedForm::GEMM_BIAS_GELU), Ops::GELU.id);
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

    ASSERT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::BIAS_ADD.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::GEMM_BIAS.id), 1u);

    // The fused node writes the value the caller named; see tests/functional/unit/README.md.
    ASSERT_EQ(report.fused().size(), 1u);
    EXPECT_EQ(report.fused()[0].result, layer.biased);
    ASSERT_NE(graph.value(layer.biased).producer, NO_NODE);

    const Graph::Node& fused = graph.node(graph.value(layer.biased).producer);
    ASSERT_EQ(fused.inputs.size(), 3u);
    EXPECT_EQ(fused.inputs[0], layer.x);
    EXPECT_EQ(fused.inputs[1], layer.weight);
    EXPECT_EQ(fused.inputs[2], layer.bias);

    // The transposition flags survive the fusion; see tests/functional/unit/README.md.
    EXPECT_FALSE(fused.attributes.flag(0));
    EXPECT_TRUE(fused.attributes.flag(1));

    // The intermediate is nobody's output now.
    EXPECT_EQ(graph.value(layer.product).producer, NO_NODE);
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

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_RELU), 1u);
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 0u);
    EXPECT_EQ(graph.node_count(), report.nodes_before() - 2u);
}

TEST_F(Unit_Fusion, TheCrossEntropyGradientsElementwiseTailBecomesOnePass)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId log_probs = builder.input(Shape{ROWS, OUT_FEATURES}, "log_probs");
    const ValueId targets = builder.input(Shape{ROWS, OUT_FEATURES}, "targets");
    const ValueId probabilities = builder.exp(log_probs);
    const ValueId difference = builder.sub(probabilities, targets);
    const ValueId scaled = builder.scale(difference, 0.25);

    const std::vector<ValueId> observed{scaled};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    ASSERT_EQ(report.count(FusedForm::EXP_SUB_SCALE), 1u);

    const Graph::Node& fused = graph.node(graph.value(scaled).producer);
    ASSERT_EQ(fused.inputs.size(), 2u);
    EXPECT_EQ(fused.inputs[0], log_probs);
    EXPECT_EQ(fused.inputs[1], targets);
    EXPECT_DOUBLE_EQ(fused.attributes.get(0), 0.25);
    EXPECT_EQ(graph.value(probabilities).producer, NO_NODE);
    EXPECT_EQ(graph.value(difference).producer, NO_NODE);
}

TEST_F(Unit_Fusion, ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch)
{
    // The entry's link position refuses swapped operands; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{ROWS, OUT_FEATURES}, "a");
    const ValueId b = builder.input(Shape{ROWS, OUT_FEATURES}, "b");
    const ValueId exponent = builder.exp(a);
    const ValueId difference = builder.sub(b, exponent);
    const ValueId scaled = builder.scale(difference, 0.5);

    const std::vector<ValueId> observed{scaled};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::EXP_SUB_SCALE), 0u);
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
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_HAS_OTHER_CONSUMERS));
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 1u);
    EXPECT_EQ(count_op(graph, Ops::BIAS_ADD.id), 1u);
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

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_RELU), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_IS_OBSERVED));

    // The shorter form rooted at the bias_add still fuses; see tests/functional/unit/README.md.
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
}

TEST_F(Unit_Fusion, AParameterIsProtectedWithoutBeingListed)
{
    // A persistent value is protected without being listed; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{ROWS, OUT_FEATURES}, "a");
    const ValueId b = builder.input(Shape{ROWS, OUT_FEATURES}, "b");
    const ValueId exponent = builder.exp(a);

    // The exp's result is forced into a persistent class; see tests/functional/unit/README.md.
    graph.value(exponent).lifetime = Graph::Lifetime::OPTIMIZER_STATE;

    const ValueId difference = builder.sub(exponent, b);
    (void)builder.scale(difference, 0.5);

    const FusionReport report = Graph::fuse(graph, FusionOptions{});

    EXPECT_EQ(report.count(FusedForm::EXP_SUB_SCALE), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_IS_OBSERVED));
}

TEST_F(Unit_Fusion, AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills)
{
    // The fused node keeps writing the pre-activation; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const ValueId activated = builder.gelu(layer.biased);
    const ValueId upstream = builder.input(Shape{ROWS, OUT_FEATURES}, "dy");
    const ValueId gradient =
        builder.activation_backward(Ops::GELU_BACKWARD.id, upstream, layer.biased);

    const std::vector<ValueId> observed{activated, gradient};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    ASSERT_EQ(report.count(FusedForm::GEMM_BIAS_GELU), 1u) << report.to_string();
    EXPECT_EQ(count_op(graph, Ops::GELU.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::BIAS_ADD.id), 0u);

    // The pre-activation is the fused node's second output; see tests/functional/unit/README.md.
    ASSERT_EQ(report.fused().size(), 1u);
    EXPECT_EQ(report.fused()[0].result, activated);
    EXPECT_EQ(report.fused()[0].spilled, layer.biased);

    const NodeId producer = graph.value(layer.biased).producer;
    ASSERT_NE(producer, NO_NODE) << "the pre-activation has no producer - the spill was lost";
    EXPECT_EQ(producer, graph.value(activated).producer)
        << "the spill and the result must come from the same node";

    const Graph::Node& fused = graph.node(producer);
    ASSERT_EQ(fused.outputs.size(), 2u);
    EXPECT_EQ(fused.outputs[0], activated);
    EXPECT_EQ(fused.outputs[1], layer.biased);

    // Only the MATMUL's result is reported as eliminated; see tests/functional/unit/README.md.
    EXPECT_EQ(report.fused()[0].eliminated, std::vector<ValueId>{layer.product});
    EXPECT_EQ(graph.value(layer.product).producer, NO_NODE);

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

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_GELU), 0u) << report.to_string();
    EXPECT_TRUE(declined_for(report, FusionDecline::SPILL_READ_TOO_EARLY));

    // The shorter form still fuses after the refusal; see tests/functional/unit/README.md.
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
    EXPECT_EQ(count_op(graph, Ops::GELU.id), 1u);

    // The rewritten graph stays topological; see tests/functional/unit/README.md.
    for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
        for (ValueId in : graph.node(id).inputs)
        {
            const NodeId producer = graph.value(in).producer;
            if (producer == NO_NODE) continue;
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

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_GELU), 1u) << report.to_string();
    EXPECT_FALSE(declined_for(report, FusionDecline::VALUE_IS_OBSERVED));
    EXPECT_NE(graph.value(layer.biased).producer, NO_NODE);
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
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_HAS_OTHER_CONSUMERS));
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 1u);
}

TEST_F(Unit_Fusion, DisablingTheGeluFormFallsBackToTheShorterOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.gelu(layer.biased);

    const std::vector<ValueId> observed{activated};
    const std::vector<FusedForm> off{FusedForm::GEMM_BIAS_GELU};

    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());
    options.disabled = SushiRuntime::span<const FusedForm>(off.data(), off.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_GELU), 0u);
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
    EXPECT_TRUE(declined_for(report, FusionDecline::DISABLED));

    // The GELU left outside is not reported as unfoldable; see tests/functional/unit/README.md.
    EXPECT_FALSE(declined_for(report, FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION));
    EXPECT_FALSE(declined_for(report, FusionDecline::NO_CATALOG_ENTRY));
}

TEST_F(Unit_Fusion, ARecoverableActivationNextToAFusedGemmIsFoldedNotReported)
{
    // ReLU's gradient reads its own result, so it folds; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    const ValueId activated = builder.relu(layer.biased);
    const ValueId upstream = builder.input(Shape{ROWS, OUT_FEATURES}, "dy");
    const ValueId gradient =
        builder.activation_backward(Ops::RELU_BACKWARD.id, upstream, activated);

    const std::vector<ValueId> observed{gradient};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_RELU), 1u);
    EXPECT_FALSE(declined_for(report, FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION));
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
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 1u);
}

TEST_F(Unit_Fusion, DisablingOneFormFallsBackToTheShorterOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);
    const ValueId activated = builder.relu(layer.biased);

    const std::vector<ValueId> observed{activated};
    const std::vector<FusedForm> off{FusedForm::GEMM_BIAS_RELU};

    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());
    options.disabled = SushiRuntime::span<const FusedForm>(off.data(), off.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_RELU), 0u);
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
    EXPECT_TRUE(declined_for(report, FusionDecline::DISABLED));
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
    const ValueId weight = builder.parameter(Shape{OUT_FEATURES, OUT_FEATURES}, "w2");
    const ValueId bias = builder.parameter(Shape{OUT_FEATURES}, "b2");
    const ValueId product = builder.matmul(hidden, weight, false, true);
    (void)builder.bias_add(product, bias);

    (void)Graph::fuse(graph, FusionOptions{});

    for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
        for (ValueId in : graph.node(id).inputs)
        {
            const NodeId producer = graph.value(in).producer;
            if (producer == NO_NODE) continue;
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

/**************************************************************************/
/* test_precision.cpp                                                     */
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
 * @file test_precision.cpp
 * @brief Tests the mixed-precision policy and where the cast insertion pass places its casts.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

using namespace SushiAI;
using Graph::CastOptions;
using Graph::CastReport;
using Graph::DTypeMismatch;
using Graph::FusedForm;
using Graph::FusionOptions;
using Graph::GraphBuilder;
using Graph::InsertedCast;
using Graph::Lifetime;
using Graph::MixedPrecisionPolicy;
using Graph::K_NO_NODE;
using Graph::K_NO_VALUE;
using Graph::NodeID;
using Graph::UniformPrecisionPolicy;
using Graph::ValueID;

namespace
{
    constexpr int64_t K_ROWS = 4;
    constexpr int64_t K_IN_FEATURES = 3;
    constexpr int64_t K_HIDDEN = 5;
    constexpr int64_t K_CLASSES = 2;

    /**
     * @struct Traced
     * @brief Holds a one-hidden-layer classifier, differentiated, with the ids named.
     * @see tests/unit/README.md
     */
    struct Traced
    {
        Graph::Graph graph;
        ValueID x = K_NO_VALUE;
        ValueID targets = K_NO_VALUE;
        ValueID w1 = K_NO_VALUE;
        ValueID b1 = K_NO_VALUE;
        ValueID w2 = K_NO_VALUE;
        ValueID b2 = K_NO_VALUE;
        ValueID logits = K_NO_VALUE;
        ValueID loss = K_NO_VALUE;
        std::vector<ValueID> gradients;
    };

    /** @brief x -> linear -> relu -> linear -> log_softmax -> nll, then backward. */
    Traced trace(bool differentiate = true)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        traced.x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x");
        traced.targets = builder.input(Shape{K_ROWS, K_CLASSES}, "targets");
        traced.w1 = builder.parameter(Shape{K_HIDDEN, K_IN_FEATURES}, "w1");
        traced.b1 = builder.parameter(Shape{K_HIDDEN}, "b1");
        traced.w2 = builder.parameter(Shape{K_CLASSES, K_HIDDEN}, "w2");
        traced.b2 = builder.parameter(Shape{K_CLASSES}, "b2");

        const ValueID h = builder.relu(
            builder.bias_add(builder.matmul(traced.x, traced.w1, false, true), traced.b1));

        traced.logits = builder.bias_add(builder.matmul(h, traced.w2, false, true), traced.b2);
        traced.loss = builder.nll_loss(builder.log_softmax(traced.logits), traced.targets);

        if (!differentiate) return traced;

        const ValueID wrt[] = {traced.w1, traced.b1, traced.w2, traced.b2};
        traced.gradients = Autograd::differentiate(
            traced.graph, traced.loss, SushiRuntime::span<const ValueID>(wrt, 4));

        return traced;
    }

    /** @brief The four values a training loop reads by id. */
    std::vector<ValueID> observed_of(const Traced& traced)
    {
        return {traced.x, traced.targets, traced.logits, traced.loss};
    }

    /** @brief How many nodes of @p op the graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

    /** @brief Every mismatch, formatted, so a failure says which node and why. */
    std::string describe_all(const std::vector<DTypeMismatch>& mismatches)
    {
        std::string text;
        for (const DTypeMismatch& mismatch : mismatches)
            text += "\n  " + Graph::describe(mismatch);
        return text;
    }
}

TEST(Unit_Precision, TheUniformPolicyAnswersTheGraphsOwnTypeToEverything)
{
    const UniformPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::K_ALL)
        for (DType dtype : {DType::Half, DType::Float32, DType::Float64})
            EXPECT_EQ(policy.choose(Graph::PrecisionQuery{entry.id, dtype}).dtype, dtype)
                << entry.name << " was moved by the policy that is supposed to move nothing";

    EXPECT_EQ(policy.name(), "fp32");
}

TEST(Unit_Precision, OnlyTheProductAndWhatAGEMMFoldsIntoItNarrow)
{
    // Every narrowing operation is listed here; see tests/unit/README.md.
    const OpID narrows[] = {
        Ops::K_MATMUL.id,
        Ops::K_GEMM_BIAS.id, Ops::K_GEMM_BIAS_RELU.id,
        Ops::K_GEMM_BIAS_SIGMOID.id, Ops::K_GEMM_BIAS_TANH.id,
        Ops::K_BIAS_ADD.id,
        Ops::K_RELU.id, Ops::K_SIGMOID.id, Ops::K_TANH.id,
        Ops::K_RELU_BACKWARD.id, Ops::K_SIGMOID_BACKWARD.id, Ops::K_TANH_BACKWARD.id,
        // LEAKY_RELU narrows, as RELU does; see tests/unit/README.md.
        Ops::K_LEAKY_RELU.id, Ops::K_LEAKY_RELU_BACKWARD.id,
    };

    for (OpID op : narrows)
        EXPECT_TRUE(Graph::narrows_under_mixed_precision(op))
            << Ops::name_of(op) << " should compute low";

    // Reductions stay wide; fp16 has ~11 significand bits; see tests/unit/README.md.
    const OpID stays[] = {
        Ops::K_SUM_ROWS.id, Ops::K_LOG_SOFTMAX.id, Ops::K_NLL_LOSS.id, Ops::K_ACCUMULATE.id,
        Ops::K_EXP.id, Ops::K_SUB.id, Ops::K_SCALE.id, Ops::K_EXP_SUB_SCALE.id,
        // DYNAMIC_SCALE applies the loss scale and stays wide; see tests/unit/README.md.
        Ops::K_DYNAMIC_SCALE.id,
        Ops::K_ADD.id, Ops::K_MUL.id, Ops::K_ZERO.id,
        // GELU's x*x*x overflows fp16 at |x| ~ 40.3; see tests/unit/README.md.
        Ops::K_GELU.id, Ops::K_GELU_BACKWARD.id, Ops::K_GEMM_BIAS_GELU.id,
        Ops::K_LEAF.id, Ops::K_CAST.id,
        Ops::K_ADAMW_UPDATE.id, Ops::K_SGD_UPDATE.id, Ops::K_CLIP_SCALE.id,
        // The twenty-three spatial identities keep the default; see tests/unit/README.md.
        Ops::K_RESHAPE.id, Ops::K_LAYOUT_CONVERT.id,
        Ops::K_IM2COL.id, Ops::K_COL2IM.id,
        Ops::K_DEPTHWISE_CONV2D.id, Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id,
        Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id,
        Ops::K_MAXPOOL2D.id, Ops::K_MAXPOOL2D_BACKWARD.id,
        Ops::K_AVGPOOL2D.id, Ops::K_AVGPOOL2D_BACKWARD.id,
        Ops::K_SPATIAL_MEAN.id, Ops::K_SPATIAL_MEAN_BACKWARD.id,
        Ops::K_UPSAMPLE2D.id, Ops::K_UPSAMPLE2D_BACKWARD.id,
        Ops::K_CONCAT_CHANNELS.id, Ops::K_SLICE_CHANNELS.id,
        Ops::K_BATCHNORM2D.id, Ops::K_BATCHNORM2D_BACKWARD.id,
        Ops::K_CHANNEL_AFFINE.id, Ops::K_BATCHNORM_UPDATE_STATS.id,
        // SILU and SOFTPLUS read their input, so neither they nor their gradients narrow.
        Ops::K_SILU.id, Ops::K_SILU_BACKWARD.id, Ops::K_SOFTPLUS.id, Ops::K_SOFTPLUS_BACKWARD.id,
        Ops::K_DIV.id, Ops::K_MINIMUM.id, Ops::K_MAXIMUM.id, Ops::K_ABS.id, Ops::K_ATAN.id,
        Ops::K_LESS_EQUAL.id, Ops::K_SELECT.id, Ops::K_STOP_GRADIENT.id, Ops::K_CONSTANT.id,
        // BATCHNORM_FOLD takes the square root of a variance plus epsilon, so it stays wide.
        Ops::K_BATCHNORM_FOLD.id,
        // A row gather and its scatter-add move rows, so they keep the table's type.
        Ops::K_GATHER_ROWS.id, Ops::K_SCATTER_ADD_ROWS.id,
        // The mask copies its kept entries, and a layer normalisation takes a square root.
        Ops::K_CAUSAL_MASK.id, Ops::K_LAYER_NORM.id, Ops::K_LAYER_NORM_BACKWARD.id,
        // DROPOUT multiplies by a mask of its operand's type, so it keeps that type.
        Ops::K_DROPOUT.id,
        // The gradient of LOG_SOFTMAX sums a row and takes an exponential, as its forward does.
        Ops::K_LOG_SOFTMAX_BACKWARD.id,
    };

    for (OpID op : stays)
        EXPECT_FALSE(Graph::narrows_under_mixed_precision(op))
            << Ops::name_of(op) << " should stay in the graph's own precision";

    // The two lists account for every identity; see tests/unit/README.md.
    EXPECT_EQ(sizeof(narrows) / sizeof(narrows[0]) + sizeof(stays) / sizeof(stays[0]),
              Ops::K_ALL.size());
}

TEST(Unit_Precision, TheDepthwiseIdentitiesKeepTheirDeclaredTypeUnderMixedPrecision)
{
    // None of the three is in the narrowing set, so each keeps its declared type.
    const MixedPrecisionPolicy policy;
    const OpID depthwise[] = {
        Ops::K_DEPTHWISE_CONV2D.id,
        Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id,
        Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id,
    };

    for (OpID op : depthwise)
    {
        EXPECT_FALSE(Graph::narrows_under_mixed_precision(op)) << Ops::name_of(op);

        for (DType dtype : {DType::Float32, DType::Float64})
            EXPECT_EQ(policy.choose(Graph::PrecisionQuery{op, dtype}).dtype, dtype)
                << Ops::name_of(op) << " narrowed a " << dtype_name(dtype) << " operand";
    }
}

TEST(Unit_Precision, GELUStaysWideEvenThoughACatalogEntryNowFoldsIt)
{
    // The catalog folds GELU and the policy keeps it wide; see tests/unit/README.md.
    EXPECT_TRUE(Graph::catalog_folds_activation(Ops::K_GELU.id))
        << "the catalog no longer folds GELU, so rule 5's first reason is back and this "
           "test is guarding nothing";
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::K_GELU.id));
    EXPECT_FALSE(Graph::narrows_under_mixed_precision(Ops::K_GELU.id));

    // The fused GELU form stays wide with it; see tests/unit/README.md.
    EXPECT_FALSE(Graph::narrows_under_mixed_precision(Ops::K_GEMM_BIAS_GELU.id));
    EXPECT_TRUE(Graph::narrows_under_mixed_precision(Ops::K_GEMM_BIAS_RELU.id));

    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_RELU.id));
    EXPECT_TRUE(Graph::narrows_under_mixed_precision(Ops::K_RELU.id));
}

TEST(Unit_Precision, ChoosingIsIdempotentForEveryOperation)
{
    // choose() is idempotent for every operation; see tests/unit/README.md.
    const MixedPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::K_ALL)
        for (DType dtype : {DType::Half, DType::Float32, DType::Float64})
        {
            const DType once = policy.choose(Graph::PrecisionQuery{entry.id, dtype}).dtype;
            const DType twice = policy.choose(Graph::PrecisionQuery{entry.id, once}).dtype;
            EXPECT_EQ(once, twice) << entry.name << " from " << dtype_name(dtype);
        }
}

TEST(Unit_Precision, AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType)
{
    const MixedPrecisionPolicy policy;

    // An fp16 graph is already at the floor; see tests/unit/README.md.
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::K_MATMUL.id, DType::Float64}).dtype,
              DType::Half);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::K_SUM_ROWS.id, DType::Float64}).dtype,
              DType::Float64);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::K_MATMUL.id, DType::Half}).dtype,
              DType::Half);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::K_SUM_ROWS.id, DType::Half}).dtype,
              DType::Half);

    // A policy may narrow to fp32; see tests/unit/README.md.
    const MixedPrecisionPolicy to_single(DType::Float32);
    EXPECT_EQ(to_single.choose(Graph::PrecisionQuery{Ops::K_MATMUL.id, DType::Float64}).dtype,
              DType::Float32);
    EXPECT_EQ(to_single.choose(Graph::PrecisionQuery{Ops::K_MATMUL.id, DType::Float32}).dtype,
              DType::Float32);
}

TEST(Unit_Precision, ANonFloatLowTypeIsRefusedAtConstruction)
{
    EXPECT_THROW(MixedPrecisionPolicy{DType::Int32}, Error);
    EXPECT_THROW(MixedPrecisionPolicy{DType::Complex32}, Error);
    EXPECT_NO_THROW(MixedPrecisionPolicy{DType::Half});
}

TEST(Unit_Precision, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    const MixedPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::K_ALL)
        EXPECT_FALSE(policy.choose(Graph::PrecisionQuery{entry.id, DType::Float32}).reason.empty())
            << entry.name << " was decided without a reason";
}

TEST(Unit_CastInsertion, TheUniformPolicyInsertsNoCasts)
{
    // The fp32 policy inserts zero casts; see tests/unit/README.md.
    Traced traced = trace();

    const std::size_t nodes_before = traced.graph.node_count();
    const std::size_t values_before = traced.graph.value_count();

    const UniformPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(report.retyped_values(), 0u);
    EXPECT_EQ(count_op(traced.graph, Ops::K_CAST.id), 0u);
    EXPECT_EQ(traced.graph.node_count(), nodes_before);
    EXPECT_EQ(traced.graph.value_count(), values_before);

    for (ValueID id = 0; id < static_cast<ValueID>(values_before); ++id)
        EXPECT_EQ(traced.graph.value(id).dtype, DType::Float32);
}

TEST(Unit_CastInsertion, NoPolicyMeansTheUniformPolicy)
{
    Traced traced = trace();
    const std::size_t nodes_before = traced.graph.node_count();

    const CastReport report = Graph::insert_casts(traced.graph);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(traced.graph.node_count(), nodes_before);
}

TEST(Unit_CastInsertion, EveryNodesOperandsAgreeWithTheTypeItComputesIn)
{
    // Every operand agrees with its node's computing type; see tests/unit/README.md.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);
    EXPECT_GT(report.count(), 0u) << "a mixed policy that inserted nothing is not mixed";

    const std::vector<DTypeMismatch> mismatches =
        Graph::find_dtype_mismatches(traced.graph, policy);

    EXPECT_TRUE(mismatches.empty()) << describe_all(mismatches);
}

TEST(Unit_CastInsertion, TheInvariantSurvivesTheFusionPassThatRunsAfterIt)
{
    // Fusion runs after the casts and keeps the invariant; see tests/unit/README.md.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    const std::vector<ValueID> observed = observed_of(traced);
    const SushiRuntime::span<const ValueID> observed_span(observed.data(), observed.size());

    CastOptions casts;
    casts.policy = &policy;
    casts.observed = observed_span;
    Graph::insert_casts(traced.graph, casts);

    FusionOptions fusion;
    fusion.observed = observed_span;
    const Graph::FusionReport fused = Graph::fuse(traced.graph, fusion);

    EXPECT_GT(fused.count(FusedForm::GEMMBiasReLU), 0u)
        << "the casts landed inside a fusable chain and broke it";

    const std::vector<DTypeMismatch> mismatches =
        Graph::find_dtype_mismatches(traced.graph, policy);
    EXPECT_TRUE(mismatches.empty()) << describe_all(mismatches);

    // After the *whole* pipeline, not just after the cast pass.
    for (ValueID parameter : {traced.w1, traced.b1, traced.w2, traced.b2})
    {
        EXPECT_EQ(traced.graph.value(parameter).dtype, DType::Float32);
        EXPECT_EQ(traced.graph.value(parameter).lifetime, Lifetime::Parameter);
    }

    for (ValueID gradient : traced.gradients)
    {
        ASSERT_NE(gradient, K_NO_VALUE);
        EXPECT_EQ(traced.graph.value(gradient).dtype, DType::Float32);
        EXPECT_EQ(traced.graph.value(gradient).lifetime, Lifetime::Gradient);
    }
}

TEST(Unit_CastInsertion, ADeliberatelyBrokenGraphIsReportedRatherThanAccepted)
{
    // The checker reports a graph that is wrong; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x", DType::Float32);
    const ValueID w = builder.parameter(Shape{K_HIDDEN, K_IN_FEATURES}, "w", DType::Float32);
    builder.matmul(x, w, false, true);

    const MixedPrecisionPolicy policy;
    const std::vector<DTypeMismatch> mismatches = Graph::find_dtype_mismatches(graph, policy);

    // Two operands and one result are float32, not half; see tests/unit/README.md.
    ASSERT_EQ(mismatches.size(), 3u);
    for (const DTypeMismatch& mismatch : mismatches)
    {
        EXPECT_EQ(mismatch.op, Ops::K_MATMUL.id);
        EXPECT_EQ(mismatch.found, DType::Float32);
        EXPECT_EQ(mismatch.expected, DType::Half);
    }

    EXPECT_NE(Graph::describe(mismatches.front()).find("sushiai.matmul"), std::string::npos);
}

TEST(Unit_CastInsertion, AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy)
{
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    for (ValueID parameter : {traced.w1, traced.b1, traced.w2, traced.b2})
    {
        EXPECT_EQ(traced.graph.value(parameter).dtype, DType::Float32)
            << "a master weight was retyped in place";
        EXPECT_EQ(traced.graph.value(parameter).lifetime, Lifetime::Parameter);

        // The GEMM reads a derived fp16 copy of the parameter; see tests/unit/README.md.
        const auto casts = report.casts();
        const auto found = std::find_if(casts.begin(), casts.end(),
            [parameter](const InsertedCast& cast) { return cast.source == parameter; });

        ASSERT_NE(found, casts.end())
            << "no low-precision copy was derived from parameter " << parameter;
        EXPECT_EQ(found->to, DType::Half);
        EXPECT_TRUE(found->feeds_operand);
        EXPECT_EQ(traced.graph.value(found->destination).lifetime, Lifetime::Activation);
        EXPECT_EQ(traced.graph.value(found->destination).dtype, DType::Half);
    }
}

TEST(Unit_CastInsertion, AGradientIsWidenedBeforeItReachesItsBuffer)
{
    // An fp16 gradient is widened into its fp32 buffer; see tests/unit/README.md.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    // Only the weight gradients come from a low MATMUL; see tests/unit/README.md.
    for (ValueID weight_gradient : {traced.gradients[0], traced.gradients[2]})
    {
        ASSERT_NE(weight_gradient, K_NO_VALUE);
        EXPECT_EQ(traced.graph.value(weight_gradient).dtype, DType::Float32);

        const NodeID producer = traced.graph.value(weight_gradient).producer;
        ASSERT_NE(producer, K_NO_NODE);
        EXPECT_EQ(traced.graph.node(producer).op, Ops::K_CAST.id)
            << "an fp16 matmul wrote straight into an fp32 gradient buffer";

        const ValueID narrow = traced.graph.node(producer).inputs.at(0);
        EXPECT_EQ(traced.graph.value(narrow).dtype, DType::Half);
        EXPECT_EQ(traced.graph.node(traced.graph.value(narrow).producer).op, Ops::K_MATMUL.id);
    }

    // The bias gradients come from a wide reduction; see tests/unit/README.md.
    for (ValueID bias_gradient : {traced.gradients[1], traced.gradients[3]})
    {
        EXPECT_EQ(traced.graph.value(bias_gradient).dtype, DType::Float32);
        EXPECT_EQ(traced.graph.node(traced.graph.value(bias_gradient).producer).op,
                  Ops::K_SUM_ROWS.id);
    }

    EXPECT_EQ(report.count_into(DType::Float32) > 0, true);
}

TEST(Unit_CastInsertion, AccumulatingGradientContributionsHappensWide)
{
    // ACCUMULATE's result aliases its accumulator's buffer; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x");
    const ValueID targets = builder.input(Shape{K_ROWS, K_CLASSES}, "targets");
    const ValueID w = builder.parameter(Shape{K_CLASSES, K_IN_FEATURES}, "w");

    // The same weight in two products, summed - the smallest fan-out there is.
    const ValueID a = builder.matmul(x, w, false, true);
    const ValueID b = builder.matmul(x, w, false, true);
    const ValueID logits = builder.add(a, b);
    const ValueID loss = builder.nll_loss(builder.log_softmax(logits), targets);

    const ValueID wrt[] = {w};
    const std::vector<ValueID> gradients =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt, 1));

    ASSERT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 1u);

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const ValueID observed[] = {x, targets, logits, loss};
    options.observed = SushiRuntime::span<const ValueID>(observed, 4);

    Graph::insert_casts(graph, options);

    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, policy).empty());
    EXPECT_EQ(graph.value(gradients[0]).dtype, DType::Float32);

    for (const Graph::Node& node : graph.nodes())
        if (node.op == Ops::K_ACCUMULATE.id)
        {
            EXPECT_EQ(graph.value(node.outputs[0]).dtype, DType::Float32);
            EXPECT_EQ(graph.value(node.inputs[0]).dtype, DType::Float32);
            EXPECT_EQ(graph.value(node.inputs[1]).dtype, DType::Float32);
        }
}

TEST(Unit_CastInsertion, AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot)
{
    // An observed activation keeps its dtype; see tests/unit/README.md.
    Traced pinned = trace();
    Traced loose = trace();

    const MixedPrecisionPolicy policy;

    CastOptions with;
    with.policy = &policy;
    const std::vector<ValueID> observed = observed_of(pinned);
    with.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
    Graph::insert_casts(pinned.graph, with);

    CastOptions without;
    without.policy = &policy;
    Graph::insert_casts(loose.graph, without);

    EXPECT_EQ(pinned.graph.value(pinned.logits).dtype, DType::Float32);
    EXPECT_EQ(loose.graph.value(loose.logits).dtype, DType::Half);

    // Naming a value that does not exist is a caller error worth reporting.
    Traced other = trace();
    CastOptions bad;
    bad.policy = &policy;
    const ValueID nonsense[] = {static_cast<ValueID>(other.graph.value_count() + 10)};
    bad.observed = SushiRuntime::span<const ValueID>(nonsense, 1);
    EXPECT_THROW(Graph::insert_casts(other.graph, bad), Error);
}

TEST(Unit_CastInsertion, AValueReadTwiceIsConvertedOnce)
{
    // w2 has two readers and one conversion; see tests/unit/README.md.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    for (const InsertedCast& cast : report.casts())
    {
        if (!cast.feeds_operand) continue;

        const std::size_t same_source = static_cast<std::size_t>(
            std::count_if(report.casts().begin(), report.casts().end(),
                          [&cast](const InsertedCast& other)
                          {
                              return other.feeds_operand
                                  && other.source == cast.source
                                  && other.to == cast.to;
                          }));

        EXPECT_EQ(same_source, 1u)
            << "value " << cast.source << " was converted to "
            << dtype_name(cast.to) << " more than once";
    }

    // And the reuse actually happened: w2 feeds two nodes in the narrow type.
    const auto casts = report.casts();
    const auto found = std::find_if(casts.begin(), casts.end(),
        [&traced](const InsertedCast& cast) { return cast.source == traced.w2; });
    ASSERT_NE(found, casts.end());

    std::size_t readers = 0;
    for (const Graph::Node& node : traced.graph.nodes())
        readers += static_cast<std::size_t>(
            std::count(node.inputs.begin(), node.inputs.end(), found->destination));

    EXPECT_EQ(readers, 2u);
}

TEST(Unit_CastInsertion, ValueIDsSurviveThePass)
{
    Traced traced = trace();
    const std::size_t values_before = traced.graph.value_count();

    std::vector<Shape> shapes;
    std::vector<Lifetime> lifetimes;
    for (ValueID id = 0; id < static_cast<ValueID>(values_before); ++id)
    {
        shapes.push_back(traced.graph.value(id).shape);
        lifetimes.push_back(traced.graph.value(id).lifetime);
    }

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    Graph::insert_casts(traced.graph, options);

    EXPECT_GT(traced.graph.value_count(), values_before);
    for (ValueID id = 0; id < static_cast<ValueID>(values_before); ++id)
    {
        EXPECT_EQ(traced.graph.value(id).shape, shapes[id]);
        EXPECT_EQ(traced.graph.value(id).lifetime, lifetimes[id]);
    }
}

TEST(Unit_CastInsertion, RunningThePassTwiceAddsNothingTheSecondTime)
{
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport first = Graph::insert_casts(traced.graph, options);
    const std::size_t nodes = traced.graph.node_count();
    const std::size_t values = traced.graph.value_count();

    const CastReport second = Graph::insert_casts(traced.graph, options);

    EXPECT_GT(first.count(), 0u);
    EXPECT_EQ(second.count(), 0u);
    EXPECT_EQ(traced.graph.node_count(), nodes);
    EXPECT_EQ(traced.graph.value_count(), values);
}

TEST(Unit_CastInsertion, TheReportNamesEveryConversionAndItsDirection)
{
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    EXPECT_EQ(report.policy(), "mixed");
    EXPECT_EQ(report.count(), count_op(traced.graph, Ops::K_CAST.id));
    EXPECT_EQ(report.nodes_after() - report.nodes_before(), report.count());
    EXPECT_EQ(report.count(), report.count_into(DType::Half) + report.count_into(DType::Float32));
    EXPECT_GT(report.count_into(DType::Half), 0u);
    EXPECT_GT(report.count_into(DType::Float32), 0u);
    EXPECT_GT(report.retyped_values(), 0u);

    EXPECT_NE(report.to_string().find("mixed"), std::string::npos);
    EXPECT_NE(report.to_string().find("casts inserted"), std::string::npos);
}

TEST(Unit_CastInsertion, AGraphThatIsNotInTopologicalOrderIsRefused)
{
    // A conversion sits immediately before its consumer; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueID a = graph.add_value(Shape{K_ROWS}, DType::Float32, Lifetime::Activation, "a");
    const ValueID b = graph.add_value(Shape{K_ROWS}, DType::Float32, Lifetime::Activation, "b");

    graph.add_node(Ops::K_RELU.id, {b}, {a});   // reads b...
    graph.add_node(Ops::K_RELU.id, {a}, {b});   // ...which is written here.

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    EXPECT_THROW(Graph::insert_casts(graph, options), Error);
}

namespace
{
    /**
     * @struct Pooled
     * @brief Holds a maxpool graph, differentiated, with the index buffer named.
     * @see tests/unit/README.md
     */
    struct Pooled
    {
        Graph::Graph graph;
        ValueID x = K_NO_VALUE;
        ValueID indices = K_NO_VALUE;
        ValueID loss = K_NO_VALUE;
    };

    Pooled trace_pool()
    {
        Pooled p;
        const GraphBuilder builder(p.graph);

        p.x = builder.parameter(Shape{2, 8, 8, 4}, "x");
        const ValueID w = builder.parameter(Shape{4, 4}, "w");

        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        g.stride_h = 2; g.stride_w = 2;

        const Graph::PoolResult pool = builder.maxpool2d(p.x, g);
        p.indices = pool.indices;

        const ValueID rows = builder.reshape(builder.spatial_mean(pool.output), Shape{2, 4});
        const ValueID hidden = builder.relu(builder.matmul(rows, w));
        const ValueID target = builder.input(Shape{2, 4}, "target");

        p.loss = builder.nll_loss(builder.log_softmax(hidden), target);
        p.graph.finalise();

        const std::vector<ValueID> wrt{w, p.x};
        (void)Autograd::differentiate(p.graph, p.loss,
            SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

        return p;
    }

    /** @brief Whether any CAST node in @p graph reads or writes @p value. */
    bool any_cast_touches(const Graph::Graph& graph, ValueID value)
    {
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op != Ops::K_CAST.id) continue;
            for (ValueID in : node.inputs) if (in == value) return true;
            for (ValueID out : node.outputs) if (out == value) return true;
        }
        return false;
    }
}

TEST(Unit_CastInsertion, AnIndexBufferHasNoPrecisionToChoose)
{
    // The INT64 argmax is not converted; see tests/unit/README.md.
    Pooled p = trace_pool();

    ASSERT_EQ(p.graph.value(p.indices).dtype, DType::Int64);
    ASSERT_EQ(count_op(p.graph, Ops::K_MAXPOOL2D_BACKWARD.id), 1u)
        << "no pool backward was emitted, so the operand half of this test is vacuous";

    EXPECT_NO_THROW(Graph::insert_casts(p.graph, CastOptions{}));

    EXPECT_EQ(p.graph.value(p.indices).dtype, DType::Int64)
        << "the argmax was retyped; its backward reads it as a routing table of offsets";
    EXPECT_FALSE(any_cast_touches(p.graph, p.indices))
        << "a CAST was wrapped around the index buffer, which is a rounding policy nobody asked for";
}

TEST(Unit_CastInsertion, AMixedPolicyLeavesTheIndexBufferAloneToo)
{
    Pooled p = trace_pool();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueID> observed{p.x, p.loss};
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    EXPECT_NO_THROW(Graph::insert_casts(p.graph, options));

    EXPECT_EQ(p.graph.value(p.indices).dtype, DType::Int64);
    EXPECT_FALSE(any_cast_touches(p.graph, p.indices));
}

TEST(Unit_CastInsertion, APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer)
{
    // A pinned argmax keeps its id and its INT64 buffer; see tests/unit/README.md.
    Pooled p = trace_pool();

    const std::vector<ValueID> observed{p.indices};
    CastOptions options;
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());

    EXPECT_NO_THROW(Graph::insert_casts(p.graph, options));

    EXPECT_EQ(p.graph.value(p.indices).dtype, DType::Int64);
    EXPECT_FALSE(any_cast_touches(p.graph, p.indices));
}

/** @brief Tests that the cast pass keeps a buffer's type, lifetime and state leaf. */
TEST(Unit_CastInsertion, ABufferKeepsItsDeclaredTypeAndItsStateLeafAcrossThePass)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{K_ROWS, K_IN_FEATURES}, "x");
    const ValueID weight = builder.parameter(Shape{K_HIDDEN, K_IN_FEATURES}, "w");
    const ValueID shift = builder.buffer(Shape{K_HIDDEN}, "running_shift");
    const ValueID biased = builder.bias_add(builder.matmul(x, weight, false, true), shift);

    graph.bind_state_leaf(Graph::StateID{31}, weight);
    graph.bind_state_leaf(Graph::StateID{32}, shift);
    graph.mark_refused();
    const auto bound_before = graph.state_leaves();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const CastReport report = Graph::insert_casts(graph, options);

    // BIAS_ADD narrows, so the pass must have cast the buffer for the lookups to prove anything.
    ASSERT_GT(report.count(), 0u);
    EXPECT_EQ(graph.value(biased).dtype, DType::Half);

    EXPECT_EQ(graph.value(shift).dtype, DType::Float32);
    EXPECT_EQ(graph.value(shift).lifetime, Lifetime::Buffer);
    EXPECT_FALSE(graph.value(shift).requires_grad);
    EXPECT_EQ(graph.state_leaf(Graph::StateID{31}), weight);
    EXPECT_EQ(graph.state_leaf(Graph::StateID{32}), shift);
    EXPECT_EQ(graph.state_leaves(), bound_before);
    EXPECT_TRUE(graph.has_refusal());

    const std::vector<DTypeMismatch> mismatches = Graph::find_dtype_mismatches(graph, policy);
    EXPECT_TRUE(mismatches.empty()) << describe_all(mismatches);
}

/** @brief Tests that the mixed policy leaves BATCHNORM_FOLD and its four operands in fp32. */
TEST(Unit_CastInsertion, BatchNormFoldStaysInTheMasterTypeUnderTheMixedPolicy)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID gamma = builder.parameter(Shape{K_HIDDEN}, "gamma");
    const ValueID beta = builder.parameter(Shape{K_HIDDEN}, "beta");
    const ValueID mean = builder.buffer(Shape{K_HIDDEN}, "running_mean");
    const ValueID variance = builder.buffer(Shape{K_HIDDEN}, "running_var");
    const Graph::BatchNormFold fold = builder.batchnorm_fold(gamma, beta, mean, variance, 1e-3);

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const CastReport report = Graph::insert_casts(graph, options);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_CAST.id), 0u);
    ASSERT_EQ(count_op(graph, Ops::K_BATCHNORM_FOLD.id), 1u);

    for (ValueID id : {gamma, beta, mean, variance, fold.scale, fold.shift})
        EXPECT_EQ(graph.value(id).dtype, DType::Float32) << "value " << id;
}

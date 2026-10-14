/**************************************************************************/
/* test_backward.cpp                                                      */
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
 * @file test_backward.cpp
 * @brief Tests differentiation: the graph it emits, its loss roots, its diagnostics, precision,
 *        layout and rematerialisation.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/rematerialise.hpp>

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

namespace SushiAI
{
    namespace Test
    {
        using Graph::GraphBuilder;
        using Graph::TensorLayout;
        using Graph::ValueID;

        TEST(Unit_Layout, EveryValueIsNHWCUnlessSaidOtherwise)
        {
            // A value that states no layout is NHWC; see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 4}, "x");

            EXPECT_EQ(graph.values()[x].layout, TensorLayout::NHWC);
        }

        TEST(Unit_Layout, FinaliseAcceptsAUniformGraph)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 4}, "x");
            const ValueID w = builder.parameter({4, 3}, "w");
            (void)builder.matmul(x, w);

            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, FinaliseRejectsAMixedGraphAndNamesTheValue)
        {
            // A mixed graph fails finalise(); see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 4}, "x");
            const ValueID w = builder.parameter({4, 3}, "w");
            const ValueID y = builder.matmul(x, w);

            graph.value(y).layout = TensorLayout::NCHW;

            EXPECT_THROW(graph.finalise(), Error);
        }

        TEST(Unit_Layout, LayoutConvertIsAllowedToChangeIt)
        {
            // LAYOUT_CONVERT alone may change the layout; see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 3, 4, 5}, "x");
            const ValueID y = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.values()[y].layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, DeclareLikeInheritsTheSourceValuesLayout)
        {
            // declare_like carries the layout forward; see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 3, 4, 5}, "x");
            const ValueID y = builder.layout_convert(x, TensorLayout::NCHW);

            const ValueID z = builder.declare_like(y, graph.value(y).shape,
                                                    graph.value(y).dtype,
                                                    Graph::Lifetime::Activation, "z");

            EXPECT_EQ(graph.values()[z].layout, TensorLayout::NCHW);
        }

        namespace
        {
            /**
             * @brief Holds a graph that crosses layouts, with the NCHW side named.
             * @see tests/unit/README.md
             */
            struct Crossed
            {
                Graph::Graph graph;
                ValueID nhwc = Graph::K_NO_VALUE;
                ValueID nchw = Graph::K_NO_VALUE;
            };

            Crossed make_crossed()
            {
                Crossed c;
                const GraphBuilder builder(c.graph);

                c.nhwc = builder.input({2, 3, 4, 5}, "x");
                c.nchw = builder.layout_convert(c.nhwc, TensorLayout::NCHW);
                c.graph.finalise();

                return c;
            }
        }

        TEST(Unit_Layout, FusionCarriesTheLayoutTagAcrossItsRebuild)
        {
            // fuse() copies the layout tag of every value; see tests/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "fuse() reset a NCHW value to NHWC; the convert that produced it is now "
                   "a no-op that copies transposed data through";
        }

        TEST(Unit_Layout, CastInsertionCarriesTheLayoutTagAcrossItsRebuild)
        {
            // Cast insertion copies the layout tag as well; see tests/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "insert_casts() reset a NCHW value to NHWC";
        }

        TEST(Unit_Layout, ARebuiltGraphStillProvesUniform)
        {
            // The proof is re-run on tags that survived; see tests/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});
            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(c.graph.finalise());
        }

        TEST(Unit_Layout, TheThreeRulesThatEmitZerosCannotBeReachedWithAnNCHWOperand)
        {
            // Rank-4 emitters refuse an NCHW operand (CV-3); see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);

            const ValueID x = builder.input({1, 2, 2, 2}, "x");
            const ValueID nchw = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_THROW((void)builder.slice_channels(nchw, 0, 1), Error);
            EXPECT_THROW((void)builder.channel_affine(nchw, x, x), Error);

            // finalise() rejects the node ahead of the rule; see tests/unit/README.md.
            const ValueID rows = builder.reshape(nchw, Shape{1, 8});
            EXPECT_EQ(graph.value(rows).layout, TensorLayout::NCHW);

            const ValueID target = builder.input({1, 8}, "target");
            EXPECT_EQ(graph.value(target).layout, TensorLayout::NHWC);

            const ValueID summed = builder.sum_rows(rows);
            EXPECT_EQ(graph.value(summed).layout, TensorLayout::NCHW);

            (void)builder.nll_loss(builder.log_softmax(rows), target);
            EXPECT_THROW(graph.finalise(), Error);
        }

        TEST(Unit_Layout, ZerosLikeCarriesTheTagAndPlainZerosCannot)
        {
            // zeros_like() inherits its operand's layout; see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);

            const ValueID x = builder.input({1, 2, 2, 2}, "x");
            const ValueID nchw = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.value(builder.zeros(Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NHWC);
            EXPECT_EQ(graph.value(builder.zeros_like(nchw, Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NCHW);
            EXPECT_EQ(graph.value(builder.zeros_like(x, Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NHWC);
        }

        TEST(Unit_Layout, FinaliseStillPropagatesRequiresGrad)
        {
            // finalise() still propagates requires_grad; see tests/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueID x = builder.input({2, 4}, "x");
            const ValueID w = builder.parameter({4, 3}, "w");
            const ValueID y = builder.matmul(x, w);

            graph.finalise();

            EXPECT_TRUE(graph.values()[y].requires_grad);
        }
    }
}

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

namespace
{
    /** @brief The nodes of @p graph reading @p value, in graph order. */
    std::vector<Graph::NodeID> readers_of(const Graph::Graph& graph, ValueID value)
    {
        std::vector<Graph::NodeID> readers;
        for (std::size_t i = 0; i < graph.node_count(); ++i)
            for (ValueID in : graph.nodes()[i].inputs)
                if (in == value) { readers.push_back(static_cast<Graph::NodeID>(i)); break; }
        return readers;
    }

    /** @brief How many IM2COL nodes @p graph holds. */
    std::size_t im2col_count(const Graph::Graph& graph)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == Ops::K_IM2COL.id) ++count;
        return count;
    }

    /**
     * @brief A convolution as a GEMM, carried on to a scalar NLL_LOSS root.
     *
     * The loss tail is what makes the graph differentiable at all; it is not
     * part of what is being tested.
     */
    struct ConvGraph
    {
        Graph::Graph graph;
        ValueID cols = Graph::K_NO_VALUE;
        ValueID weight = Graph::K_NO_VALUE;
        ValueID loss = Graph::K_NO_VALUE;
    };

    /** @brief Builds and differentiates a one-convolution graph. */
    void build_and_differentiate(ConvGraph& c, Shape image, Shape weight_shape)
    {
        const GraphBuilder builder(c.graph);
        const ValueID x = builder.input(image, "x");
        c.weight = builder.parameter(weight_shape, "w");

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        c.cols = builder.im2col(x, g);
        const ValueID out = builder.matmul(c.cols, c.weight);

        const Shape logits = c.graph.value(out).shape;
        const ValueID target = builder.input(logits, "target");
        c.loss = builder.nll_loss(builder.log_softmax(out), target);
        c.graph.finalise();

        const std::array<ValueID, 1> wrt = {c.weight};
        (void)Autograd::differentiate(c.graph, c.loss,
            SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    }
}

TEST(Unit_Rematerialise, TheColumnBufferStopsBeingAnActivation)
{
    // The column buffer's lifetime drops to SCRATCH; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{2, 5, 5, 3}, Shape{27, 8});

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::Activation);
    ASSERT_GT(readers_of(c.graph, c.cols).size(), std::size_t{1})
        << "the backward pass did not read the column matrix, so there is "
           "nothing for the pass to do and this test proves nothing";

    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::Scratch);

    // And it really did die early: exactly one node still reads the original.
    EXPECT_EQ(readers_of(c.graph, c.cols).size(), std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneComputesTheSameThingAsTheOriginal)
{
    // The clone reads the same operands and attributes; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    const std::size_t before = c.graph.node_count();
    ASSERT_EQ(im2col_count(c.graph), std::size_t{1});

    Graph::rematerialise_im2col(c.graph);

    EXPECT_GT(c.graph.node_count(), before) << "no clone was inserted";
    EXPECT_EQ(c.graph.node_count(), before + 1) << "one clone, shared by the later consumers";

    const Graph::NodeID original_id = c.graph.value(c.cols).producer;
    const Graph::Node& original = c.graph.node(original_id);

    std::size_t clones = 0;
    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
    {
        const Graph::Node& n = c.graph.nodes()[i];
        if (n.op != Ops::K_IM2COL.id || i == original_id) continue;
        ++clones;
        EXPECT_EQ(n.inputs, original.inputs);
        for (std::size_t slot = 0; slot < Graph::Attributes::K_CAPACITY; ++slot)
            EXPECT_EQ(n.attributes.get(slot), original.attributes.get(slot))
                << "the clone lost attribute slot " << slot;

        // The clone keeps the description and the layout; see tests/unit/README.md.
        const Graph::Value& copy = c.graph.value(n.outputs.front());
        const Graph::Value& stored = c.graph.value(c.cols);
        EXPECT_EQ(copy.shape, stored.shape);
        EXPECT_EQ(copy.dtype, stored.dtype);
        EXPECT_EQ(copy.layout, stored.layout);
        EXPECT_EQ(copy.lifetime, Graph::Lifetime::Scratch);
    }
    EXPECT_EQ(clones, std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneIsInsertedAheadOfEveryConsumerItServes)
{
    // A clone precedes every consumer it serves; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});
    Graph::rematerialise_im2col(c.graph);

    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
        for (ValueID in : c.graph.nodes()[i].inputs)
        {
            const Graph::NodeID producer = c.graph.value(in).producer;
            if (producer == Graph::K_NO_NODE) continue;
            EXPECT_LT(producer, i) << "node " << i << " reads value " << in
                                   << ", which node " << producer << " writes";
        }

    // The invariant, asserted by the component that depends on it.
    EXPECT_NO_THROW((void)Graph::MemoryPlan::build(c.graph));

    // And the graph's own proofs still hold after the rewrite.
    EXPECT_NO_THROW(c.graph.finalise());
}

TEST(Unit_Rematerialise, AnIm2colWithNoBackwardConsumerIsLeftAlone)
{
    // An inference graph has no second consumer; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 4, 4, 2}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;

    const ValueID cols = builder.im2col(x, g);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::Activation);
}

TEST(Unit_Rematerialise, AForwardOnlyConvolutionIsLeftAlone)
{
    // A column matrix with one reader is not cloned; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 4, 4, 2}, "x");
    const ValueID w = builder.input(Shape{18, 4}, "w");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;

    const ValueID cols = builder.im2col(x, g);
    (void)builder.matmul(cols, w);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::Activation);
}

TEST(Unit_Rematerialise, RunningItTwiceAddsNothingTheSecondTime)
{
    // The clone has one consumer, so a second run adds none; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    Graph::rematerialise_im2col(c.graph);
    const std::size_t once = c.graph.node_count();
    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.node_count(), once);
}

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

    /** @brief Bounds the error allowed on an attribute that a rule writes as an exact constant. */
    constexpr double K_ATTRIBUTE_TOLERANCE = 1e-12;

    /** @brief Counts the rules that no finite-difference case names; an entry is a visible edit. */
    constexpr std::size_t K_UNPROVEN_RULES = 8;

    /** @brief Names one forward operation and how to emit it over two parameters. */
    struct ForwardCase
    {
        std::string_view name;                                        /**< @brief Failure label. */
        ValueID (*emit)(const GraphBuilder&, ValueID a, ValueID b);   /**< @brief The emitter. */
    };

    /** @brief Holds what differentiating one operation under an NLL_LOSS root appended. */
    struct Appended
    {
        std::size_t first = 0;            /**< @brief Index of the first appended node. */
        ValueID seed = K_NO_VALUE;          /**< @brief Gradient the tested operation receives. */
        std::vector<ValueID> gradients;   /**< @brief One entry per requested value. */
    };

    /**
     * @brief Differentiates nll_loss(@p result, targets) with respect to @p wrt.
     * @param result A rank-2 FLOAT32 value; its producer is the operation under test.
     * @return The appended range; its first node is the SCALE that NLL_LOSS's rule emits.
     */
    Appended differentiate_through(Graph::Graph& graph, ValueID result,
                                   const std::vector<ValueID>& wrt)
    {
        const GraphBuilder builder(graph);
        const ValueID targets = builder.input(graph.value(result).shape, "targets");
        const ValueID loss = builder.nll_loss(result, targets);

        Appended appended;
        appended.first = graph.node_count();
        appended.gradients = Autograd::differentiate(
            graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

        if (graph.node_count() > appended.first)
            appended.seed = graph.nodes()[appended.first].outputs.at(0);
        return appended;
    }

    /** @brief Returns the node @p offset places after the first appended one. */
    const Graph::Node& appended_node(const Graph::Graph& graph, const Appended& appended,
                                     std::size_t offset)
    {
        return graph.nodes().at(appended.first + offset);
    }

} // namespace

TEST(Unit_Backward, EveryImplementedOperationHasARule)
{
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_MATMUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_BIAS_ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SUB.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_MUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SCALE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_EXP.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_RELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_GELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SIGMOID.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_TANH.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SUM_ROWS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_ACCUMULATE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_NLL_LOSS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_MAXPOOL2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_AVGPOOL2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_UPSAMPLE2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SPATIAL_MEAN.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_CONCAT_CHANNELS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SLICE_CHANNELS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_BATCHNORM2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_CHANNEL_AFFINE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_DIV.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_MINIMUM.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_MAXIMUM.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_ABS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_ATAN.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SOFTPLUS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SELECT.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_SILU.id));

    // A gradient barrier is stepped over by the reverse walk, so no rule is ever asked of it.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_LESS_EQUAL.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_CONSTANT.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_STOP_GRADIENT.id));

    // An activation gradient has no rule; a second differentiation over it throws.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_SILU_BACKWARD.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_SOFTPLUS_BACKWARD.id));

    // Running statistics are a buffer, not a gradient path; see tests/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_BATCHNORM_UPDATE_STATS.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_BATCHNORM_FOLD.id));

    // LEAF and ZERO have no operand to hand a gradient to; see tests/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_LEAF.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_ZERO.id));

    // Casts are inserted after differentiation; see tests/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_CAST.id));
}

TEST(Unit_Backward, EveryRuleInTheTableIsNamedByANumericTest)
{
    // Every rule names the numeric test that proves it; see tests/unit/README.md.
    struct Proven
    {
        OpID op;
        std::string_view proof;
    };

    const std::vector<Proven> ledger = {
        {Ops::K_MATMUL.id,     "Integration_Autograd.MatmulWithoutTransposes"},
        {Ops::K_MATMUL.id,     "Integration_Autograd.MatmulWithATransposedRightOperand"},
        {Ops::K_MATMUL.id,     "Integration_Autograd.MatmulWithATransposedLeftOperand"},
        {Ops::K_MATMUL.id,     "Integration_Autograd.MatmulWithBothOperandsTransposed"},
        {Ops::K_MATMUL.id,     "Integration_Autograd.MatmulGradientOfTheRightOperand"},
        {Ops::K_GATHER_ROWS.id,
         "Integration_Autograd.GatherRowsScatterAddsARepeatedIndexIntoItsTable"},
        {Ops::K_GATHER_ROWS.id,
         "Integration_Autograd.GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape"},
        {Ops::K_SCATTER_ADD_ROWS.id,
         "Integration_Autograd.ScatterAddRowsGathersARepeatedIndexBackOntoItsRows"},
        {Ops::K_SCATTER_ADD_ROWS.id,
         "Integration_Autograd.ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape"},
        {Ops::K_LAYER_NORM.id,
         "Integration_LayerNorm.TheInputGradientMatchesCentralDifferences"},
        {Ops::K_LAYER_NORM.id,
         "Integration_LayerNorm.TheGainGradientMatchesCentralDifferences"},
        {Ops::K_LAYER_NORM.id,
         "Integration_LayerNorm.TheBiasGradientMatchesCentralDifferences"},
        {Ops::K_CAUSAL_MASK.id,
         "Integration_CausalAttention.TheMaskPassesItsGradientThroughTheKeptEntriesAlone"},
        {Ops::K_DROPOUT.id,
         "Integration_Dropout.TheGradientAtTheInputIsTheMaskOfThatStep"},
        {Ops::K_BIAS_ADD.id,   "Integration_Autograd.BiasAddPassesTheMatrixGradientThrough"},
        {Ops::K_BIAS_ADD.id,   "Integration_Autograd.BiasAddSumsOverTheBroadcastAxis"},
        {Ops::K_ADD.id,        "Integration_Autograd.ElementwiseAddAndSubtract"},
        {Ops::K_SUB.id,        "Integration_Autograd.ElementwiseSubtractFromTheRight"},
        {Ops::K_MUL.id,        "Integration_Autograd.ElementwiseMultiply"},
        {Ops::K_MUL.id,        "Integration_Autograd.ElementwiseMultiplyFromTheRight"},
        {Ops::K_SCALE.id,      "Integration_Autograd.Scale"},
        {Ops::K_EXP.id,        "Integration_Autograd.Exp"},
        {Ops::K_RELU.id,       "Integration_Autograd.ReLUAwayFromItsKink"},
        {Ops::K_GELU.id,       "Integration_Autograd.GELU"},
        {Ops::K_SIGMOID.id,    "Integration_Autograd.SigmoidReadsItsForwardOutput"},
        {Ops::K_TANH.id,       "Integration_Autograd.TanhReadsItsForwardOutput"},
        {Ops::K_LEAKY_RELU.id, "Integration_Autograd.LeakyReLUHasBothOfItsSlopes"},
        {Ops::K_SUM_ROWS.id,   "Integration_Autograd.SumRowsBroadcastsItsGradientBack"},
        {Ops::K_ACCUMULATE.id, "Integration_Autograd.AccumulateInTheForwardGraph"},
        {Ops::K_NLL_LOSS.id,   "Integration_Autograd.FusedCrossEntropyAgainstOneHotTargets"},

        {Ops::K_LOG_SOFTMAX.id,
         "Integration_AttentionGradient.AnUnfusedLogSoftmaxMatchesItsClosedForm"},
        {Ops::K_LOG_SOFTMAX.id,
         "Integration_AttentionGradient.ASoftmaxGradientMatchesCentralDifferences"},
        {Ops::K_LOG_SOFTMAX.id,
         "Integration_AttentionGradient.ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes"},
        {Ops::K_LOG_SOFTMAX.id,
         "Integration_AttentionGradient.ACausalAttentionDifferentiatesAtItsInput"},
        {Ops::K_LOG_SOFTMAX.id,
         "Integration_AttentionGradient.ACausalAttentionDifferentiatesAtItsProjectionWeight"},

        {Ops::K_RESHAPE.id,        "Integration_Autograd.ReshapeMovesItsGradientBackUnchanged"},
        {Ops::K_LAYOUT_CONVERT.id, "Integration_Autograd.LayoutConvertsGradientIsThePermutedGradient"},

        {Ops::K_IM2COL.id, "Integration_Autograd.AWholeConvolutionDifferentiatesThroughIm2colAndMatmul"},
        {Ops::K_COL2IM.id, "Integration_Autograd.Col2imsGradientGathersEveryWindowThatWroteAPixel"},

        {Ops::K_MAXPOOL2D.id, "Integration_Autograd.MaxPoolRoutesEachGradientToTheElementThatWon"},
        {Ops::K_AVGPOOL2D.id, "Integration_Autograd.AvgPoolSpreadsItsGradientOverTheWholeWindow"},
        {Ops::K_UPSAMPLE2D.id, "Integration_Autograd.UpsampleSumsEachSourcePixelsReplicas"},
        {Ops::K_DEPTHWISE_CONV2D.id,
         "Integration_Autograd.DepthwiseConvolutionDifferentiatesOverItsInputAndItsTaps"},
        {Ops::K_SPATIAL_MEAN.id, "Integration_Autograd.SpatialMeanBroadcastsScaledByOneOverHW"},
        {Ops::K_CONCAT_CHANNELS.id, "Integration_Autograd.ConcatGivesEachOperandItsOwnChannelRange"},
        {Ops::K_SLICE_CHANNELS.id, "Integration_Autograd.SliceScattersItsGradientBackIntoZeros"},

        {Ops::K_BATCHNORM2D.id, "Integration_Autograd.BatchNormsGradientAccountsForTheStatisticsItComputed"},
        {Ops::K_BATCHNORM2D.id, "Integration_Autograd.BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput"},
        {Ops::K_BATCHNORM2D.id, "Integration_Autograd.BatchNormsShiftGradientIsThePlainSumOfDy"},

        {Ops::K_CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffineDifferentiatesAtItsInput"},
        {Ops::K_CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffinesScaleGradientSumsOverNHAndW"},
        {Ops::K_CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffinesShiftGradientSumsOverNHAndW"},
    };

    // No finite-difference case names these rules; an entry leaves this list when one does.
    const std::vector<OpID> unproven = {
        Ops::K_DIV.id, Ops::K_MINIMUM.id, Ops::K_MAXIMUM.id, Ops::K_ABS.id, Ops::K_ATAN.id,
        Ops::K_SOFTPLUS.id, Ops::K_SELECT.id, Ops::K_SILU.id,
    };

    EXPECT_EQ(unproven.size(), K_UNPROVEN_RULES)
        << "the unproven list changed size; a rule may join it only with K_UNPROVEN_RULES edited";

    for (const Proven& entry : ledger)
    {
        EXPECT_TRUE(Autograd::has_backward_rule(entry.op))
            << Ops::name_of(entry.op) << " lost its rule, but " << entry.proof
            << " still claims to prove it";
        EXPECT_EQ(std::count(unproven.begin(), unproven.end(), entry.op), 0)
            << Ops::name_of(entry.op) << " is named by " << entry.proof
            << " and still listed as unproven";
    }

    for (OpID op : unproven)
        EXPECT_TRUE(Autograd::has_backward_rule(op))
            << Ops::name_of(op) << " is listed as unproven but has no rule";

    for (const Ops::OpName& declared : Ops::K_ALL)
    {
        if (!Autograd::has_backward_rule(declared.id)) continue;

        const bool listed = std::any_of(ledger.begin(), ledger.end(),
                                        [&declared](const Proven& entry)
                                        {
                                            return entry.op == declared.id;
                                        })
                         || std::count(unproven.begin(), unproven.end(), declared.id) > 0;

        EXPECT_TRUE(listed)
            << declared.name << " has a backward rule named by neither the ledger nor the "
               "unproven list. Add a FLOAT64 finite-difference case to "
               "test_autograd_numeric.cpp and name it in the ledger - an unverified rule trains "
               "a model slightly wrong and fails nothing.";
    }
}

TEST(Unit_Backward, FanOutEmitsAnAccumulateChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    // x has two contributors, so ACCUMULATE is emitted; see tests/unit/README.md.
    const ValueID left = builder.relu(x);
    const ValueID right = builder.tanh(x);
    const ValueID sum = builder.add(left, right);
    const ValueID loss = builder.nll_loss(sum, targets);

    const std::vector<ValueID> wrt = {x};
    const std::vector<ValueID> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 1u);
    ASSERT_NE(grads[0], K_NO_VALUE);
    EXPECT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 1u);
    EXPECT_EQ(graph.node(graph.value(grads[0]).producer).op, Ops::K_ACCUMULATE.id);
}

TEST(Unit_Backward, ThreeContributorsChainTwoAccumulates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    const ValueID a = builder.relu(x);
    const ValueID b = builder.tanh(x);
    const ValueID c = builder.sigmoid(x);
    const ValueID loss = builder.nll_loss(builder.add(builder.add(a, b), c), targets);

    const std::vector<ValueID> wrt = {x};
    (void)Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    EXPECT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 2u);
}

TEST(Unit_Backward, ADisconnectedParameterGetsASymbolicZeroNotABuffer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueID loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueID> wrt = {orphan, x};
    const std::size_t before = graph.node_count();

    const std::vector<ValueID> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 2u);
    EXPECT_EQ(grads[0], K_NO_VALUE) << "a disconnected parameter has no gradient value at all";
    EXPECT_NE(grads[1], K_NO_VALUE);

    // A zero gradient emits no ZERO node; see tests/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::K_ZERO.id), 0u);
    EXPECT_EQ(graph.node_count(), before + 2);
}

TEST(Unit_Backward, ADeadLossPathEmitsNothingAtAll)
{
    // A graph with no gradient request emits no node; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");
    const ValueID orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueID loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueID> wrt = {orphan};
    const std::size_t before = graph.node_count();

    const std::vector<ValueID> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

    EXPECT_EQ(grads.at(0), K_NO_VALUE);
    EXPECT_EQ(graph.node_count(), before);
}

TEST(Unit_Backward, AnOperationWithNoRuleThrowsAndNamesItself)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    // An identity that no rule covers stands in for a new op; see tests/unit/README.md.
    const ValueID mystery = graph.add_value(Shape{4, 3}, DType::Float32,
                                            Graph::Lifetime::Activation, "mystery");
    graph.add_node(OpID(0x1234'5678u), {x}, {mystery});

    const ValueID loss = builder.nll_loss(mystery, targets);
    const std::vector<ValueID> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())),
                 Error);
}

TEST(Unit_Backward, ANonScalarRootIsRejected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.relu(x);
    const std::vector<ValueID> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, y, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())),
                 Error);
}

/** @brief Tests that a log-softmax a RELU reads gets one gradient node over [dy, its result]. */
TEST(Unit_Backward, LogSoftmaxOutsideTheFusedPairAppendsItsGradientIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID log_probs = builder.log_softmax(x);

    // A RELU consumer keeps the cross-entropy fusion away; see tests/unit/README.md.
    const Appended appended = differentiate_through(graph, builder.relu(log_probs), {x});

    EXPECT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.relu_backward",
        "sushiai.log_softmax_backward",
    }));

    const Graph::Node& through_relu = appended_node(graph, appended, 1);
    const Graph::Node& gradient = appended_node(graph, appended, 2);
    EXPECT_EQ(gradient.inputs, (std::vector<ValueID>{through_relu.outputs.at(0), log_probs}));
    ASSERT_EQ(appended.gradients.size(), 1u);
    EXPECT_EQ(gradient.outputs, (std::vector<ValueID>{appended.gradients[0]}));
    EXPECT_EQ(graph.value(appended.gradients[0]).shape, (Shape{4, 3}));
    EXPECT_EQ(graph.value(appended.gradients[0]).dtype, graph.value(x).dtype);
}

/** @brief Tests that the fused cross-entropy still absorbs its log-softmax. */
TEST(Unit_Backward, TheFusedPairAppendsNoLogSoftmaxGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const Appended appended = differentiate_through(graph, builder.log_softmax(x), {x});

    EXPECT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.exp",
        "sushiai.sub",
        "sushiai.scale",
    }));
    EXPECT_EQ(count_op(graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 0u);
}

/** @brief Tests that a log-softmax over an input adds no gradient node. */
TEST(Unit_Backward, ALogSoftmaxWhoseOperandNeedsNoGradientAddsNoNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID weight = builder.parameter(Shape{4, 3}, "weight");
    const ValueID product = builder.mul(builder.log_softmax(x), weight);
    const Appended appended = differentiate_through(graph, product, {weight});

    ASSERT_EQ(appended.gradients.size(), 1u);
    EXPECT_NE(appended.gradients[0], K_NO_VALUE);
    EXPECT_EQ(count_op(graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 0u);
}

/** @brief Tests that a softmax hands its log-softmax the gradient its exponential produced. */
TEST(Unit_Backward, ASoftmaxDifferentiatesThroughItsExponentialAndItsLogSoftmax)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{2, 3, 4}, "x");
    const ValueID rows = builder.reshape(builder.softmax(x), Shape{6, 4});
    const std::size_t forward_nodes = graph.node_count();
    const Appended appended = differentiate_through(graph, rows, {x});

    ASSERT_EQ(count_op(graph, Ops::K_LOG_SOFTMAX.id), 1u);
    ASSERT_EQ(count_op(graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);

    ValueID log_probs = K_NO_VALUE;
    ValueID through_exp = K_NO_VALUE;
    const Graph::Node* gradient = nullptr;
    for (std::size_t i = 0; i < graph.node_count(); ++i)
    {
        const Graph::Node& node = graph.nodes()[i];
        if (node.op == Ops::K_LOG_SOFTMAX.id)
            log_probs = node.outputs.at(0);
        if (node.op == Ops::K_MUL.id && i >= forward_nodes)
            through_exp = node.outputs.at(0);
        if (node.op == Ops::K_LOG_SOFTMAX_BACKWARD.id)
            gradient = &node;
    }

    ASSERT_NE(gradient, nullptr);
    ASSERT_NE(through_exp, K_NO_VALUE);
    EXPECT_EQ(gradient->inputs, (std::vector<ValueID>{through_exp, log_probs}));
    ASSERT_EQ(appended.gradients.size(), 1u);
    EXPECT_EQ(graph.value(appended.gradients[0]).shape, (Shape{2, 3, 4}));
}

/** @brief Tests that DIV's rule divides dy by the divisor and reuses the forward quotient. */
TEST(Unit_Backward, DivReadsItsDivisorAndItsOwnQuotient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID c = builder.div(a, b);

    const Appended appended = differentiate_through(graph, c, {a, b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.div",
        "sushiai.mul",
        "sushiai.scale",
    }));

    const Graph::Node& quotient = appended_node(graph, appended, 1);
    const Graph::Node& product = appended_node(graph, appended, 2);
    const Graph::Node& negated = appended_node(graph, appended, 3);

    EXPECT_EQ(quotient.inputs, (std::vector<ValueID>{appended.seed, b}));
    EXPECT_EQ(product.inputs, (std::vector<ValueID>{quotient.outputs.at(0), c}));
    EXPECT_EQ(negated.inputs, (std::vector<ValueID>{product.outputs.at(0)}));
    EXPECT_NEAR(negated.attributes.get(0), -1.0, K_ATTRIBUTE_TOLERANCE);

    EXPECT_EQ(appended.gradients.at(0), quotient.outputs.at(0));
    EXPECT_EQ(appended.gradients.at(1), negated.outputs.at(0));
}

/** @brief Tests that DIV's rule emits no divisor gradient when only the dividend needs one. */
TEST(Unit_Backward, DivEmitsOnlyTheQuotientWhenTheDivisorNeedsNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");
    const ValueID c = builder.div(a, b);

    const Appended appended = differentiate_through(graph, c, {a});

    EXPECT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.div",
    }));
}

/** @brief Tests that DIV's rule emits the divisor gradient alone when the dividend needs none. */
TEST(Unit_Backward, DivEmitsTheDivisorGradientAloneWhenTheDividendNeedsNone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID c = builder.div(a, b);

    const Appended appended = differentiate_through(graph, c, {b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.div",
        "sushiai.mul",
        "sushiai.scale",
    }));

    const Graph::Node& quotient = appended_node(graph, appended, 1);
    const Graph::Node& product = appended_node(graph, appended, 2);
    const Graph::Node& negated = appended_node(graph, appended, 3);

    EXPECT_EQ(quotient.inputs, (std::vector<ValueID>{appended.seed, b}));
    EXPECT_EQ(product.inputs, (std::vector<ValueID>{quotient.outputs.at(0), c}));
    EXPECT_EQ(appended.gradients.at(0), negated.outputs.at(0));
}

/** @brief Tests that MINIMUM's rule masks by a <= b, so a tie sends dy to the first operand. */
TEST(Unit_Backward, MinimumRoutesByALessEqualBSoATieGoesToTheFirstOperand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID c = builder.minimum(a, b);

    const Appended appended = differentiate_through(graph, c, {a, b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
        "sushiai.select",
    }));

    const Graph::Node& mask = appended_node(graph, appended, 1);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    const Graph::Node& to_first = appended_node(graph, appended, 3);
    const Graph::Node& to_second = appended_node(graph, appended, 4);

    EXPECT_EQ(mask.inputs, (std::vector<ValueID>{a, b}));
    EXPECT_EQ(to_first.inputs, (std::vector<ValueID>{mask.outputs.at(0), appended.seed, zero}));
    EXPECT_EQ(to_second.inputs, (std::vector<ValueID>{mask.outputs.at(0), zero, appended.seed}));

    EXPECT_EQ(appended.gradients.at(0), to_first.outputs.at(0));
    EXPECT_EQ(appended.gradients.at(1), to_second.outputs.at(0));
}

/** @brief Tests that MAXIMUM's rule masks by b <= a, so a tie sends dy to the first operand. */
TEST(Unit_Backward, MaximumRoutesByBLessEqualASoATieGoesToTheFirstOperand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID c = builder.maximum(a, b);

    const Appended appended = differentiate_through(graph, c, {a, b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
        "sushiai.select",
    }));

    const Graph::Node& mask = appended_node(graph, appended, 1);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    const Graph::Node& to_first = appended_node(graph, appended, 3);
    const Graph::Node& to_second = appended_node(graph, appended, 4);

    EXPECT_EQ(mask.inputs, (std::vector<ValueID>{b, a}));
    EXPECT_EQ(to_first.inputs, (std::vector<ValueID>{mask.outputs.at(0), appended.seed, zero}));
    EXPECT_EQ(to_second.inputs, (std::vector<ValueID>{mask.outputs.at(0), zero, appended.seed}));

    EXPECT_EQ(appended.gradients.at(0), to_first.outputs.at(0));
    EXPECT_EQ(appended.gradients.at(1), to_second.outputs.at(0));
}

/** @brief Tests that MINIMUM's rule emits one SELECT when one operand needs a gradient. */
TEST(Unit_Backward, MinimumEmitsOneSelectWhenOnlyTheSecondOperandNeedsAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID c = builder.minimum(a, b);

    const Appended appended = differentiate_through(graph, c, {b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
    }));

    const ValueID mask = appended_node(graph, appended, 1).outputs.at(0);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    EXPECT_EQ(appended_node(graph, appended, 3).inputs,
              (std::vector<ValueID>{mask, zero, appended.seed}));
}

/** @brief Tests that MAXIMUM's rule emits one SELECT when only the first operand needs one. */
TEST(Unit_Backward, MaximumEmitsOneSelectWhenOnlyTheFirstOperandNeedsAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");
    const ValueID c = builder.maximum(a, b);

    const Appended appended = differentiate_through(graph, c, {a});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
    }));

    const Graph::Node& mask = appended_node(graph, appended, 1);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    const Graph::Node& to_first = appended_node(graph, appended, 3);

    EXPECT_EQ(mask.inputs, (std::vector<ValueID>{b, a}));
    EXPECT_EQ(to_first.inputs, (std::vector<ValueID>{mask.outputs.at(0), appended.seed, zero}));
    EXPECT_EQ(appended.gradients.at(0), to_first.outputs.at(0));
}

/** @brief Tests that ABS's rule multiplies dy by [0 <= x] - [x <= 0], which is 0 at x = 0. */
TEST(Unit_Backward, AbsMultipliesByASignThatIsZeroAtZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.abs(x);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.zero",
        "sushiai.less_equal",
        "sushiai.less_equal",
        "sushiai.sub",
        "sushiai.mul",
    }));

    const ValueID zero = appended_node(graph, appended, 1).outputs.at(0);
    const Graph::Node& not_negative = appended_node(graph, appended, 2);
    const Graph::Node& not_positive = appended_node(graph, appended, 3);
    const Graph::Node& sign = appended_node(graph, appended, 4);
    const Graph::Node& dx = appended_node(graph, appended, 5);

    EXPECT_EQ(graph.value(zero).shape, graph.value(x).shape);
    EXPECT_EQ(not_negative.inputs, (std::vector<ValueID>{zero, x}));
    EXPECT_EQ(not_positive.inputs, (std::vector<ValueID>{x, zero}));
    EXPECT_EQ(sign.inputs,
              (std::vector<ValueID>{not_negative.outputs.at(0), not_positive.outputs.at(0)}));
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{appended.seed, sign.outputs.at(0)}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that ATAN's rule divides dy by x*x plus a CONSTANT one shaped like the input. */
TEST(Unit_Backward, AtanDividesByOnePlusTheSquaredInput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.atan(x);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.mul",
        "sushiai.constant",
        "sushiai.add",
        "sushiai.div",
    }));

    const Graph::Node& squared = appended_node(graph, appended, 1);
    const Graph::Node& one = appended_node(graph, appended, 2);
    const Graph::Node& denominator = appended_node(graph, appended, 3);
    const Graph::Node& dx = appended_node(graph, appended, 4);

    EXPECT_EQ(squared.inputs, (std::vector<ValueID>{x, x}));
    EXPECT_TRUE(one.inputs.empty());
    EXPECT_NEAR(one.attributes.get(0), 1.0, K_ATTRIBUTE_TOLERANCE);
    EXPECT_EQ(graph.value(one.outputs.at(0)).shape, graph.value(x).shape);
    EXPECT_EQ(graph.value(one.outputs.at(0)).dtype, graph.value(x).dtype);
    EXPECT_EQ(denominator.inputs,
              (std::vector<ValueID>{squared.outputs.at(0), one.outputs.at(0)}));
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{appended.seed, denominator.outputs.at(0)}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that SOFTPLUS's rule emits one SOFTPLUS_BACKWARD over dy and the forward input. */
TEST(Unit_Backward, SoftplusBackwardReadsTheForwardInputNotTheOutput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.softplus(x);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.softplus_backward",
    }));

    const Graph::Node& dx = appended_node(graph, appended, 1);
    EXPECT_NE(x, y);
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{appended.seed, x}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that SILU's rule emits one SILU_BACKWARD over dy and the forward input. */
TEST(Unit_Backward, SiLUBackwardReadsTheForwardInputNotTheOutput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.silu(x);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.silu_backward",
    }));

    const Graph::Node& dx = appended_node(graph, appended, 1);
    EXPECT_NE(x, y);
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{appended.seed, x}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that SELECT's rule gives dy to a under the condition and to b elsewhere. */
TEST(Unit_Backward, SelectRoutesItsGradientByItsOwnCondition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID y = builder.select(condition, a, b);

    const Appended appended = differentiate_through(graph, y, {a, b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.zero",
        "sushiai.select",
        "sushiai.select",
    }));

    const ValueID zero = appended_node(graph, appended, 1).outputs.at(0);
    const Graph::Node& to_a = appended_node(graph, appended, 2);
    const Graph::Node& to_b = appended_node(graph, appended, 3);

    EXPECT_EQ(to_a.inputs, (std::vector<ValueID>{condition, appended.seed, zero}));
    EXPECT_EQ(to_b.inputs, (std::vector<ValueID>{condition, zero, appended.seed}));
    EXPECT_EQ(appended.gradients.at(0), to_a.outputs.at(0));
    EXPECT_EQ(appended.gradients.at(1), to_b.outputs.at(0));
}

/** @brief Tests that SELECT's rule emits one SELECT when only the second branch needs one. */
TEST(Unit_Backward, SelectEmitsOneSelectWhenOnlyTheSecondBranchNeedsAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.parameter(Shape{4, 3}, "b");
    const ValueID y = builder.select(condition, a, b);

    const Appended appended = differentiate_through(graph, y, {b});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.zero",
        "sushiai.select",
    }));

    const ValueID zero = appended_node(graph, appended, 1).outputs.at(0);
    const Graph::Node& to_b = appended_node(graph, appended, 2);

    EXPECT_EQ(to_b.inputs, (std::vector<ValueID>{condition, zero, appended.seed}));
    EXPECT_EQ(appended.gradients.at(0), to_b.outputs.at(0));
}

/** @brief Tests that SELECT's rule emits nothing when only its condition needs a gradient. */
TEST(Unit_Backward, SelectGivesItsConditionNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID condition = builder.parameter(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");
    const ValueID y = builder.select(condition, a, b);

    const Appended appended = differentiate_through(graph, y, {condition});

    EXPECT_EQ(op_names(graph, appended.first), (std::vector<std::string>{"sushiai.scale"}));
    EXPECT_EQ(appended.gradients.at(0), K_NO_VALUE);
}

/** @brief Tests that a comparison feeding a SELECT condition gives its operands no gradient. */
TEST(Unit_Backward, AComparisonUnderASelectLeavesItsOperandsWithoutAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID p = builder.parameter(Shape{4, 3}, "p");
    const ValueID q = builder.input(Shape{4, 3}, "q");
    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID y = builder.select(builder.less_equal(p, q), a, q);

    const Appended appended = differentiate_through(graph, y, {p, a});

    EXPECT_EQ(appended.gradients.at(0), K_NO_VALUE);
    EXPECT_NE(appended.gradients.at(1), K_NO_VALUE);
    EXPECT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.zero",
        "sushiai.select",
    }));
}

/** @brief Tests that a mask multiplied into a product receives no gradient node of its own. */
TEST(Unit_Backward, AComparisonMultipliedIntoAProductEmitsNothingForTheMask)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID p = builder.parameter(Shape{4, 3}, "p");
    const ValueID q = builder.input(Shape{4, 3}, "q");
    const ValueID a = builder.parameter(Shape{4, 3}, "a");
    const ValueID mask = builder.less_equal(p, q);
    const ValueID y = builder.mul(mask, a);

    Appended appended;
    ASSERT_NO_THROW(appended = differentiate_through(graph, y, {p, a}));

    EXPECT_FALSE(graph.value(mask).requires_grad);
    EXPECT_EQ(appended.gradients.at(0), K_NO_VALUE);
    ASSERT_NE(appended.gradients.at(1), K_NO_VALUE);

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.mul",
    }));

    const Graph::Node& da = appended_node(graph, appended, 1);

    EXPECT_EQ(da.inputs, (std::vector<ValueID>{appended.seed, mask}));
    EXPECT_EQ(appended.gradients.at(1), da.outputs.at(0));
}

/** @brief Tests that a CONSTANT added to a parameter path receives no gradient node. */
TEST(Unit_Backward, AConstantAddedToAParameterPathEmitsNothingForTheConstant)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID epsilon = builder.constant_like(x, 1e-7);
    const ValueID y = builder.mul(builder.add(x, epsilon), epsilon);

    Appended appended;
    ASSERT_NO_THROW(appended = differentiate_through(graph, y, {x}));

    EXPECT_FALSE(graph.value(epsilon).requires_grad);
    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.mul",
    }));

    const Graph::Node& dx = appended_node(graph, appended, 1);
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{appended.seed, epsilon}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that MAXIMUM of x and a zero CONSTANT, in that order, sends a tie's dy to x. */
TEST(Unit_Backward, MaximumOfXAndAZeroConstantRoutesATieToX)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID floor = builder.constant_like(x, 0.0);
    const ValueID y = builder.maximum(x, floor);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
    }));

    const Graph::Node& mask = appended_node(graph, appended, 1);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    const Graph::Node& dx = appended_node(graph, appended, 3);

    // The mask is [floor <= x], which holds at x = 0, so x takes dy there.
    EXPECT_EQ(mask.inputs, (std::vector<ValueID>{floor, x}));
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{mask.outputs.at(0), appended.seed, zero}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));
}

/** @brief Tests that MAXIMUM of a zero CONSTANT and x, the wide clamp, gives x nothing at a tie. */
TEST(Unit_Backward, MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID floor = builder.constant_like(x, 0.0);
    const ValueID y = builder.maximum(floor, x);

    const Appended appended = differentiate_through(graph, y, {x});

    ASSERT_EQ(op_names(graph, appended.first), (std::vector<std::string>{
        "sushiai.scale",
        "sushiai.less_equal",
        "sushiai.zero",
        "sushiai.select",
    }));

    const Graph::Node& mask = appended_node(graph, appended, 1);
    const ValueID zero = appended_node(graph, appended, 2).outputs.at(0);
    const Graph::Node& dx = appended_node(graph, appended, 3);

    // The mask is [x <= floor], which holds at x = 0, so x takes the zero branch there.
    EXPECT_EQ(mask.inputs, (std::vector<ValueID>{x, floor}));
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{mask.outputs.at(0), zero, appended.seed}));
    EXPECT_EQ(appended.gradients.at(0), dx.outputs.at(0));

    for (std::size_t i = 0; i < graph.node_count(); ++i)
    {
        const OpID op = graph.nodes()[i].op;
        EXPECT_FALSE(Graph::narrows_under_mixed_precision(op))
            << Ops::name_of(op) << " narrows, so the clamp would leave the loss's precision";
    }
}

/** @brief Tests that no rule of the eight forward operations appends a node that narrows. */
TEST(Unit_Backward, NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows)
{
    const std::vector<ForwardCase> cases = {
        {"div", [](const GraphBuilder& g, ValueID a, ValueID b) { return g.div(a, b); }},
        {"minimum", [](const GraphBuilder& g, ValueID a, ValueID b) { return g.minimum(a, b); }},
        {"maximum", [](const GraphBuilder& g, ValueID a, ValueID b) { return g.maximum(a, b); }},
        {"abs", [](const GraphBuilder& g, ValueID a, ValueID) { return g.abs(a); }},
        {"atan", [](const GraphBuilder& g, ValueID a, ValueID) { return g.atan(a); }},
        {"softplus", [](const GraphBuilder& g, ValueID a, ValueID) { return g.softplus(a); }},
        {"silu", [](const GraphBuilder& g, ValueID a, ValueID) { return g.silu(a); }},
        {"select", [](const GraphBuilder& g, ValueID a, ValueID b)
                   {
                       return g.select(g.less_equal(a, b), a, b);
                   }},
    };

    for (const ForwardCase& forward : cases)
    {
        SCOPED_TRACE(std::string(forward.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID a = builder.parameter(Shape{4, 3}, "a");
        const ValueID b = builder.parameter(Shape{4, 3}, "b");

        const Appended appended = differentiate_through(graph, forward.emit(builder, a, b), {a, b});

        ASSERT_GT(graph.node_count(), appended.first + 1) << "the rule appended nothing";
        for (std::size_t i = appended.first; i < graph.node_count(); ++i)
        {
            const OpID op = graph.nodes()[i].op;
            EXPECT_FALSE(Graph::narrows_under_mixed_precision(op))
                << Ops::name_of(op) << " narrows under the mixed-precision policy";
        }
    }
}

/** @brief Tests that a gradient reaching the batch variance, output 3, is refused by the rule. */
TEST(Unit_Backward, AGradientOnTheBatchVarianceOfBatchNormThrows)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2, 2, 1}, "x");
    const ValueID gamma = builder.parameter(Shape{1}, "gamma");
    const ValueID beta = builder.parameter(Shape{1}, "beta");
    const Graph::BatchNormResult bn = builder.batchnorm2d(x, gamma, beta, 1e-5);
    const ValueID loss = builder.scale(bn.batch_variance, 2.0);

    const ValueID wrt[] = {gamma};
    SA_EXPECT_ERROR((void)Autograd::differentiate(graph, loss,
                                                  SushiRuntime::span<const ValueID>(wrt, 1)),
                    "BATCHNORM2D", "saved statistics received a gradient");
}

/** @brief Tests that a gradient through output 0 of a four-output BATCHNORM2D is still emitted. */
TEST(Unit_Backward, BatchNormWithFourOutputsStillDifferentiatesThroughItsFirst)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2, 2, 1}, "x");
    const ValueID gamma = builder.parameter(Shape{1}, "gamma");
    const ValueID beta = builder.parameter(Shape{1}, "beta");
    const Graph::BatchNormResult bn = builder.batchnorm2d(x, gamma, beta, 1e-5);
    const ValueID loss = builder.sum_channels(bn.output);

    const ValueID wrt[] = {gamma, beta};
    const std::vector<ValueID> gradients =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueID>(wrt, 2));

    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_NE(gradients[0], K_NO_VALUE);
    EXPECT_NE(gradients[1], K_NO_VALUE);
    ASSERT_EQ(count_op(graph, Ops::K_BATCHNORM2D_BACKWARD.id), 1u);

    // The backward node reads the saved mean and inverse deviation, never the variance.
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op != Ops::K_BATCHNORM2D_BACKWARD.id)
            continue;

        EXPECT_EQ(node.inputs[3], bn.saved_mean);
        EXPECT_EQ(node.inputs[4], bn.saved_inv_std);
    }
}

/** @brief Tests that differentiation fails at BATCHNORM_FOLD when gamma requires a gradient. */
TEST(Unit_Backward, DifferentiatingThroughBatchNormFoldFails)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");
    const ValueID gamma = builder.parameter(Shape{1}, "gamma");
    const ValueID beta = builder.parameter(Shape{1}, "beta");
    const ValueID mean = builder.buffer(Shape{1}, "running_mean");
    const ValueID variance = builder.buffer(Shape{1}, "running_var");
    const Graph::BatchNormFold fold = builder.batchnorm_fold(gamma, beta, mean, variance, 1e-3);
    const ValueID loss = builder.spatial_mean(builder.channel_affine(x, fold.scale, fold.shift));

    const ValueID wrt[] = {gamma};
    SA_EXPECT_ERROR((void)Autograd::differentiate(graph, loss,
                                                  SushiRuntime::span<const ValueID>(wrt, 1)),
                    "No backward rule", "sushiai.batchnorm_fold");
}

/** @brief Tests that differentiate refuses a buffer in wrt before it appends any node. */
TEST(Unit_Backward, ABufferInWrtIsRefusedAndAppendsNoNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 5}, "targets");
    const ValueID weight = builder.parameter(Shape{5, 3}, "weight");
    const ValueID shift = builder.buffer(Shape{5}, "running_shift");
    const ValueID logits = builder.bias_add(builder.matmul(x, weight, false, true), shift);
    const ValueID loss = builder.nll_loss(builder.log_softmax(logits), targets);
    const std::size_t nodes_before = graph.node_count();

    const ValueID with_buffer[] = {weight, shift};
    SA_EXPECT_ERROR((void)Autograd::differentiate(
                        graph, loss, SushiRuntime::span<const ValueID>(with_buffer, 2)),
                    "differentiate()", "running_shift", "buffer");
    EXPECT_EQ(graph.node_count(), nodes_before);

    // The same graph differentiates once the buffer is left out of the request.
    const ValueID parameters_only[] = {weight};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(parameters_only, 1));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).lifetime, Graph::Lifetime::Gradient);
    EXPECT_EQ(graph.value(shift).lifetime, Graph::Lifetime::Buffer);
}

namespace
{

    /** @brief Holds the factor of the SCALE term in WeightedSum. */
    constexpr double K_TERM_WEIGHT = 2.0;

    /** @brief Holds the factor of the SCALE that ends a loss in the tests that need one. */
    constexpr double K_ROOT_WEIGHT = 0.5;

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

} // namespace

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

namespace
{
    /** @brief Differentiates @p loss with respect to @p wrt, for a one-line assertion. */
    void differentiate_wrt(Graph::Graph& graph, ValueID loss, ValueID wrt)
    {
        const std::vector<ValueID> targets = {wrt};
        (void)Autograd::differentiate(graph, loss,
                                      SushiRuntime::span<const ValueID>(targets.data(), targets.size()));
    }

} // namespace

TEST(Unit_Diagnostics, ANonScalarRootNamesTheValueAndItsShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID y = builder.relu(x);

    SA_EXPECT_ERROR(differentiate_wrt(graph, y, x),
                    "needs a scalar to differentiate", "[4, 3]");
}

/** @brief Tests that a one-element root a differentiable operation produces is seeded. */
TEST(Unit_Diagnostics, AOneElementRootOfADifferentiableOperationIsSeeded)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1, 4}, "x");
    const ValueID w = builder.input(Shape{4, 1}, "w");
    const ValueID scalar = builder.matmul(x, w);
    const std::size_t forward_nodes = graph.node_count();

    const std::vector<ValueID> wrt = {x};
    std::vector<ValueID> gradients;
    ASSERT_NO_THROW(gradients = Autograd::differentiate(
                        graph, scalar, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(graph.node_count(), forward_nodes + 2);
    const Graph::Node& seed = graph.nodes()[forward_nodes];
    const Graph::Node& dx = graph.nodes()[forward_nodes + 1];

    EXPECT_EQ(seed.op, Ops::K_CONSTANT.id);
    EXPECT_EQ(graph.value(seed.outputs.at(0)).shape, graph.value(scalar).shape);
    EXPECT_EQ(dx.op, Ops::K_MATMUL.id);
    EXPECT_EQ(dx.inputs, (std::vector<ValueID>{seed.outputs.at(0), w}));

    ASSERT_EQ(gradients.size(), 1u);
    EXPECT_EQ(gradients[0], dx.outputs.at(0));
    EXPECT_EQ(graph.value(gradients[0]).shape, graph.value(x).shape);
}

/** @brief Tests that a root no parameter reaches is refused with a message naming that cause. */
TEST(Unit_Diagnostics, ARootThatRequiresNoGradientNamesItsProducerAndTheCause)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1, 4}, "x");
    const ValueID a = builder.input(Shape{1, 4}, "a");
    const ValueID w = builder.input(Shape{4, 1}, "w");
    const ValueID unreached = builder.matmul(a, w);
    const ValueID stopped = builder.matmul(builder.stop_gradient(x), w);

    SA_EXPECT_ERROR(differentiate_wrt(graph, unreached, x),
                    "value 3", "sushiai.matmul", "requires no gradient");
    SA_EXPECT_ERROR(differentiate_wrt(graph, stopped, x),
                    "sushiai.matmul", "requires no gradient", "STOP_GRADIENT");
}

/** @brief Tests that a root whose producer has no backward rule is refused naming the producer. */
TEST(Unit_Diagnostics, ARootWhoseProducerHasNoRuleNamesTheProducer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{1}, "x");
    const ValueID factor = builder.input(Shape{1}, "factor");
    const ValueID scaled = builder.dynamic_scale(x, factor);

    SA_EXPECT_ERROR(differentiate_wrt(graph, scaled, x),
                    "sushiai.dynamic_scale", "has no backward rule", "autograd/rules/");
}

TEST(Unit_Diagnostics, ALossNoOperationProducesIsRejected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // A loss with no producer has no operation to name; see tests/unit/README.md.
    const ValueID orphan = graph.add_value(Shape{1}, DType::Float32,
                                           Graph::Lifetime::Activation, "orphan");
    SA_EXPECT_ERROR(differentiate_wrt(graph, orphan, orphan), "nothing produces it", "NLL_LOSS");

    // A leaf does have a producing node, and it is named.
    const ValueID leaf = builder.parameter(Shape{1}, "leaf");
    SA_EXPECT_ERROR(differentiate_wrt(graph, leaf, leaf), "sushiai.leaf", "NLL_LOSS");
}

TEST(Unit_Diagnostics, AnOperationWithNoBackwardRuleNamesItselfAndItsNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{4, 3}, "x");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    const ValueID mystery = graph.add_value(Shape{4, 3}, DType::Float32,
                                            Graph::Lifetime::Activation, "mystery");
    graph.add_node(OpID(0x1234'5678u), {x}, {mystery});

    const ValueID loss = builder.nll_loss(mystery, targets);

    // A foreign identity prints as sushiai.unknown; see tests/unit/README.md.
    SA_EXPECT_ERROR(differentiate_wrt(graph, loss, x),
                    "No backward rule for", "sushiai.unknown", "node 2");
}

TEST(Unit_Diagnostics, AGradientContributedToTheWrongShapeNamesBothShapes)
{
    // A SUB with a mis-shaped result provokes the error; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{2, 6}, "x");
    const ValueID other = builder.input(Shape{4, 3}, "other");
    const ValueID targets = builder.input(Shape{4, 3}, "targets");

    const ValueID difference = graph.add_value(Shape{4, 3}, DType::Float32,
                                               Graph::Lifetime::Activation, "bad_sub");
    graph.add_node(Ops::K_SUB.id, {x, other}, {difference});

    const ValueID loss = builder.nll_loss(difference, targets);

    SA_EXPECT_ERROR(differentiate_wrt(graph, loss, x),
                    "gradient of shape [4, 3]", "shape [2, 6]");
}

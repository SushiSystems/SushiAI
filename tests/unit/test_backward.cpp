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
 * @brief Tests the graph that differentiation emits and the coverage of the backward rule table.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

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
}

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

TEST(Unit_Backward, TwoLayerMLPProducesTheExpectedBackwardTopology)
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

    // Cross-entropy differentiates as one fused rule; see tests/unit/README.md.
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

    // One contributor per value emits no ACCUMULATE; see tests/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::K_ACCUMULATE.id), 0u);

    ASSERT_EQ(grads.size(), 4u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], K_NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
        EXPECT_EQ(graph.value(grads[i]).lifetime, Graph::Lifetime::Gradient);
    }
}

TEST(Unit_Backward, TheInputIsNotDifferentiatedWhenItDoesNotRequireOne)
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

    // The input gradient's MATMUL is skipped; see tests/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::K_MATMUL.id), 2u);
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

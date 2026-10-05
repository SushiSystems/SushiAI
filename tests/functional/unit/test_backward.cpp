/****************************************************************/
/* test_backward.cpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_backward.cpp
 * @brief Tests the graph that differentiation emits and the coverage of the backward rule table.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::NO_VALUE;
using Graph::ValueId;

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
}

TEST(Unit_Backward, EveryImplementedOperationHasARule)
{
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::MATMUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::BIAS_ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SUB.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::MUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SCALE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::EXP.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::RELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::GELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SIGMOID.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::TANH.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SUM_ROWS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::ACCUMULATE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::NLL_LOSS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::MAXPOOL2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::AVGPOOL2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::UPSAMPLE2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SPATIAL_MEAN.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::CONCAT_CHANNELS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SLICE_CHANNELS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::BATCHNORM2D.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::CHANNEL_AFFINE.id));

    // Running statistics are a buffer, not a gradient path; see tests/functional/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::BATCHNORM_UPDATE_STATS.id));

    // LEAF and ZERO have no operand to hand a gradient to; see tests/functional/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::LEAF.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::ZERO.id));

    // Casts are inserted after differentiation; see tests/functional/unit/README.md.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::CAST.id));
}

TEST(Unit_Backward, EveryRuleInTheTableIsNamedByANumericTest)
{
    // Every rule names the numeric test that proves it; see tests/functional/unit/README.md.
    struct Proven
    {
        OpID op;
        std::string_view proof;
    };

    const std::vector<Proven> ledger = {
        {Ops::MATMUL.id,     "Integration_Autograd.MatmulWithoutTransposes"},
        {Ops::MATMUL.id,     "Integration_Autograd.MatmulWithATransposedRightOperand"},
        {Ops::MATMUL.id,     "Integration_Autograd.MatmulWithATransposedLeftOperand"},
        {Ops::MATMUL.id,     "Integration_Autograd.MatmulWithBothOperandsTransposed"},
        {Ops::MATMUL.id,     "Integration_Autograd.MatmulGradientOfTheRightOperand"},
        {Ops::BIAS_ADD.id,   "Integration_Autograd.BiasAddPassesTheMatrixGradientThrough"},
        {Ops::BIAS_ADD.id,   "Integration_Autograd.BiasAddSumsOverTheBroadcastAxis"},
        {Ops::ADD.id,        "Integration_Autograd.ElementwiseAddAndSubtract"},
        {Ops::SUB.id,        "Integration_Autograd.ElementwiseSubtractFromTheRight"},
        {Ops::MUL.id,        "Integration_Autograd.ElementwiseMultiply"},
        {Ops::MUL.id,        "Integration_Autograd.ElementwiseMultiplyFromTheRight"},
        {Ops::SCALE.id,      "Integration_Autograd.Scale"},
        {Ops::EXP.id,        "Integration_Autograd.Exp"},
        {Ops::RELU.id,       "Integration_Autograd.ReluAwayFromItsKink"},
        {Ops::GELU.id,       "Integration_Autograd.Gelu"},
        {Ops::SIGMOID.id,    "Integration_Autograd.SigmoidReadsItsForwardOutput"},
        {Ops::TANH.id,       "Integration_Autograd.TanhReadsItsForwardOutput"},
        {Ops::LEAKY_RELU.id, "Integration_Autograd.LeakyReluHasBothOfItsSlopes"},
        {Ops::SUM_ROWS.id,   "Integration_Autograd.SumRowsBroadcastsItsGradientBack"},
        {Ops::ACCUMULATE.id, "Integration_Autograd.AccumulateInTheForwardGraph"},
        {Ops::NLL_LOSS.id,   "Integration_Autograd.FusedCrossEntropyAgainstOneHotTargets"},

        // LOG_SOFTMAX's rule only raises its diagnostic; see tests/functional/unit/README.md.
        {Ops::LOG_SOFTMAX.id, "Unit_Diagnostics.LogSoftmaxOutsideTheFusedPairExplainsTheRestriction"},

        {Ops::RESHAPE.id,        "Integration_Autograd.ReshapeMovesItsGradientBackUnchanged"},
        {Ops::LAYOUT_CONVERT.id, "Integration_Autograd.LayoutConvertsGradientIsThePermutedGradient"},

        {Ops::IM2COL.id, "Integration_Autograd.AWholeConvolutionDifferentiatesThroughIm2colAndMatmul"},
        {Ops::COL2IM.id, "Integration_Autograd.Col2imsGradientGathersEveryWindowThatWroteAPixel"},

        {Ops::MAXPOOL2D.id, "Integration_Autograd.MaxPoolRoutesEachGradientToTheElementThatWon"},
        {Ops::AVGPOOL2D.id, "Integration_Autograd.AvgPoolSpreadsItsGradientOverTheWholeWindow"},
        {Ops::UPSAMPLE2D.id, "Integration_Autograd.UpsampleSumsEachSourcePixelsReplicas"},
        {Ops::SPATIAL_MEAN.id, "Integration_Autograd.SpatialMeanBroadcastsScaledByOneOverHW"},
        {Ops::CONCAT_CHANNELS.id, "Integration_Autograd.ConcatGivesEachOperandItsOwnChannelRange"},
        {Ops::SLICE_CHANNELS.id, "Integration_Autograd.SliceScattersItsGradientBackIntoZeros"},

        {Ops::BATCHNORM2D.id, "Integration_Autograd.BatchNormsGradientAccountsForTheStatisticsItComputed"},
        {Ops::BATCHNORM2D.id, "Integration_Autograd.BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput"},
        {Ops::BATCHNORM2D.id, "Integration_Autograd.BatchNormsShiftGradientIsThePlainSumOfDy"},

        {Ops::CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffineDifferentiatesAtItsInput"},
        {Ops::CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffinesScaleGradientSumsOverNHAndW"},
        {Ops::CHANNEL_AFFINE.id, "Integration_Autograd.ChannelAffinesShiftGradientSumsOverNHAndW"},
    };

    for (const Proven& entry : ledger)
        EXPECT_TRUE(Autograd::has_backward_rule(entry.op))
            << Ops::name_of(entry.op) << " lost its rule, but " << entry.proof
            << " still claims to prove it";

    for (const Ops::OpName& declared : Ops::ALL)
    {
        if (!Autograd::has_backward_rule(declared.id)) continue;

        const bool listed = std::any_of(ledger.begin(), ledger.end(),
                                        [&declared](const Proven& entry)
                                        {
                                            return entry.op == declared.id;
                                        });

        EXPECT_TRUE(listed)
            << declared.name << " gained a backward rule with no entry in this ledger. Add a "
               "FLOAT64 finite-difference case to test_autograd_numeric.cpp and name it here - "
               "an unverified rule trains a model slightly wrong and fails nothing.";
    }
}

TEST(Unit_Backward, TwoLayerMlpProducesTheExpectedBackwardTopology)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{8, 6}, "x");
    const ValueId targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    const ValueId logits = model.forward(builder, x);
    const ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::size_t forward_nodes = graph.node_count();

    std::vector<ValueId> wrt;
    NN::for_each_parameter(model, [&wrt](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id());
    });
    ASSERT_EQ(wrt.size(), 4u);

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    // Cross-entropy differentiates as one fused rule; see tests/functional/unit/README.md.
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
    EXPECT_EQ(count_op(graph, Ops::LOG_SOFTMAX.id), 1u);

    // One contributor per value emits no ACCUMULATE; see tests/functional/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 0u);

    ASSERT_EQ(grads.size(), 4u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
        EXPECT_EQ(graph.value(grads[i]).lifetime, Graph::Lifetime::GRADIENT);
    }
}

TEST(Unit_Backward, TheInputIsNotDifferentiatedWhenItDoesNotRequireOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{8, 6}, "x");
    const ValueId targets = builder.input(Shape{8, 3}, "targets");

    NN::Linear layer{6, 3};
    const ValueId logits = layer.forward(builder, x);
    const ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::vector<ValueId> wrt = {layer.weight.id()};
    (void)Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    // The input gradient's MATMUL is skipped; see tests/functional/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 2u);
}

TEST(Unit_Backward, FanOutEmitsAnAccumulateChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // x has two contributors, so ACCUMULATE is emitted; see tests/functional/unit/README.md.
    const ValueId left = builder.relu(x);
    const ValueId right = builder.tanh(x);
    const ValueId sum = builder.add(left, right);
    const ValueId loss = builder.nll_loss(sum, targets);

    const std::vector<ValueId> wrt = {x};
    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 1u);
    ASSERT_NE(grads[0], NO_VALUE);
    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 1u);
    EXPECT_EQ(graph.node(graph.value(grads[0]).producer).op, Ops::ACCUMULATE.id);
}

TEST(Unit_Backward, ThreeContributorsChainTwoAccumulates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    const ValueId a = builder.relu(x);
    const ValueId b = builder.tanh(x);
    const ValueId c = builder.sigmoid(x);
    const ValueId loss = builder.nll_loss(builder.add(builder.add(a, b), c), targets);

    const std::vector<ValueId> wrt = {x};
    (void)Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 2u);
}

TEST(Unit_Backward, ADisconnectedParameterGetsASymbolicZeroNotABuffer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");
    const ValueId orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueId loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueId> wrt = {orphan, x};
    const std::size_t before = graph.node_count();

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 2u);
    EXPECT_EQ(grads[0], NO_VALUE) << "a disconnected parameter has no gradient value at all";
    EXPECT_NE(grads[1], NO_VALUE);

    // A zero gradient emits no ZERO node; see tests/functional/unit/README.md.
    EXPECT_EQ(count_op(graph, Ops::ZERO.id), 0u);
    EXPECT_EQ(graph.node_count(), before + 2);
}

TEST(Unit_Backward, ADeadLossPathEmitsNothingAtAll)
{
    // A graph with no gradient request emits no node; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");
    const ValueId orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueId loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueId> wrt = {orphan};
    const std::size_t before = graph.node_count();

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    EXPECT_EQ(grads.at(0), NO_VALUE);
    EXPECT_EQ(graph.node_count(), before);
}

TEST(Unit_Backward, AnOperationWithNoRuleThrowsAndNamesItself)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // An identity that no rule covers stands in for a new op; see tests/functional/unit/README.md.
    const ValueId mystery = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                            Graph::Lifetime::ACTIVATION, "mystery");
    graph.add_node(OpID(0x1234'5678u), {x}, {mystery});

    const ValueId loss = builder.nll_loss(mystery, targets);
    const std::vector<ValueId> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}

TEST(Unit_Backward, ANonScalarRootIsRejected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId y = builder.relu(x);
    const std::vector<ValueId> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, y, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}

TEST(Unit_Backward, LogSoftmaxOutsideTheFusedPairSaysSoClearly)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // A RELU consumer keeps the cross-entropy fusion away; see tests/functional/unit/README.md.
    const ValueId detoured = builder.relu(builder.log_softmax(x));
    const ValueId loss = builder.nll_loss(detoured, targets);

    const std::vector<ValueId> wrt = {x};
    EXPECT_THROW((void)Autograd::differentiate(
                     graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}

/**************************************************************************/
/* test_fusion.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the static fusion pass is a pure function of the IR, exactly as
// the memory planner is, so every case here is a synthetic graph and a table of
// expectations with no engine anywhere.
//
// What is being pinned is the set of refusals, not the matches. A pass that
// fuses too little is slow; a pass that fuses too much is *silently wrong* -
// the graph still runs, the loss still falls, and the only symptom is a model
// that converges somewhere slightly different. So the cases below are mostly
// graphs the pass must decline, one per rule:
//
//   - an intermediate with a second consumer (fusing it means losing it),
//   - an intermediate the caller reads by id (same, from outside),
//   - a spilled pre-activation read by a node that runs before the fused one.
//
// The activation whose gradient differentiates at its input used to be a fourth
// entry on that list. It is not any more: GEMM_BIAS_GELU folds it by spilling
// the pre-activation to a second output, so the refusal became a *condition* -
// an INPUT-gradient activation may be folded only by an entry that writes the
// tensor its gradient reads, and only if it writes the right one. That has a
// compile-time half in fusion.cpp: an entry breaking either half fails a
// static_assert. It cannot be tested from here, because a test that fails to
// compile is not a test, so what is checked here is the pair of properties the
// assertion reads - and, above all, that the fold really does leave the
// pre-activation written and readable by the backward node.

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

// =========================================================================
// 1. The catalog is well formed
// =========================================================================

TEST_F(Unit_Fusion, EveryFormRoundTripsThroughItsOperationIdentity)
{
    // fused_form_op and fused_form_of are what the lowering dispatches on, so a
    // form that does not round-trip would lower as "no recorder for this
    // operation" - loud, but only at run time and only on a graph that happens
    // to contain it.
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
    // The property fusion.cpp's static_assert reads. If this ever flips for an
    // activation that has a catalog entry, the assertion is the thing that
    // catches it - but only if this stays true of the table it consults, which
    // is what is pinned here.
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::RELU.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::SIGMOID.id));
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::TANH.id));

    // SushiBLAS's gelu_backward takes the pre-activation, so an entry that
    // wrote only its result would differentiate at a tensor that never reached
    // memory. That has not changed and must not: it is the premise the
    // static_assert reasons from.
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::GELU.id));

    // What changed is the conclusion drawn from it. GELU is folded, by the one
    // entry that spills - so "survives fusion" and "the catalog folds it" are
    // now two questions with two answers, and a caller reading the first as the
    // second gets the wrong one for exactly one activation.
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

// =========================================================================
// 2. The matches
// =========================================================================

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

    // The fused node reads A, B and the bias, and still writes the value the
    // caller named - which is the whole reason ids are preserved.
    ASSERT_EQ(report.fused().size(), 1u);
    EXPECT_EQ(report.fused()[0].result, layer.biased);
    ASSERT_NE(graph.value(layer.biased).producer, NO_NODE);

    const Graph::Node& fused = graph.node(graph.value(layer.biased).producer);
    ASSERT_EQ(fused.inputs.size(), 3u);
    EXPECT_EQ(fused.inputs[0], layer.x);
    EXPECT_EQ(fused.inputs[1], layer.weight);
    EXPECT_EQ(fused.inputs[2], layer.bias);

    // The transposition flags have to survive: the fused recorder hands them
    // straight to execute_gemm, and losing one is a wrong answer of exactly the
    // right shape.
    EXPECT_FALSE(fused.attributes.flag(0));
    EXPECT_TRUE(fused.attributes.flag(1));

    // The intermediate is nobody's output now.
    EXPECT_EQ(graph.value(layer.product).producer, NO_NODE);
}

TEST_F(Unit_Fusion, TheLongestMatchWinsOverTheShorterOneInsideIt)
{
    // Both gemm+bias and gemm+bias+relu match here. Rooted at the bias_add the
    // shorter one would win and the relu would be left outside for ever, which
    // is why the matcher walks roots in reverse.
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
    // (exp(a) - b) and (b - exp(a)) are different functions with the same
    // shapes, so nothing downstream could tell them apart. The link position in
    // the catalog entry is what refuses, and this is the test that it is
    // actually consulted.
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

// =========================================================================
// 3. The refusals - the half that matters
// =========================================================================

TEST_F(Unit_Fusion, AnIntermediateWithASecondConsumerIsNotFusedAway)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Layer layer = trace_linear(builder);

    // A second reader of the MATMUL's result. Fusing would mean recomputing it
    // for this one, or losing it.
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

    // `biased` is the logits of a real classifier: the training loop reads it
    // after every replay to score accuracy, and a fused form gives it no
    // buffer to read.
    const std::vector<ValueId> observed{layer.biased, activated};
    FusionOptions options;
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const FusionReport report = Graph::fuse(graph, options);

    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS_RELU), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_IS_OBSERVED));

    // It still falls back to the shorter form rooted at the bias_add, whose own
    // intermediate is free.
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
}

TEST_F(Unit_Fusion, AParameterIsProtectedWithoutBeingListed)
{
    // A persistent value's buffer outlives the step whatever the graph says, so
    // the caller must not have to remember to list it. Here the "intermediate"
    // is a parameter - an odd graph, but exactly the shape a caller would get
    // wrong.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{ROWS, OUT_FEATURES}, "a");
    const ValueId b = builder.input(Shape{ROWS, OUT_FEATURES}, "b");
    const ValueId exponent = builder.exp(a);

    // Force the exp's result into a persistent class, as a checkpointed
    // activation would be.
    graph.value(exponent).lifetime = Graph::Lifetime::OPTIMIZER_STATE;

    const ValueId difference = builder.sub(exponent, b);
    (void)builder.scale(difference, 0.5);

    const FusionReport report = Graph::fuse(graph, FusionOptions{});

    EXPECT_EQ(report.count(FusedForm::EXP_SUB_SCALE), 0u);
    EXPECT_TRUE(declined_for(report, FusionDecline::VALUE_IS_OBSERVED));
}

TEST_F(Unit_Fusion, AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills)
{
    // The trap this whole design is arranged around, and the shape of the graph
    // in which it is sprung: the differentiation pass has already emitted
    // gelu_backward reading the pre-activation, so that value has two
    // consumers. Every other entry would stop there - rule 1 - and the reason
    // this one does not is the reason it is allowed to exist: the fused node
    // *keeps writing* that value, as a second output, under the same id.
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

    // The load-bearing assertion, and the one a plain GEMM_BIAS_GELU with no
    // spill would fail: the pre-activation is still written, by the fused node,
    // as its second output - and it is the *same* value, so gelu_backward's
    // operand needed no rewriting and the backward rule needed no change.
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

    // Only the MATMUL's result was dropped. Reporting the spilled value as
    // eliminated would be a report that reads like the silently wrong version
    // of this pass.
    EXPECT_EQ(report.fused()[0].eliminated, std::vector<ValueId>{layer.product});
    EXPECT_EQ(graph.value(layer.product).producer, NO_NODE);

    // And gelu_backward still reads the pre-activation, not the result.
    const Graph::Node& backward = graph.node(graph.value(gradient).producer);
    ASSERT_EQ(backward.inputs.size(), 2u);
    EXPECT_EQ(backward.inputs[1], layer.biased);
}

TEST_F(Unit_Fusion, ASpilledPreActivationReadBeforeTheFusedNodeIsRefused)
{
    // The one new way this entry could break the graph, and it is not a wrong
    // number - it is a wrong *order*. The fused node is emitted at the GELU's
    // position, so a reader of the pre-activation sitting between the bias_add
    // and the GELU would end up reading a value written after it, and every
    // pass below assumes that cannot happen.
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

    // It still falls back to the shorter form, exactly as any other refusal of
    // the three-node entry does.
    EXPECT_EQ(report.count(FusedForm::GEMM_BIAS), 1u);
    EXPECT_EQ(count_op(graph, Ops::GELU.id), 1u);

    // And the rewritten graph really is still topological, which is the
    // property the refusal exists to protect.
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
    // The mirror of VALUE_IS_OBSERVED. That rule exists because a fused form
    // gives an interior value no buffer to read; this entry gives it one, so
    // naming the pre-activation as observed must not refuse the fold.
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
    // A spill licenses one interior value and no others. The MATMUL's result is
    // not spilled, so a second reader of it refuses the whole chain exactly as
    // it does for every other entry - and the fallback is nothing, because
    // gemm+bias is blocked by the same value.
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

    // And the GELU that stayed outside is not also reported as unfoldable.
    // "Its gradient differentiates at its input" and "no catalog entry composes
    // it" are both false here and both would send a reader to change the wrong
    // thing; the DISABLED line above is the whole and only reason.
    EXPECT_FALSE(declined_for(report, FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION));
    EXPECT_FALSE(declined_for(report, FusionDecline::NO_CATALOG_ENTRY));
}

TEST_F(Unit_Fusion, ARecoverableActivationNextToAFusedGemmIsFoldedNotReported)
{
    // The mirror of the case above, and the reason ReLU's backward rule reads
    // the output: with the gradient reading the activation's own result, the
    // pre-activation has one consumer and folds with no special case anywhere.
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

// =========================================================================
// 4. The pass keeps the graph usable
// =========================================================================

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
    // Everything a caller holds across the pass - StepSpec, a Parameter's own
    // id, a gradient - depends on this. A pass that compacted ids would need a
    // remap reaching into the model's layer objects, and forgetting one would
    // bind the optimizer to the wrong tensor.
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
    // Everything downstream - the memory planner's reachability closure, the
    // lowering's single forward walk - rests on it, and the fused node is
    // emitted at the *root's* position while reading operands from the chain's
    // first node, which is exactly where a rewrite gets this wrong.
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

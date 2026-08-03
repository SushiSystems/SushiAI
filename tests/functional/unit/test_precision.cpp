/**************************************************************************/
/* test_precision.cpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the precision policy is a table of judgements and the cast pass
// is a pure function of the IR, exactly as the fusion pass and the memory
// planner are.
//
// What is being pinned here is *placement*, not survival. A mixed-precision
// graph that runs proves almost nothing: a missing cast throws in SushiBLAS,
// so "it ran" only rules out the loud half of the failure space. The quiet
// half - a cast in the wrong place - leaves a graph that trains and merely
// learns worse, so every case below is either
//
//   - the whole-graph invariant (every node's operands agree with the type it
//     computes in), asserted by walking the graph rather than by spot checks,
//   - a control (the same graph under the fp32 policy must get *zero* casts,
//     computed rather than short-circuited), or
//   - one of the two places the master-weight arrangement can silently break:
//     a parameter retyped in place, or a gradient that reaches its fp32 buffer
//     without being widened.

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
using Graph::NO_NODE;
using Graph::NO_VALUE;
using Graph::NodeId;
using Graph::UniformPrecisionPolicy;
using Graph::ValueId;

namespace
{
    constexpr int64_t ROWS = 4;
    constexpr int64_t IN_FEATURES = 3;
    constexpr int64_t HIDDEN = 5;
    constexpr int64_t CLASSES = 2;

    /**
     * @struct Traced
     * @brief A one-hidden-layer classifier, differentiated, with the ids named.
     *
     * Built by hand rather than through NN::Sequential so the test can name
     * every value it wants to assert about, and so the graph is the smallest
     * one that still contains all four interesting shapes: a parameter read by
     * both passes, a reduction, a loss, and a gradient buffer.
     */
    struct Traced
    {
        Graph::Graph graph;
        ValueId x = NO_VALUE;
        ValueId targets = NO_VALUE;
        ValueId w1 = NO_VALUE;
        ValueId b1 = NO_VALUE;
        ValueId w2 = NO_VALUE;
        ValueId b2 = NO_VALUE;
        ValueId logits = NO_VALUE;
        ValueId loss = NO_VALUE;
        std::vector<ValueId> gradients;
    };

    /** @brief x -> linear -> relu -> linear -> log_softmax -> nll, then backward. */
    Traced trace(bool differentiate = true)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        traced.x = builder.input(Shape{ROWS, IN_FEATURES}, "x");
        traced.targets = builder.input(Shape{ROWS, CLASSES}, "targets");
        traced.w1 = builder.parameter(Shape{HIDDEN, IN_FEATURES}, "w1");
        traced.b1 = builder.parameter(Shape{HIDDEN}, "b1");
        traced.w2 = builder.parameter(Shape{CLASSES, HIDDEN}, "w2");
        traced.b2 = builder.parameter(Shape{CLASSES}, "b2");

        const ValueId h = builder.relu(
            builder.bias_add(builder.matmul(traced.x, traced.w1, false, true), traced.b1));

        traced.logits = builder.bias_add(builder.matmul(h, traced.w2, false, true), traced.b2);
        traced.loss = builder.nll_loss(builder.log_softmax(traced.logits), traced.targets);

        if (!differentiate) return traced;

        const ValueId wrt[] = {traced.w1, traced.b1, traced.w2, traced.b2};
        traced.gradients = Autograd::differentiate(
            traced.graph, traced.loss, SushiRuntime::span<const ValueId>(wrt, 4));

        return traced;
    }

    /** @brief The four values a training loop reads by id. */
    std::vector<ValueId> observed_of(const Traced& traced)
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

// =========================================================================
// 1. The policy itself
// =========================================================================

TEST(Unit_Precision, TheUniformPolicyAnswersTheGraphsOwnTypeToEverything)
{
    const UniformPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::ALL)
        for (DType dtype : {DType::HALF, DType::FLOAT32, DType::FLOAT64})
            EXPECT_EQ(policy.choose(Graph::PrecisionQuery{entry.id, dtype}).dtype, dtype)
                << entry.name << " was moved by the policy that is supposed to move nothing";

    EXPECT_EQ(policy.name(), "fp32");
}

TEST(Unit_Precision, OnlyTheProductAndWhatAGemmFoldsIntoItNarrow)
{
    // The rule set, stated as a table so that adding an operation to the
    // narrowing set is a visible edit here and not a silent behaviour change.
    // Everything absent from the true list stays in the graph's own precision,
    // which is the safe answer for an operation nobody has thought about yet.
    const OpID narrows[] = {
        Ops::MATMUL.id,
        Ops::GEMM_BIAS.id, Ops::GEMM_BIAS_RELU.id,
        Ops::GEMM_BIAS_SIGMOID.id, Ops::GEMM_BIAS_TANH.id,
        Ops::BIAS_ADD.id,
        Ops::RELU.id, Ops::SIGMOID.id, Ops::TANH.id,
        Ops::RELU_BACKWARD.id, Ops::SIGMOID_BACKWARD.id, Ops::TANH_BACKWARD.id,
    };

    for (OpID op : narrows)
        EXPECT_TRUE(Graph::narrows_under_mixed_precision(op))
            << Ops::name_of(op) << " should compute low";

    // The reductions and anything that sums many terms. fp16 carries ~11 bits
    // of significand and SushiBLAS's own reduction header documents a 2048
    // ceiling for HALF, so these are the ones that go quietly wrong.
    const OpID stays[] = {
        Ops::SUM_ROWS.id, Ops::LOG_SOFTMAX.id, Ops::NLL_LOSS.id, Ops::ACCUMULATE.id,
        Ops::EXP.id, Ops::SUB.id, Ops::SCALE.id, Ops::EXP_SUB_SCALE.id,
        // DYNAMIC_SCALE carries the gradient loss scale, which is the one
        // number in the graph that exists to keep values *out* of fp16's
        // subnormals. Narrowing the node that applies it would overflow the
        // very quantity it is protecting - see optim/loss_scaler.hpp.
        Ops::DYNAMIC_SCALE.id,
        Ops::ADD.id, Ops::MUL.id, Ops::ZERO.id,
        Ops::GELU.id, Ops::GELU_BACKWARD.id,
        Ops::LEAF.id, Ops::CAST.id,
        Ops::ADAMW_UPDATE.id, Ops::SGD_UPDATE.id,
    };

    for (OpID op : stays)
        EXPECT_FALSE(Graph::narrows_under_mixed_precision(op))
            << Ops::name_of(op) << " should stay in the graph's own precision";

    // Every identity is accounted for by one of the two lists above, so a new
    // operation cannot join the enum without someone deciding which it is.
    EXPECT_EQ(sizeof(narrows) / sizeof(narrows[0]) + sizeof(stays) / sizeof(stays[0]),
              Ops::ALL.size());
}

TEST(Unit_Precision, GeluIsTheOneActivationThatStaysWide)
{
    // Two reasons, and the test pins the one that is a property of the code
    // rather than of the arithmetic: no catalog entry folds GELU, because its
    // gradient differentiates at its input (Ops::Activation). The other reason
    // is that SushiBLAS's tanh approximation computes x*x*x, which overflows
    // fp16 at |x| about 40 - true, and not something a unit test can assert
    // without a device.
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::GELU.id));
    EXPECT_FALSE(Graph::narrows_under_mixed_precision(Ops::GELU.id));

    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::RELU.id));
    EXPECT_TRUE(Graph::narrows_under_mixed_precision(Ops::RELU.id));
}

TEST(Unit_Precision, ChoosingIsIdempotentForEveryOperation)
{
    // The contract graph/precision.hpp states, and the one that makes the
    // invariant checkable after the fact instead of only while the pass runs.
    // Without it, find_dtype_mismatches would report a graph the pass had just
    // finished making correct.
    const MixedPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::ALL)
        for (DType dtype : {DType::HALF, DType::FLOAT32, DType::FLOAT64})
        {
            const DType once = policy.choose(Graph::PrecisionQuery{entry.id, dtype}).dtype;
            const DType twice = policy.choose(Graph::PrecisionQuery{entry.id, once}).dtype;
            EXPECT_EQ(once, twice) << entry.name << " from " << dtype_name(dtype);
        }
}

TEST(Unit_Precision, AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType)
{
    const MixedPrecisionPolicy policy;

    // An fp64 graph keeps fp64 masters and still gets fp16 products; an fp16
    // graph is already at the floor and is left alone rather than "narrowed"
    // to something wider.
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::MATMUL.id, DType::FLOAT64}).dtype,
              DType::HALF);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::SUM_ROWS.id, DType::FLOAT64}).dtype,
              DType::FLOAT64);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::MATMUL.id, DType::HALF}).dtype,
              DType::HALF);
    EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::SUM_ROWS.id, DType::HALF}).dtype,
              DType::HALF);

    // A policy narrowing to fp32 is a legal thing to ask for, and is what an
    // fp64 model would use.
    const MixedPrecisionPolicy to_single(DType::FLOAT32);
    EXPECT_EQ(to_single.choose(Graph::PrecisionQuery{Ops::MATMUL.id, DType::FLOAT64}).dtype,
              DType::FLOAT32);
    EXPECT_EQ(to_single.choose(Graph::PrecisionQuery{Ops::MATMUL.id, DType::FLOAT32}).dtype,
              DType::FLOAT32);
}

TEST(Unit_Precision, ANonFloatLowTypeIsRefusedAtConstruction)
{
    EXPECT_THROW(MixedPrecisionPolicy{DType::INT32}, Error);
    EXPECT_THROW(MixedPrecisionPolicy{DType::COMPLEX32}, Error);
    EXPECT_NO_THROW(MixedPrecisionPolicy{DType::HALF});
}

TEST(Unit_Precision, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    const MixedPrecisionPolicy policy;

    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_FALSE(policy.choose(Graph::PrecisionQuery{entry.id, DType::FLOAT32}).reason.empty())
            << entry.name << " was decided without a reason";
}

// =========================================================================
// 2. The control: the fp32 policy changes nothing
// =========================================================================

TEST(Unit_CastInsertion, TheUniformPolicyInsertsNoCasts)
{
    // The control the mixed cases are read against. If this ever shows casts,
    // the pass is inserting them unconditionally and every "the mixed policy
    // inserted N casts" assertion below is measuring nothing.
    Traced traced = trace();

    const std::size_t nodes_before = traced.graph.node_count();
    const std::size_t values_before = traced.graph.value_count();

    const UniformPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(report.retyped_values(), 0u);
    EXPECT_EQ(count_op(traced.graph, Ops::CAST.id), 0u);
    EXPECT_EQ(traced.graph.node_count(), nodes_before);
    EXPECT_EQ(traced.graph.value_count(), values_before);

    for (ValueId id = 0; id < static_cast<ValueId>(values_before); ++id)
        EXPECT_EQ(traced.graph.value(id).dtype, DType::FLOAT32);
}

TEST(Unit_CastInsertion, NoPolicyMeansTheUniformPolicy)
{
    Traced traced = trace();
    const std::size_t nodes_before = traced.graph.node_count();

    const CastReport report = Graph::insert_casts(traced.graph);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(traced.graph.node_count(), nodes_before);
}

// =========================================================================
// 3. The invariant
// =========================================================================

TEST(Unit_CastInsertion, EveryNodesOperandsAgreeWithTheTypeItComputesIn)
{
    // The assertion worth more than a dozen spot checks: it walks the whole
    // graph and states the rule, so it keeps holding as operations are added.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);
    EXPECT_GT(report.count(), 0u) << "a mixed policy that inserted nothing is not mixed";

    const std::vector<DTypeMismatch> mismatches =
        Graph::find_dtype_mismatches(traced.graph, policy);

    EXPECT_TRUE(mismatches.empty()) << describe_all(mismatches);
}

TEST(Unit_CastInsertion, TheInvariantSurvivesTheFusionPassThatRunsAfterIt)
{
    // Fusion rewrites nodes and re-pairs operands, so it is exactly where a
    // correctly cast graph could stop being one - and where a master weight
    // could get swallowed. graph/cast_insertion.hpp argues the ordering; this
    // is the assertion behind the argument.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    const std::vector<ValueId> observed = observed_of(traced);
    const SushiRuntime::span<const ValueId> observed_span(observed.data(), observed.size());

    CastOptions casts;
    casts.policy = &policy;
    casts.observed = observed_span;
    Graph::insert_casts(traced.graph, casts);

    FusionOptions fusion;
    fusion.observed = observed_span;
    const Graph::FusionReport fused = Graph::fuse(traced.graph, fusion);

    EXPECT_GT(fused.count(FusedForm::GEMM_BIAS_RELU), 0u)
        << "the casts landed inside a fusable chain and broke it";

    const std::vector<DTypeMismatch> mismatches =
        Graph::find_dtype_mismatches(traced.graph, policy);
    EXPECT_TRUE(mismatches.empty()) << describe_all(mismatches);

    // After the *whole* pipeline, not just after the cast pass.
    for (ValueId parameter : {traced.w1, traced.b1, traced.w2, traced.b2})
    {
        EXPECT_EQ(traced.graph.value(parameter).dtype, DType::FLOAT32);
        EXPECT_EQ(traced.graph.value(parameter).lifetime, Lifetime::PARAMETER);
    }

    for (ValueId gradient : traced.gradients)
    {
        ASSERT_NE(gradient, NO_VALUE);
        EXPECT_EQ(traced.graph.value(gradient).dtype, DType::FLOAT32);
        EXPECT_EQ(traced.graph.value(gradient).lifetime, Lifetime::GRADIENT);
    }
}

TEST(Unit_CastInsertion, ADeliberatelyBrokenGraphIsReportedRatherThanAccepted)
{
    // find_dtype_mismatches is the test's own instrument, so it has to be shown
    // to fail on a graph that is wrong - otherwise every "no mismatches" above
    // could be a checker that never looks.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{ROWS, IN_FEATURES}, "x", DType::FLOAT32);
    const ValueId w = builder.parameter(Shape{HIDDEN, IN_FEATURES}, "w", DType::FLOAT32);
    builder.matmul(x, w, false, true);

    const MixedPrecisionPolicy policy;
    const std::vector<DTypeMismatch> mismatches = Graph::find_dtype_mismatches(graph, policy);

    // An uncast fp32 matmul under a mixed policy: two operands and one result,
    // all three of them float32 where the policy says half.
    ASSERT_EQ(mismatches.size(), 3u);
    for (const DTypeMismatch& mismatch : mismatches)
    {
        EXPECT_EQ(mismatch.op, Ops::MATMUL.id);
        EXPECT_EQ(mismatch.found, DType::FLOAT32);
        EXPECT_EQ(mismatch.expected, DType::HALF);
    }

    EXPECT_NE(Graph::describe(mismatches.front()).find("sushiai.matmul"), std::string::npos);
}

// =========================================================================
// 4. Master weights, and the direction that goes silently wrong
// =========================================================================

TEST(Unit_CastInsertion, AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy)
{
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    for (ValueId parameter : {traced.w1, traced.b1, traced.w2, traced.b2})
    {
        EXPECT_EQ(traced.graph.value(parameter).dtype, DType::FLOAT32)
            << "a master weight was retyped in place";
        EXPECT_EQ(traced.graph.value(parameter).lifetime, Lifetime::PARAMETER);

        // The fp16 the GEMM reads is a *different* value, produced by a cast
        // and living with the step's scratch rather than for the whole run.
        const auto casts = report.casts();
        const auto found = std::find_if(casts.begin(), casts.end(),
            [parameter](const InsertedCast& cast) { return cast.source == parameter; });

        ASSERT_NE(found, casts.end())
            << "no low-precision copy was derived from parameter " << parameter;
        EXPECT_EQ(found->to, DType::HALF);
        EXPECT_TRUE(found->feeds_operand);
        EXPECT_EQ(traced.graph.value(found->destination).lifetime, Lifetime::ACTIVATION);
        EXPECT_EQ(traced.graph.value(found->destination).dtype, DType::HALF);
    }
}

TEST(Unit_CastInsertion, AGradientIsWidenedBeforeItReachesItsBuffer)
{
    // The backward direction. A matmul computing in fp16 produces an fp16
    // gradient, and if that reached the fp32 gradient buffer uncast, SushiBLAS
    // would throw - but if the *buffer* had been retyped to fp16 instead,
    // nothing would throw and the optimizer would accumulate in fp16, which is
    // the whole failure master weights exist to prevent.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    // w1 and w2 are produced by a backward MATMUL, which computes low; b1 and
    // b2 come from SUM_ROWS, which does not, so only the weights need widening.
    for (ValueId weight_gradient : {traced.gradients[0], traced.gradients[2]})
    {
        ASSERT_NE(weight_gradient, NO_VALUE);
        EXPECT_EQ(traced.graph.value(weight_gradient).dtype, DType::FLOAT32);

        const NodeId producer = traced.graph.value(weight_gradient).producer;
        ASSERT_NE(producer, NO_NODE);
        EXPECT_EQ(traced.graph.node(producer).op, Ops::CAST.id)
            << "an fp16 matmul wrote straight into an fp32 gradient buffer";

        const ValueId narrow = traced.graph.node(producer).inputs.at(0);
        EXPECT_EQ(traced.graph.value(narrow).dtype, DType::HALF);
        EXPECT_EQ(traced.graph.node(traced.graph.value(narrow).producer).op, Ops::MATMUL.id);
    }

    // The bias gradients come out of a reduction that stayed wide, so nothing
    // was widened into them and nothing needed to be.
    for (ValueId bias_gradient : {traced.gradients[1], traced.gradients[3]})
    {
        EXPECT_EQ(traced.graph.value(bias_gradient).dtype, DType::FLOAT32);
        EXPECT_EQ(traced.graph.node(traced.graph.value(bias_gradient).producer).op,
                  Ops::SUM_ROWS.id);
    }

    EXPECT_EQ(report.count_into(DType::FLOAT32) > 0, true);
}

TEST(Unit_CastInsertion, AccumulatingGradientContributionsHappensWide)
{
    // A parameter read twice gets two gradient contributions and an ACCUMULATE
    // to sum them. Two things have to hold at once: the sum runs in the wide
    // type (it is a sum), and its result and its first operand agree - because
    // the lowering makes the result *alias the accumulator's buffer*, so a
    // disagreement there is a wrong-width write and not merely a wrong number.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{ROWS, IN_FEATURES}, "x");
    const ValueId targets = builder.input(Shape{ROWS, CLASSES}, "targets");
    const ValueId w = builder.parameter(Shape{CLASSES, IN_FEATURES}, "w");

    // The same weight in two products, summed - the smallest fan-out there is.
    const ValueId a = builder.matmul(x, w, false, true);
    const ValueId b = builder.matmul(x, w, false, true);
    const ValueId logits = builder.add(a, b);
    const ValueId loss = builder.nll_loss(builder.log_softmax(logits), targets);

    const ValueId wrt[] = {w};
    const std::vector<ValueId> gradients =
        Autograd::differentiate(graph, loss, SushiRuntime::span<const ValueId>(wrt, 1));

    ASSERT_EQ(count_op(graph, Ops::ACCUMULATE.id), 1u);

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const ValueId observed[] = {x, targets, logits, loss};
    options.observed = SushiRuntime::span<const ValueId>(observed, 4);

    Graph::insert_casts(graph, options);

    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, policy).empty());
    EXPECT_EQ(graph.value(gradients[0]).dtype, DType::FLOAT32);

    for (const Graph::Node& node : graph.nodes())
        if (node.op == Ops::ACCUMULATE.id)
        {
            EXPECT_EQ(graph.value(node.outputs[0]).dtype, DType::FLOAT32);
            EXPECT_EQ(graph.value(node.inputs[0]).dtype, DType::FLOAT32);
            EXPECT_EQ(graph.value(node.inputs[1]).dtype, DType::FLOAT32);
        }
}

// =========================================================================
// 5. What the pass promises about ids, reuse and repetition
// =========================================================================

TEST(Unit_CastInsertion, AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot)
{
    // The half of the pinning rule that is *not* covered by a lifetime class.
    // The logits are an ordinary activation; the training loop reads them back
    // by id for accuracy, and a caller who forgets to say so gets an fp16
    // buffer where their host-side argmax expects floats.
    Traced pinned = trace();
    Traced loose = trace();

    const MixedPrecisionPolicy policy;

    CastOptions with;
    with.policy = &policy;
    const std::vector<ValueId> observed = observed_of(pinned);
    with.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());
    Graph::insert_casts(pinned.graph, with);

    CastOptions without;
    without.policy = &policy;
    Graph::insert_casts(loose.graph, without);

    EXPECT_EQ(pinned.graph.value(pinned.logits).dtype, DType::FLOAT32);
    EXPECT_EQ(loose.graph.value(loose.logits).dtype, DType::HALF);

    // Naming a value that does not exist is a caller error worth reporting.
    Traced other = trace();
    CastOptions bad;
    bad.policy = &policy;
    const ValueId nonsense[] = {static_cast<ValueId>(other.graph.value_count() + 10)};
    bad.observed = SushiRuntime::span<const ValueId>(nonsense, 1);
    EXPECT_THROW(Graph::insert_casts(other.graph, bad), Error);
}

TEST(Unit_CastInsertion, AValueReadTwiceIsConvertedOnce)
{
    // w2 is read by the forward matmul and again by the backward one. Two
    // conversions would be two kernels and two buffers for one number, and the
    // reuse is safe precisely because the IR is single-assignment.
    Traced traced = trace();

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

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

TEST(Unit_CastInsertion, ValueIdsSurviveThePass)
{
    Traced traced = trace();
    const std::size_t values_before = traced.graph.value_count();

    std::vector<Shape> shapes;
    std::vector<Lifetime> lifetimes;
    for (ValueId id = 0; id < static_cast<ValueId>(values_before); ++id)
    {
        shapes.push_back(traced.graph.value(id).shape);
        lifetimes.push_back(traced.graph.value(id).lifetime);
    }

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    Graph::insert_casts(traced.graph, options);

    EXPECT_GT(traced.graph.value_count(), values_before);
    for (ValueId id = 0; id < static_cast<ValueId>(values_before); ++id)
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

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

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

    const std::vector<ValueId> observed = observed_of(traced);
    options.observed = SushiRuntime::span<const ValueId>(observed.data(), observed.size());

    const CastReport report = Graph::insert_casts(traced.graph, options);

    EXPECT_EQ(report.policy(), "mixed");
    EXPECT_EQ(report.count(), count_op(traced.graph, Ops::CAST.id));
    EXPECT_EQ(report.nodes_after() - report.nodes_before(), report.count());
    EXPECT_EQ(report.count(), report.count_into(DType::HALF) + report.count_into(DType::FLOAT32));
    EXPECT_GT(report.count_into(DType::HALF), 0u);
    EXPECT_GT(report.count_into(DType::FLOAT32), 0u);
    EXPECT_GT(report.retyped_values(), 0u);

    EXPECT_NE(report.to_string().find("mixed"), std::string::npos);
    EXPECT_NE(report.to_string().find("casts inserted"), std::string::npos);
}

TEST(Unit_CastInsertion, AGraphThatIsNotInTopologicalOrderIsRefused)
{
    // The same guard fuse() and MemoryPlan::build() carry, for the same reason:
    // the pass emits a conversion immediately before its consumer, which is
    // only a valid position if producers already come first.
    Graph::Graph graph;

    const ValueId a = graph.add_value(Shape{ROWS}, DType::FLOAT32, Lifetime::ACTIVATION, "a");
    const ValueId b = graph.add_value(Shape{ROWS}, DType::FLOAT32, Lifetime::ACTIVATION, "b");

    graph.add_node(Ops::RELU.id, {b}, {a});   // reads b...
    graph.add_node(Ops::RELU.id, {a}, {b});   // ...which is written here.

    const MixedPrecisionPolicy policy;
    CastOptions options;
    options.policy = &policy;

    EXPECT_THROW(Graph::insert_casts(graph, options), Error);
}

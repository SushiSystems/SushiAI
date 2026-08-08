/**************************************************************************/
/* test_autograd_numeric.cpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Finite-difference verification of every backward rule, in FLOAT64.
//
// The whole graph - forward, loss and backward - runs on the device in double
// precision, and the analytic gradient it produces is compared against a
// central difference of the loss the same graph computes. That is the only
// check that can distinguish a correct backward rule from one that is merely
// plausible, and it must be fp64: an fp32 difference quotient bottoms out
// near 2.4e-5 relative accuracy, which is the same order as the error a
// genuinely wrong rule produces. See gradcheck.hpp.

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;

namespace
{
    class Integration_Autograd : public SaTest::EngineTest {};

    /**
     * @brief A FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     *
     * The variable under test is declared as the only gradient-requiring leaf;
     * everything the build callback declares alongside it is a constant the
     * test fills. The loss is always the mean negative log-likelihood against
     * a target matrix, which is a perfectly good scalar functional of anything
     * shaped [N, C] - and, when the model ends in a log-softmax, exercises the
     * fused cross-entropy rule as well.
     */
    class Probe
    {
        public:
            template<typename BuildFn>
            Probe(SushiBLAS::Engine& engine, Shape variable_shape, BuildFn&& build,
                  bool fuse = false)
                : pool_(engine)
            {
                const GraphBuilder builder(graph_);

                variable_ = builder.parameter(variable_shape, "x", DType::FLOAT64);

                const ValueId logits = build(builder, variable_);
                const Shape logit_shape = graph_.value(logits).shape;

                targets_ = builder.input(logit_shape, "targets", DType::FLOAT64);
                loss_ = builder.nll_loss(logits, targets_);

                const std::vector<ValueId> wrt = {variable_};
                const std::vector<ValueId> grads = Autograd::differentiate(
                    graph_, loss_, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

                gradient_ = grads.at(0);
                EXPECT_NE(gradient_, Graph::NO_VALUE);

                // Off by default, because every case below this one is checking
                // a backward *rule* and fusion is not allowed to change one. It
                // is switchable because one case is checking the opposite
                // direction: that a fused form which rewrites what the backward
                // pass reads still leaves the same derivative behind. Ids
                // survive the pass, so nothing recorded above needs remapping.
                if (fuse)
                {
                    const ValueId observed[] = {loss_, targets_};
                    Graph::FusionOptions options;
                    options.observed = SushiRuntime::span<const ValueId>(observed, 2);
                    fusion_ = Graph::fuse(graph_, options);
                }

                pool_.materialise_leaves(graph_);
                step_ = std::make_unique<CompiledStep>(engine, graph_, pool_);
            }

            /** @brief What the fusion pass did, when the probe was asked to run it. */
            const Graph::FusionReport& fusion() const noexcept { return fusion_; }

            /** @brief What the lowering actually spelled, so a decline cannot pass unnoticed. */
            SushiRuntime::span<const Graph::FusionSelection> selections() const noexcept
            {
                return step_->selections();
            }

            /** @brief Writes a constant leaf's values. */
            void fill(ValueId id, const std::vector<double>& values)
            {
                SaTest::write_tensor(pool_.tensor(id), values);
            }

            /** @brief Fills a constant leaf from the deterministic host stream. */
            void fill_noise(ValueId id, std::uint64_t seed, double scale = 1.0)
            {
                std::vector<double> values(
                    static_cast<std::size_t>(pool_.tensor(id).num_elements));
                SaTest::HostNoise(seed).fill(values, scale);
                fill(id, values);
            }

            /** @brief Fills the target matrix with a balanced one-hot pattern. */
            void fill_one_hot(int64_t rows, int64_t classes)
            {
                std::vector<double> values(static_cast<std::size_t>(rows * classes), 0.0);
                for (int64_t row = 0; row < rows; ++row)
                    values[static_cast<std::size_t>(row * classes + row % classes)] = 1.0;
                fill(targets_, values);
            }

            /** @brief The variable's element count. */
            std::size_t size() const { return static_cast<std::size_t>(
                pool_.tensor(variable_).num_elements); }

            /** @brief The target matrix's value id, for a test that wants to shape it. */
            ValueId targets() const noexcept { return targets_; }

            /** @brief The loss at @p point. */
            double value(SushiRuntime::span<const double> point)
            {
                SaTest::write_tensor(pool_.tensor(variable_), point);
                step_->run();
                return SaTest::read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief The analytic gradient at @p point. */
            void grad(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                SaTest::write_tensor(pool_.tensor(variable_), point);
                step_->run();

                const std::vector<double> values = SaTest::read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i) out[i] = values[i];
            }

            /** @brief A ScalarFunction bound to this probe. */
            Autograd::ScalarFunction value_fn()
            {
                return [this](SushiRuntime::span<const double> point) { return value(point); };
            }

            /** @brief A GradientFunction bound to this probe. */
            Autograd::GradientFunction grad_fn()
            {
                return [this](SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
                {
                    grad(point, out);
                };
            }

        private:
            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueId variable_ = Graph::NO_VALUE;
            ValueId targets_ = Graph::NO_VALUE;
            ValueId loss_ = Graph::NO_VALUE;
            ValueId gradient_ = Graph::NO_VALUE;
            Graph::FusionReport fusion_;
    };

    /** @brief Runs the full-Jacobian check and reports the result on failure. */
    void expect_gradient_matches(Probe& probe, std::vector<double>& point,
                                 const Autograd::GradCheckOptions& options = {})
    {
        ASSERT_LE(point.size(), 200u) << "use the directional mode above 200 inputs";

        const Autograd::GradCheckResult result =
            Autograd::check_full_jacobian(probe.value_fn(), probe.grad_fn(), point, options);

        EXPECT_TRUE(result.passed) << result.to_string();
    }
}

TEST_F(Integration_Autograd, MatmulWithoutTransposes)
{
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 5}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::FLOAT64);
        return builder.matmul(x, other, false, false);
    });

    probe.fill_noise(other, 11, 1.0);
    probe.fill_noise(probe.targets(), 12, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(13).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, MatmulWithATransposedRightOperand)
{
    // The shape Linear actually emits: X [N, in] against W [out, in]^T.
    ValueId weight = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 5}, [&weight](const GraphBuilder& builder, ValueId x)
    {
        weight = builder.input(Shape{3, 5}, "w", DType::FLOAT64);
        return builder.matmul(x, weight, false, true);
    });

    probe.fill_noise(weight, 21, 1.0);
    probe.fill_noise(probe.targets(), 22, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(23).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, MatmulWithATransposedLeftOperand)
{
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{5, 4}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::FLOAT64);
        return builder.matmul(x, other, true, false);
    });

    probe.fill_noise(other, 31, 1.0);
    probe.fill_noise(probe.targets(), 32, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(33).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, MatmulWithBothOperandsTransposed)
{
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{5, 4}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{3, 5}, "b", DType::FLOAT64);
        return builder.matmul(x, other, true, true);
    });

    probe.fill_noise(other, 41, 1.0);
    probe.fill_noise(probe.targets(), 42, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(43).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, MatmulGradientOfTheRightOperand)
{
    // The same node from the other side: the variable is now the weight, so
    // this covers the dB half of the rule rather than the dA half.
    ValueId activations = Graph::NO_VALUE;

    Probe probe(*engine, Shape{3, 5}, [&activations](const GraphBuilder& builder, ValueId w)
    {
        activations = builder.input(Shape{4, 5}, "x", DType::FLOAT64);
        return builder.matmul(activations, w, false, true);
    });

    probe.fill_noise(activations, 51, 1.0);
    probe.fill_noise(probe.targets(), 52, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(53).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, BiasAddPassesTheMatrixGradientThrough)
{
    ValueId bias = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&bias](const GraphBuilder& builder, ValueId x)
    {
        bias = builder.input(Shape{3}, "b", DType::FLOAT64);
        return builder.bias_add(x, bias);
    });

    probe.fill_noise(bias, 61, 1.0);
    probe.fill_noise(probe.targets(), 62, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(63).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, BiasAddSumsOverTheBroadcastAxis)
{
    ValueId base = Graph::NO_VALUE;

    Probe probe(*engine, Shape{3}, [&base](const GraphBuilder& builder, ValueId bias)
    {
        base = builder.input(Shape{4, 3}, "x", DType::FLOAT64);
        return builder.bias_add(base, bias);
    });

    probe.fill_noise(base, 71, 1.0);
    probe.fill_noise(probe.targets(), 72, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(73).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ElementwiseAddAndSubtract)
{
    for (int which = 0; which < 2; ++which)
    {
        ValueId other = Graph::NO_VALUE;

        Probe probe(*engine, Shape{4, 3}, [&other, which](const GraphBuilder& builder, ValueId x)
        {
            other = builder.input(Shape{4, 3}, "b", DType::FLOAT64);
            return which == 0 ? builder.add(x, other) : builder.sub(x, other);
        });

        probe.fill_noise(other, 81 + static_cast<std::uint64_t>(which), 1.0);
        probe.fill_noise(probe.targets(), 91 + static_cast<std::uint64_t>(which), 1.0);

        std::vector<double> point(probe.size());
        SaTest::HostNoise(101 + static_cast<std::uint64_t>(which)).fill(point, 1.0);

        expect_gradient_matches(probe, point);
    }
}

TEST_F(Integration_Autograd, ElementwiseSubtractFromTheRight)
{
    // dB = -dC, the half the left-hand check cannot reach.
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::FLOAT64);
        return builder.sub(other, x);
    });

    probe.fill_noise(other, 111, 1.0);
    probe.fill_noise(probe.targets(), 112, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(113).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ElementwiseMultiply)
{
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{4, 3}, "b", DType::FLOAT64);
        return builder.mul(x, other);
    });

    probe.fill_noise(other, 121, 1.0);
    probe.fill_noise(probe.targets(), 122, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(123).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ElementwiseMultiplyFromTheRight)
{
    // dB = dC * A, the half the left-hand check cannot reach. The rule is
    // written symmetrically, which is exactly why a copy-paste slip in it -
    // reading inputs[1] on both sides - would leave the left-hand test green.
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::FLOAT64);
        return builder.mul(other, x);
    });

    probe.fill_noise(other, 124, 1.0);
    probe.fill_noise(probe.targets(), 125, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(126).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, AccumulateInTheForwardGraph)
{
    // ACCUMULATE has a backward rule and, until this test, nothing exercised
    // it. The two fan-out tests above build an ACCUMULATE chain in the
    // BACKWARD graph, and differentiate() only walks the forward nodes - so
    // those chains are never themselves differentiated. Reaching the rule at
    // all needs an ACCUMULATE the caller put in the forward pass.
    //
    // Its lowering aliases the result onto the accumulator's own buffer, so
    // running the step leaves the variable holding x + other; every probe
    // rewrites the whole variable first, which is what keeps that harmless.
    ValueId other = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueId x)
    {
        other = builder.input(Shape{4, 3}, "delta", DType::FLOAT64);
        return builder.accumulate(x, other);
    });

    probe.fill_noise(other, 231, 1.0);
    probe.fill_noise(probe.targets(), 232, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(233).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Scale)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.scale(x, -2.5);
    });

    probe.fill_noise(probe.targets(), 131, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(132).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Exp)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.exp(x);
    });

    probe.fill_noise(probe.targets(), 141, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(142).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ReluAwayFromItsKink)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.relu(x);
    });

    probe.fill_noise(probe.targets(), 151, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(152).fill(point, 1.0);

    // ReLU is not differentiable at zero, and a difference quotient that
    // straddles the kink reports the average of the two one-sided slopes,
    // which no correct rule can match. Sample clear of it.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Gelu)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.gelu(x);
    });

    probe.fill_noise(probe.targets(), 161, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(162).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, AFusedGeluEpilogueHasTheSameDerivativeAsAnUnfusedOne)
{
    // The strongest oracle in the repository, pointed at the one fused form
    // whose correctness is not a matter of "the same operations in the same
    // order". GEMM_BIAS_GELU *rewrites what the backward pass reads*: the
    // pre-activation now arrives from a store inside a GEMM kernel rather than
    // from the ger that used to write it. A central difference does not care
    // where it came from - it asks the graph what its own loss does when the
    // variable moves, and compares that with what the graph says its gradient
    // is. A spill holding the wrong tensor cannot survive that, and would
    // survive any amount of "it trained and the loss fell".
    //
    // FLOAT64 throughout, as gradcheck.hpp requires; the whole epilogue is
    // instantiated for double by execute_gemm's REAL domain.
    ValueId weight = Graph::NO_VALUE;
    ValueId bias = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 5},
                [&](const GraphBuilder& builder, ValueId x)
                {
                    // x is the variable: [4, 5] activations into a [3, 5]^T
                    // weight, so the chain is matmul -> bias_add -> gelu and the
                    // logits are [4, 3].
                    weight = builder.input(Shape{3, 5}, "w", DType::FLOAT64);
                    bias = builder.input(Shape{3}, "b", DType::FLOAT64);
                    return builder.gelu(
                        builder.bias_add(builder.matmul(x, weight, false, true), bias));
                },
                /*fuse=*/true);

    // The test is only a test if the fused arm actually ran. Both halves are
    // checked: the static pass matched, and the selector did not hand the
    // lowering back the unfused spelling - which it would on a device without
    // double precision, leaving this case quietly measuring the old path.
    ASSERT_EQ(probe.fusion().count(Graph::FusedForm::GEMM_BIAS_GELU), 1u)
        << probe.fusion().to_string();

    bool spelled = false;
    for (const Graph::FusionSelection& selection : probe.selections())
        if (selection.chosen == Graph::FusedForm::GEMM_BIAS_GELU) spelled = true;

    ASSERT_TRUE(spelled)
        << "this device declined the fused GELU epilogue, so the gradient below is the "
           "unfused one and this case is not checking what it claims to";

    probe.fill_noise(weight, 171, 0.8);
    probe.fill_noise(bias, 172, 0.5);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(173).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, SigmoidReadsItsForwardOutput)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.sigmoid(x);
    });

    probe.fill_noise(probe.targets(), 171, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(172).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, TanhReadsItsForwardOutput)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.tanh(x);
    });

    probe.fill_noise(probe.targets(), 181, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(182).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, SumRowsBroadcastsItsGradientBack)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        // [4, 3] -> [3] -> [1, 3], so the loss can consume it as a one-row batch.
        const ValueId sums = builder.sum_rows(x);
        return builder.bias_add(builder.zeros(Shape{1, 3}, DType::FLOAT64), sums);
    });

    probe.fill_noise(probe.targets(), 191, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(192).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, FusedCrossEntropyAgainstOneHotTargets)
{
    // The headline gradient: d/dlogits of the mean NLL is (softmax - Y) / N.
    // It is emitted as one fused rule rather than by differentiating the
    // seven-op log-softmax recipe, so it needs its own numeric proof.
    Probe probe(*engine, Shape{4, 5}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.log_softmax(x);
    });

    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(201).fill(point, 2.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, GradientAccumulationAtAFanOut)
{
    // Two branches over one variable, so the gradient is an ACCUMULATE chain
    // rather than a single contribution.
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.add(builder.tanh(x), builder.sigmoid(x));
    });

    probe.fill_noise(probe.targets(), 211, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(212).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, WholeTwoLayerModelUnderRandomDirections)
{
    // The end-to-end check. A full Jacobian over every weight would be 2n
    // device runs; k random unit directions is 2k, independent of n, and a
    // wrong gradient survives sixteen of them with vanishing probability.
    ValueId hidden_weight = Graph::NO_VALUE;
    ValueId output_weight = Graph::NO_VALUE;
    ValueId inputs = Graph::NO_VALUE;

    Probe probe(*engine, Shape{6}, [&](const GraphBuilder& builder, ValueId bias)
    {
        inputs = builder.input(Shape{4, 5}, "x", DType::FLOAT64);
        hidden_weight = builder.input(Shape{6, 5}, "w0", DType::FLOAT64);
        output_weight = builder.input(Shape{3, 6}, "w1", DType::FLOAT64);

        const ValueId hidden = builder.relu(
            builder.bias_add(builder.matmul(inputs, hidden_weight, false, true), bias));

        return builder.log_softmax(builder.matmul(hidden, output_weight, false, true));
    });

    probe.fill_noise(inputs, 221, 1.0);
    probe.fill_noise(hidden_weight, 222, 0.6);
    probe.fill_noise(output_weight, 223, 0.6);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(224).fill(point, 0.5);
    Autograd::push_off_kinks(point);

    const Autograd::GradCheckResult result =
        Autograd::check_directional(probe.value_fn(), probe.grad_fn(), point, 16);

    EXPECT_TRUE(result.passed) << result.to_string();
}

TEST_F(Integration_Autograd, ReshapeMovesItsGradientBackUnchanged)
{
    // A reshape moves no data, so its gradient is its incoming gradient in the
    // original shape. Worth a numeric check anyway: "reshape back to the input
    // shape" and "reshape back to the output shape" are both one line and only
    // one is right.
    Probe probe(*engine, Shape{3, 8}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.log_softmax(builder.reshape(builder.reshape(x, Shape{4, 6}), Shape{3, 8}));
    });

    // The fused LOG_SOFTMAX + NLL_LOSS rule's closed form assumes a one-hot
    // target row (nll_loss_backward's comment: "valid because the targets are
    // one-hot"); noise there would make the *analytic* gradient legitimately
    // disagree with the numeric one for reasons that have nothing to do with
    // RESHAPE.
    probe.fill_one_hot(3, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(232).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, LayoutConvertsGradientIsThePermutedGradient)
{
    // The load-bearing case: a *single* layout_convert, so the loss genuinely
    // depends on the permuted element order rather than on a round trip that
    // any involutive, shape-correct pair of backward hops would also pass -
    // a rule that mistakenly reshaped dy back to the operand's shape instead
    // of permuting it (same magnitude, wrong place) would still satisfy
    // every shape check Differentiator::contribute makes, and a round trip
    // cannot tell the two apart because both compose back to the identity.
    //
    // CV-4 forces every non-LEAF, non-LAYOUT_CONVERT node's operand and
    // result to agree on layout, so a *variable* cannot legally reach
    // log_softmax by converting one way and stopping - graph::finalise()
    // would refuse the mismatched log_softmax output before any gradient
    // ran. What CV-4 exempts is a leaf itself: a LEAF has no operands to
    // disagree with, so its own layout tag is free. This is also the honest
    // real-world shape of the op: LAYOUT_CONVERT exists for boundary
    // interop, e.g. loading an NCHW checkpoint, where the *source* data
    // (not some mid-graph value) is the thing tagged the other way. Tagging
    // the variable NCHW before its one LAYOUT_CONVERT node, then reshaping
    // straight into log_softmax, reaches a CV-4-legal NHWC consumer with no
    // round trip at all - so the permutation is genuinely load-bearing.
    Probe probe(*engine, Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueId x)
    {
        builder.graph().value(x).layout = Graph::TensorLayout::NCHW;
        const ValueId nhwc = builder.layout_convert(x, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(nhwc, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(291).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, LayoutConvertRoundTripComposesBothDirections)
{
    // The two directions composing back to the identity is a fair thing to
    // check on its own - but by itself it cannot tell a correct permutation
    // from a same-shape non-permutation (see the single-hop case above for
    // why), so it is kept only as a secondary check, not the proof.
    Probe probe(*engine, Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueId x)
    {
        const ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
        const ValueId back = builder.layout_convert(nchw, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(back, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(242).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, AWholeConvolutionDifferentiatesThroughIm2colAndMatmul)
{
    // CV-1's claim, checked numerically: with no CONV2D identity, a
    // convolution's gradient is im2col_backward composed with the matmul
    // rule that was already there. If this passes, convolution needs no
    // rule of its own - which is the entire reason there isn't one.
    //
    // Single-hop by design (A-7): im2col feeds straight into a matmul, never
    // into its own col2im. A round trip through col2im would make the
    // forward composite the identity, and then any involutive, shape-correct
    // backward - including a plain reshape - would pass for the wrong
    // reason.
    // spatial_mean has no gradient rule yet (Task 12), so [N, C] is reached
    // straight off the matmul: with a 1x1 window grid folded into rows there
    // would be nothing to average over anyway, and the matmul's own output is
    // already rank 2 - [N*H_o*W_o, C_out] - so no reshape is needed either.
    ValueId weights = Graph::NO_VALUE;

    Probe probe(*engine, Shape{2, 5, 5, 3}, [&weights](const GraphBuilder& builder, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = builder.im2col(x, g);                      // [2*25, 27]
        weights = builder.parameter(Shape{27, 8}, "w", DType::FLOAT64); // 8 output channels
        return builder.log_softmax(builder.matmul(cols, weights));      // [2*25, 8]
    });

    probe.fill_noise(weights, 301, 1.0);
    // A probe ending in log_softmax must use one-hot targets (A-6): the fused
    // NLL_LOSS rule's closed form is only the true derivative of this loss
    // when the target row is one-hot.
    probe.fill_one_hot(50, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(302).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Im2colsGradientAccumulatesEveryWindowThatReadAPixel)
{
    // A stride-1 padded 3x3 has every interior pixel read by nine windows, so
    // its gradient is a sum of nine terms. A col2im that overwrote instead of
    // accumulating would give exactly one of them - a gradient that is the
    // right shape, the right sign, and roughly a ninth of the right size.
    ValueId weights = Graph::NO_VALUE;

    Probe probe(*engine, Shape{1, 4, 4, 2}, [&weights](const GraphBuilder& builder, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = builder.im2col(x, g);
        weights = builder.parameter(Shape{18, 4}, "w", DType::FLOAT64);
        return builder.log_softmax(builder.matmul(cols, weights)); // [16, 4]
    });

    probe.fill_noise(weights, 311, 1.0);
    probe.fill_one_hot(16, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(312).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Col2imsGradientGathersEveryWindowThatWroteAPixel)
{
    // COL2IM's own rule (im2col_backward's mirror image), checked without
    // routing through im2col at all: the variable here *is* the column
    // matrix, fed straight into col2im and then reshaped into log_softmax.
    // That keeps this single-hop by construction (A-7) - there is no im2col
    // on this path for a col2im to be the adjoint of, so nothing here could
    // be satisfied by an involutive round trip.
    Probe probe(*engine, Shape{9, 18}, [](const GraphBuilder& builder, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId image = builder.col2im(x, Shape{1, 3, 3, 2}, g); // [1, 3, 3, 2]
        return builder.log_softmax(builder.reshape(image, Shape{1, 18}));
    });

    probe.fill_one_hot(1, 18);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(321).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

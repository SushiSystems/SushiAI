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
#include <span>
#include <string>
#include <vector>
#include <gtest/gtest.h>
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
            Probe(SushiBLAS::Engine& engine, Shape variable_shape, BuildFn&& build)
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
                    graph_, loss_, std::span<const ValueId>(wrt.data(), wrt.size()));

                gradient_ = grads.at(0);
                EXPECT_NE(gradient_, Graph::NO_VALUE);

                pool_.materialise_leaves(graph_);
                step_ = std::make_unique<CompiledStep>(engine, graph_, pool_);
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
            double value(std::span<const double> point)
            {
                SaTest::write_tensor(pool_.tensor(variable_), point);
                step_->run();
                return SaTest::read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief The analytic gradient at @p point. */
            void grad(std::span<const double> point, std::span<double> out)
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
                return [this](std::span<const double> point) { return value(point); };
            }

            /** @brief A GradientFunction bound to this probe. */
            Autograd::GradientFunction grad_fn()
            {
                return [this](std::span<const double> point, std::span<double> out)
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

/****************************************************************/
/* test_autograd_numeric.cpp                                    */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_autograd_numeric.cpp
 * @brief Tests every backward rule against a central difference of the loss, in FLOAT64.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/functional/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;

namespace
{
    class Integration_Autograd : public SaTest::EngineTest {};

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     * @pre The variable under test is the only gradient-requiring leaf; see
     *      tests/functional/integration/README.md.
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

                // Fusion is off unless a case asks; see tests/functional/integration/README.md.
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
    // The weight is the variable, covering dB; see tests/functional/integration/README.md.
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
    // Covers dB = dC * A, the right-hand half; see tests/functional/integration/README.md.
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
    // Every probe rewrites the whole variable first; see tests/functional/integration/README.md.
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

    // ReLU is sampled clear of its kink at zero; see tests/functional/integration/README.md.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, LeakyReluHasBothOfItsSlopes)
{
    // The sample range puts elements on both slopes; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape{4, 8}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.log_softmax(builder.leaky_relu(x, 0.1));
    });

    probe.fill_one_hot(4, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(171).fill(point, 1.5);

    // LEAKY_RELU is not differentiable at zero, same as RELU.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, LeakyReluBranchSwapIsCaughtByProbePoints)
{
    // Pins the sign split of the sample used above; see tests/functional/integration/README.md.
    std::vector<double> point(4 * 8);
    SaTest::HostNoise(171).fill(point, 1.5);
    Autograd::push_off_kinks(point);

    const auto negative = std::count_if(point.begin(), point.end(),
                                        [](double v) { return v < 0.0; });
    EXPECT_GE(negative, 10) << "too few negative-side points to exercise the "
                                "slope branch of leaky_relu's gradient";

    const double min_abs = std::abs(*std::min_element(point.begin(), point.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); }));
    EXPECT_GT(min_abs, 0.05) << "a point too close to zero would sit on the kink "
                                 "push_off_kinks is supposed to have cleared";
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
    // The fused epilogue is checked in FLOAT64; see tests/functional/integration/README.md.
    ValueId weight = Graph::NO_VALUE;
    ValueId bias = Graph::NO_VALUE;

    Probe probe(*engine, Shape{4, 5},
                [&](const GraphBuilder& builder, ValueId x)
                {
                    // x is the variable: matmul -> bias_add -> gelu gives [4, 3] logits.
                    weight = builder.input(Shape{3, 5}, "w", DType::FLOAT64);
                    bias = builder.input(Shape{3}, "b", DType::FLOAT64);
                    return builder.gelu(
                        builder.bias_add(builder.matmul(x, weight, false, true), bias));
                },
                /*fuse=*/true);

    // The fused arm must really have run; see tests/functional/integration/README.md.
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
    // The fused rule gives (softmax - Y) / N; see tests/functional/integration/README.md.
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
    // Two branches over one variable make the gradient an ACCUMULATE chain.
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
    // Sixteen random directions cost 2k runs, not 2n; see tests/functional/integration/README.md.
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
    // The gradient returns in the input shape; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape{3, 8}, [](const GraphBuilder& builder, ValueId x)
    {
        return builder.log_softmax(builder.reshape(builder.reshape(x, Shape{4, 6}), Shape{3, 8}));
    });

    // The fused NLL rule needs one-hot targets; see tests/functional/integration/README.md.
    probe.fill_one_hot(3, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(232).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, LayoutConvertsGradientIsThePermutedGradient)
{
    // A single layout_convert, so the order matters; see tests/functional/integration/README.md.
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
    // The round trip is a secondary check only; see tests/functional/integration/README.md.
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
    // Single hop: im2col feeds matmul, never col2im; see tests/functional/integration/README.md.
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
    // A log_softmax probe needs one-hot targets; see tests/functional/integration/README.md.
    probe.fill_one_hot(50, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(302).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Im2colsGradientAccumulatesEveryWindowThatReadAPixel)
{
    // Nine windows read each interior pixel; see tests/functional/integration/README.md.
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

TEST_F(Integration_Autograd, AnAsymmetricallyPaddedConvolutionDifferentiatesOverItsRealWindowGrid)
{
    // The real grid is 5x4; a symmetric guess is 4x5; see tests/functional/integration/README.md.
    ValueId weights = Graph::NO_VALUE;

    Probe probe(*engine, Shape{1, 4, 5, 1}, [&weights](const GraphBuilder& builder, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = builder.im2col_with_extents(x, g, 5, 4);  // [20, 9]
        weights = builder.parameter(Shape{9, 4}, "w", DType::FLOAT64);
        return builder.log_softmax(builder.matmul(cols, weights));     // [20, 4]
    });

    probe.fill_noise(weights, 331, 1.0);
    probe.fill_one_hot(20, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(332).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, Col2imsGradientGathersEveryWindowThatWroteAPixel)
{
    // The variable is the column matrix itself; see tests/functional/integration/README.md.
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

TEST_F(Integration_Autograd, MaxPoolRoutesEachGradientToTheElementThatWon)
{
    // push_off_kinks leaves no ties in a window; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 3}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        g.stride_h = 2; g.stride_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.fill_one_hot(2, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(401).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, OverlappingMaxPoolWindowsAccumulateOnADominantPixel)
{
    // At stride 1 a pixel can win several windows; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({1, 5, 5, 2}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(411).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, AvgPoolSpreadsItsGradientOverTheWholeWindow)
{
    // One batch row keeps this under 200 inputs; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({1, 6, 6, 3}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 3; g.stride_w = 3;
        return b.log_softmax(b.spatial_mean(b.avgpool2d(x, g)));
    });

    probe.fill_one_hot(1, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(421).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, UpsampleSumsEachSourcePixelsReplicas)
{
    // Scale 2 makes each gradient a sum of four; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({2, 3, 3, 4}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(b.upsample2d(x, 2, 2)));
    });

    probe.fill_one_hot(2, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(431).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, SpatialMeanBroadcastsScaledByOneOverHW)
{
    // The broadcast is per (n, c); see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 5}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(x));
    });

    probe.fill_one_hot(2, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(441).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ConcatGivesEachOperandItsOwnChannelRange)
{
    // Operands are reordered against their slices; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 6}), [](const GraphBuilder& b, ValueId x)
    {
        const ValueId left = b.slice_channels(x, 0, 2);
        const ValueId right = b.slice_channels(x, 2, 4);
        const std::array<ValueId, 2> parts = {right, left};   // deliberately reordered
        return b.log_softmax(b.spatial_mean(b.concat_channels(
            SushiRuntime::span<const ValueId>(parts.data(), parts.size()))));
    });

    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(451).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, SliceScattersItsGradientBackIntoZeros)
{
    // Channels outside the slice receive zero; see tests/functional/integration/README.md.
    Probe probe(*engine, Shape({1, 3, 3, 8}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(b.slice_channels(x, 3, 2)));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(461).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, BatchNormsGradientAccountsForTheStatisticsItComputed)
{
    // dx has three terms, not one; see tests/functional/integration/README.md.
    ValueId gamma = Graph::NO_VALUE;
    ValueId beta = Graph::NO_VALUE;

    Probe probe(*engine, Shape({4, 3, 3, 5}),
                [&gamma, &beta](const GraphBuilder& b, ValueId x)
    {
        gamma = b.parameter(Shape{5}, "gamma", DType::FLOAT64);
        beta = b.parameter(Shape{5}, "beta", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(gamma, 471, 1.0);
    probe.fill_noise(beta, 472, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(473).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput)
{
    // The same node, differentiated at gamma; see tests/functional/integration/README.md.
    ValueId x = Graph::NO_VALUE;
    ValueId beta = Graph::NO_VALUE;

    Probe probe(*engine, Shape{5}, [&x, &beta](const GraphBuilder& b, ValueId gamma)
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::FLOAT64);
        beta = b.parameter(Shape{5}, "beta", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(x, 481, 1.0);
    probe.fill_noise(beta, 482, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(483).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, BatchNormsShiftGradientIsThePlainSumOfDy)
{
    ValueId x = Graph::NO_VALUE;
    ValueId gamma = Graph::NO_VALUE;

    Probe probe(*engine, Shape{5}, [&x, &gamma](const GraphBuilder& b, ValueId beta)
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::FLOAT64);
        gamma = b.parameter(Shape{5}, "gamma", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(x, 491, 1.0);
    probe.fill_noise(gamma, 492, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(493).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ChannelAffineDifferentiatesAtItsInput)
{
    ValueId gamma = Graph::NO_VALUE;
    ValueId beta = Graph::NO_VALUE;

    Probe probe(*engine, Shape({2, 4, 4, 6}),
                [&gamma, &beta](const GraphBuilder& b, ValueId x)
    {
        gamma = b.parameter(Shape{6}, "gamma", DType::FLOAT64);
        beta = b.parameter(Shape{6}, "beta", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(gamma, 501, 1.0);
    probe.fill_noise(beta, 502, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(503).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ChannelAffinesScaleGradientSumsOverNHAndW)
{
    // dgamma sums over N, H and W at once; see tests/functional/integration/README.md.
    ValueId x = Graph::NO_VALUE;
    ValueId beta = Graph::NO_VALUE;

    Probe probe(*engine, Shape{6}, [&x, &beta](const GraphBuilder& b, ValueId gamma)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::FLOAT64);
        beta = b.parameter(Shape{6}, "beta", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 511, 1.0);
    probe.fill_noise(beta, 512, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(513).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Integration_Autograd, ChannelAffinesShiftGradientSumsOverNHAndW)
{
    ValueId x = Graph::NO_VALUE;
    ValueId gamma = Graph::NO_VALUE;

    Probe probe(*engine, Shape{6}, [&x, &gamma](const GraphBuilder& b, ValueId beta)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::FLOAT64);
        gamma = b.parameter(Shape{6}, "gamma", DType::FLOAT64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 521, 1.0);
    probe.fill_noise(gamma, 522, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(523).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

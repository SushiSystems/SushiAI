/**************************************************************************/
/* test_host_values.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_host_values.cpp
 * @brief Tests fourteen lowerings against values computed on the host.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::Traced;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every lowered-operation case records on. */
    class Medium_LoweredOperations : public SaTest::EngineTest
    {

    };

    /** @brief Holds the bound |got - expected| <= absolute + relative * |expected|. */
    struct Tolerance
    {
        double absolute; /**< @brief Holds the term that bounds a result near zero. */
        double relative; /**< @brief Holds the factor applied to the expected magnitude. */
    };

    /**
     * @brief Bounds a FLOAT32 result of a device kernel; 32 unit roundoffs of float absolute and 256
     * relative, the smallest powers of two that cover 1e-6 and 1e-5.
     */
    constexpr Tolerance K_FLOAT32_TOLERANCE = {32.0 * TestSupport::unit_roundoff<float>(),
                                               256.0 * TestSupport::unit_roundoff<float>()};

    /**
     * @brief Bounds a FLOAT64 result; 16384 unit roundoffs of double absolute and 2^24 relative, the
     * smallest powers of two that cover 1e-12 and 1e-9.
     */
    constexpr Tolerance K_FLOAT64_TOLERANCE = {16384.0 * TestSupport::unit_roundoff<double>(),
                                               16777216.0 * TestSupport::unit_roundoff<double>()};

    /** @brief Names a builder emitter that takes one operand. */
    using UnaryEmitter = Traced (GraphBuilder::*)(const Traced&) const;

    /** @brief Names a builder emitter that takes two operands. */
    using BinaryEmitter = Traced (GraphBuilder::*)(const Traced&, const Traced&) const;

    /** @brief Lowers a graph, runs it once, and hands back the pool. */
    class OneShot
    {
        public:
            /**
             * @brief Materialises the leaves of @p graph on @p engine.
             * @param engine Must outlive this object.
             * @param graph Must outlive this object.
             */
            OneShot(SushiBLAS::Engine& engine, const Graph::Graph& graph)
                : pool(engine), graph_(&graph), engine_(&engine)
            {
                SA_EXPECT_OK(pool.materialise_leaves(graph));
            }

            /** @brief Compiles the graph; call after every leaf has been written. */
            void compile()
            {
                SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine_, *graph_, pool));
            }

            /** @brief Compiles the graph and runs the step once. */
            void compile_and_run()
            {
                ASSERT_NO_FATAL_FAILURE(compile());
                SA_EXPECT_OK(step->run());
            }

            TensorPool pool;                    /**< @brief Holds every value's tensor. */
            std::unique_ptr<CompiledStep> step; /**< @brief Holds the step compile() built. */

        private:
            /** @brief Holds the graph that compile() lowers. */
            const Graph::Graph* graph_;

            /** @brief Holds the engine that compile() records on. */
            SushiBLAS::Engine* engine_;
    };

    /** @brief Compares every element of @p got with @p expected under @p tolerance. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         const Tolerance& tolerance)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const double bound = tolerance.absolute + tolerance.relative * std::abs(expected[i]);
            EXPECT_NEAR(got[i], expected[i], bound) << "at " << i;
        }
    }

    /**
     * @brief Runs one single-operand operation over a FLOAT32 row and reads its result.
     * @param input The operand's elements; the row is as long as this vector.
     * @param result Receives the result's elements.
     */
    void run_unary(SushiBLAS::Engine& engine, UnaryEmitter emit, const std::vector<double>& input,
                   std::vector<double>& result)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const int64_t columns = static_cast<int64_t>(input.size());
        const ValueID x = builder.input(Shape{1, columns}, "x");
        SA_ASSERT_OK_AND_ASSIGN(const ValueID y, (builder.*emit)(x));

        OneShot run(engine, graph);
        write_tensor(run.pool.tensor(x), input);

        ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

        result = read_tensor(run.pool.tensor(y));

        // An out-of-place lowering leaves its operand as it was written.
        expect_elements(read_tensor(run.pool.tensor(x)), input, Tolerance{0.0, 0.0});
    }

    /**
     * @brief Runs one two-operand operation over two FLOAT32 rows and reads its result.
     * @param left The first operand's elements; @p right must be as long.
     * @param result Receives the result's elements.
     */
    void run_binary(SushiBLAS::Engine& engine, BinaryEmitter emit,
                    const std::vector<double>& left, const std::vector<double>& right,
                    std::vector<double>& result)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const int64_t columns = static_cast<int64_t>(left.size());
        const ValueID a = builder.input(Shape{1, columns}, "a");
        const ValueID b = builder.input(Shape{1, columns}, "b");
        SA_ASSERT_OK_AND_ASSIGN(const ValueID y, (builder.*emit)(a, b));

        OneShot run(engine, graph);
        write_tensor(run.pool.tensor(a), left);
        write_tensor(run.pool.tensor(b), right);

        ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

        result = read_tensor(run.pool.tensor(y));
    }

    /**
     * @brief Differentiates loss = -sum(f(x) * targets) / 2 at a FLOAT64 [2, 3] point.
     * @param activation Emits f; its backward rule supplies the gradient node under test.
     * @param point Six elements of x; @p targets holds the six weights of the loss.
     * @param gradient Receives dloss/dx, which is -targets[i] / 2 * f'(point[i]).
     */
    void run_activation_gradient(SushiBLAS::Engine& engine, UnaryEmitter activation,
                                 const std::vector<double>& point,
                                 const std::vector<double>& targets,
                                 std::vector<double>& gradient)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.parameter(Shape{2, 3}, "x", DType::Float64);
        const ValueID weights = builder.input(Shape{2, 3}, "targets", DType::Float64);
        SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                                builder.nll_loss((builder.*activation)(x), weights));

        const std::array<ValueID, 1> wrt = {x};
        const std::vector<ValueID> gradients = Autograd::differentiate(
            graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();

        ASSERT_EQ(gradients.size(), 1u);
        ASSERT_NE(gradients[0], Graph::K_NO_VALUE);

        OneShot run(engine, graph);
        write_tensor(run.pool.tensor(x), point);
        write_tensor(run.pool.tensor(weights), targets);

        ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

        gradient = read_tensor(run.pool.tensor(gradients[0]));
    }

    /** @brief Returns the logistic function of @p x, computed on the host. */
    double host_sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    /** @brief Returns @p count values of the pattern ((i * stride) % modulus) - offset. */
    std::vector<double> patterned(std::size_t count, std::size_t stride, std::size_t modulus,
                                  double offset)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<double>((i * stride) % modulus) - offset;
        return values;
    }

    /** @brief Returns @p function applied to every element of @p values. */
    template <typename Function>
    std::vector<double> mapped(const std::vector<double>& values, Function function)
    {
        std::vector<double> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            result[i] = function(values[i]);
        return result;
    }

    /** @brief Returns @p function applied to every pair of elements of @p left and @p right. */
    template <typename Function>
    std::vector<double> zipped(const std::vector<double>& left, const std::vector<double>& right,
                               Function function)
    {
        std::vector<double> result(left.size());
        for (std::size_t i = 0; i < left.size(); ++i)
            result[i] = function(left[i], right[i]);
        return result;
    }

    /**
     * @brief Returns the 3 x 3 depthwise convolution of a [4, 4, 2] image with a [3, 3, 2] kernel
     *        and one pixel of padding at the top and left, taken on the host.
     *
     * Each output sums the taps whose input pixel lies inside the image; the rest add nothing.
     */
    std::vector<double> host_depthwise_conv(const std::vector<double>& image,
                                            const std::vector<double>& taps)
    {
        constexpr int64_t K_HEIGHT = 4;
        constexpr int64_t K_WIDTH = 4;
        constexpr int64_t K_CHANNELS = 2;
        constexpr int64_t K_KERNEL = 3;
        constexpr int64_t K_PAD = 1;

        const auto pixel = [](int64_t row, int64_t col, int64_t c)
        {
            return static_cast<std::size_t>((row * K_WIDTH + col) * K_CHANNELS + c);
        };
        const auto tap = [](int64_t r, int64_t s, int64_t c)
        {
            return static_cast<std::size_t>((r * K_KERNEL + s) * K_CHANNELS + c);
        };

        std::vector<double> result(image.size(), 0.0);
        for (int64_t row = 0; row < K_HEIGHT; ++row)
        {
            for (int64_t col = 0; col < K_WIDTH; ++col)
            {
                for (int64_t c = 0; c < K_CHANNELS; ++c)
                {
                    for (int64_t r = 0; r < K_KERNEL; ++r)
                    {
                        for (int64_t s = 0; s < K_KERNEL; ++s)
                        {
                            const int64_t in_row = row + r - K_PAD;
                            const int64_t in_col = col + s - K_PAD;
                            if (in_row < 0 || in_row >= K_HEIGHT || in_col < 0
                                || in_col >= K_WIDTH)
                            {
                                continue;
                            }

                            result[pixel(row, col, c)] +=
                                image[pixel(in_row, in_col, c)] * taps[tap(r, s, c)];
                        }
                    }
                }
            }
        }
        return result;
    }
}

/** @brief Tests that DIV divides elementwise over both signs of each operand. */
TEST_F(Medium_LoweredOperations, DivDividesElementwise)
{
    const std::vector<double> dividend = {-6.0, 0.0, 3.0, 1.0, -7.5, 9.0};
    const std::vector<double> divisor = {2.0, -4.0, -1.5, 8.0, -2.5, 3.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::div, dividend, divisor, result));

    expect_elements(result, zipped(dividend, divisor, [](double a, double b) { return a / b; }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that MINIMUM takes the smaller element whichever operand holds it. */
TEST_F(Medium_LoweredOperations, MinimumTakesTheSmallerOperand)
{
    const std::vector<double> left = {-2.0, 0.0, 3.0, 1.5, -1.0, 4.0};
    const std::vector<double> right = {1.0, 0.0, -3.0, 1.5, -4.0, 5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::minimum, left, right, result));

    expect_elements(result, zipped(left, right, [](double a, double b) { return std::min(a, b); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that MAXIMUM takes the larger element whichever operand holds it. */
TEST_F(Medium_LoweredOperations, MaximumTakesTheLargerOperand)
{
    const std::vector<double> left = {-2.0, 0.0, 3.0, 1.5, -1.0, 4.0};
    const std::vector<double> right = {1.0, 0.0, -3.0, 1.5, -4.0, 5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::maximum, left, right, result));

    expect_elements(result, zipped(left, right, [](double a, double b) { return std::max(a, b); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that ABS returns the magnitude of negative, zero and positive elements. */
TEST_F(Medium_LoweredOperations, AbsReturnsTheMagnitude)
{
    const std::vector<double> input = {-2.5, -0.125, 0.0, 0.125, 3.0, 1024.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::abs, input, result));

    expect_elements(result, mapped(input, [](double v) { return std::abs(v); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that ATAN returns the arctangent on both sides of zero and near its limit. */
TEST_F(Medium_LoweredOperations, AtanReturnsTheArctangent)
{
    const std::vector<double> input = {-64.0, -1.0, -0.25, 0.0, 0.25, 1.0, 64.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::atan, input, result));

    expect_elements(result, mapped(input, [](double v) { return std::atan(v); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that LESS_EQUAL writes 1 for less and for equal, and 0 for greater. */
TEST_F(Medium_LoweredOperations, LessEqualWritesOneForLessAndForEqual)
{
    const std::vector<double> left = {-1.0, 0.0, 2.0, 3.0, -2.0, 5.0};
    const std::vector<double> right = {0.0, 0.0, 2.0, 1.0, -2.0, -5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(
        run_binary(*engine, &GraphBuilder::less_equal, left, right, result));

    expect_elements(result,
                    zipped(left, right, [](double a, double b) { return a <= b ? 1.0 : 0.0; }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SELECT takes the first branch where the mask is 1 and the second at 0. */
TEST_F(Medium_LoweredOperations, SelectTakesEachBranchWhereTheMaskSaysSo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID condition = builder.input(Shape{1, 6}, "condition");
    const ValueID a = builder.input(Shape{1, 6}, "a");
    const ValueID b = builder.input(Shape{1, 6}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.select(condition, a, b));

    const std::vector<double> mask = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const std::vector<double> taken = {-1.0, 2.0, 0.0, 4.0, -5.0, 6.0};
    const std::vector<double> other = {10.0, -20.0, 30.0, 0.0, 50.0, -60.0};

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(condition), mask);
    write_tensor(run.pool.tensor(a), taken);
    write_tensor(run.pool.tensor(b), other);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

    // Where the mask is 1 the first branch, where it is 0 the second.
    expect_elements(read_tensor(run.pool.tensor(y)),
                    std::vector<double>{-1.0, -20.0, 0.0, 0.0, 50.0, 6.0}, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that CONSTANT fills every element with its attribute, on each run. */
TEST_F(Medium_LoweredOperations, ConstantFillsEveryElementWithItsValue)
{
    constexpr double K_FILL = -1.75;
    constexpr double K_LIKE_FILL = 0.5;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID filled, builder.constant(Shape{4}, K_FILL));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID like, builder.constant_like(x, K_LIKE_FILL));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID shifted, builder.add(x, like));

    const std::vector<double> input = {-2.0, -0.5, 0.0, 0.5, 2.0, 8.0};

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), input);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(filled)), std::vector<double>(4, K_FILL),
                    K_FLOAT32_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(like)), std::vector<double>(6, K_LIKE_FILL),
                    K_FLOAT32_TOLERANCE);

    expect_elements(read_tensor(run.pool.tensor(shifted)),
                    mapped(input, [](double v) { return v + K_LIKE_FILL; }), K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SILU returns x * sigmoid(x) on both sides of zero. */
TEST_F(Medium_LoweredOperations, SiLUMatchesItsDefinition)
{
    const std::vector<double> input = {-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::silu, input, result));

    expect_elements(result, mapped(input, [](double v) { return v * host_sigmoid(v); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SOFTPLUS returns log(1 + exp(x)), including where it nears 0 and x. */
TEST_F(Medium_LoweredOperations, SoftplusMatchesItsDefinition)
{
    const std::vector<double> input = {-20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 20.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::softplus, input, result));

    expect_elements(result, mapped(input, [](double v) { return std::log1p(std::exp(v)); }),
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests SILU_BACKWARD against sigmoid(x) * (1 + x * (1 - sigmoid(x))). */
TEST_F(Medium_LoweredOperations, SiLUBackwardMatchesTheClosedFormDerivative)
{
    const std::vector<double> point = {-2.0, -0.5, 0.0, 0.5, 2.0, 4.0};
    const std::vector<double> targets = {1.0, -2.0, 0.5, 3.0, -1.0, 2.0};

    std::vector<double> gradient;
    ASSERT_NO_FATAL_FAILURE(
        run_activation_gradient(*engine, &GraphBuilder::silu, point, targets, gradient));

    const std::vector<double> derivative = mapped(point, [](double v)
    {
        const double s = host_sigmoid(v);
        return s * (1.0 + v * (1.0 - s));
    });
    expect_elements(gradient,
                    zipped(targets, derivative, [](double t, double d) { return -t / 2.0 * d; }),
                    K_FLOAT64_TOLERANCE);
}

/** @brief Tests SOFTPLUS_BACKWARD against sigmoid(x), the derivative of log(1 + exp(x)). */
TEST_F(Medium_LoweredOperations, SoftplusBackwardMatchesTheClosedFormDerivative)
{
    const std::vector<double> point = {-2.0, -0.5, 0.0, 0.5, 2.0, 4.0};
    const std::vector<double> targets = {1.0, -2.0, 0.5, 3.0, -1.0, 2.0};

    std::vector<double> gradient;
    ASSERT_NO_FATAL_FAILURE(
        run_activation_gradient(*engine, &GraphBuilder::softplus, point, targets, gradient));

    expect_elements(gradient,
                    zipped(targets, mapped(point, host_sigmoid),
                           [](double t, double d) { return -t / 2.0 * d; }),
                    K_FLOAT64_TOLERANCE);
}

/** @brief Tests that STOP_GRADIENT copies its operand's elements unchanged. */
TEST_F(Medium_LoweredOperations, StopGradientCopiesItsOperand)
{
    const std::vector<double> input = {-3.5, -0.25, 0.0, 0.25, 7.0, 4096.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::stop_gradient, input, result));

    EXPECT_EQ(result, input);
}

/** @brief Tests that BATCHNORM_FOLD writes gamma / sqrt(variance + eps) and its shift. */
TEST_F(Medium_LoweredOperations, BatchNormFoldWritesTheScaleAndTheShift)
{
    constexpr double K_EPSILON = 1e-3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID gamma = builder.input(Shape{4}, "gamma");
    const ValueID beta = builder.input(Shape{4}, "beta");
    const ValueID mean = builder.input(Shape{4}, "mean");
    const ValueID variance = builder.input(Shape{4}, "variance");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::BatchNormFold fold,
                            builder.batchnorm_fold(gamma, beta, mean, variance, K_EPSILON));

    // The zero variance leaves epsilon alone under the root.
    const std::vector<double> gammas = {2.0, -1.5, 0.0, 0.75};
    const std::vector<double> betas = {0.5, -2.0, 3.0, 0.0};
    const std::vector<double> means = {-1.0, 0.0, 4.0, 2.5};
    const std::vector<double> variances = {0.0, 0.25, 9.0, 1.0};

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(gamma), gammas);
    write_tensor(run.pool.tensor(beta), betas);
    write_tensor(run.pool.tensor(mean), means);
    write_tensor(run.pool.tensor(variance), variances);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

    const std::vector<double> expected_scale =
        zipped(gammas, variances, [](double g, double v) { return g / std::sqrt(v + K_EPSILON); });
    const std::vector<double> expected_shift =
        zipped(betas, zipped(means, expected_scale, [](double m, double s) { return m * s; }),
               [](double b, double scaled_mean) { return b - scaled_mean; });

    expect_elements(read_tensor(run.pool.tensor(fold.scale)), expected_scale,
                    K_FLOAT32_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(fold.shift)), expected_shift,
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that DEPTHWISE_CONV2D equals the sums over a padded 3x3 window taken on the host. */
TEST_F(Medium_LoweredOperations, DepthwiseConv2dMatchesTheWindowSums)
{
    constexpr int64_t K_HEIGHT = 4;
    constexpr int64_t K_WIDTH = 4;
    constexpr int64_t K_CHANNELS = 2;
    constexpr int64_t K_KERNEL = 3;
    constexpr int64_t K_PAD = 1;

    // Integer-valued data, so every product and sum is exact in FLOAT32.
    const std::vector<double> image =
        patterned(static_cast<std::size_t>(K_HEIGHT * K_WIDTH * K_CHANNELS), 7, 11, 5.0);
    const std::vector<double> taps =
        patterned(static_cast<std::size_t>(K_KERNEL * K_KERNEL * K_CHANNELS), 5, 7, 3.0);

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, K_HEIGHT, K_WIDTH, K_CHANNELS}, "x");
    const ValueID weights = builder.input(Shape{K_KERNEL, K_KERNEL, K_CHANNELS}, "taps");

    Graph::ConvGeometry g;
    g.kernel_h = K_KERNEL;
    g.kernel_w = K_KERNEL;
    g.pad_top = K_PAD;
    g.pad_left = K_PAD;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.depthwise_conv2d(x, weights, g));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), image);
    write_tensor(run.pool.tensor(weights), taps);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

    const Tolerance window_tolerance = {K_FLOAT32_TOLERANCE.absolute, 0.0};
    expect_elements(read_tensor(run.pool.tensor(y)), host_depthwise_conv(image, taps),
                    window_tolerance);
}

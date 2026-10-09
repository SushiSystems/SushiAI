/**************************************************************************/
/* test_lowered_operations.cpp                                            */
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
 * @file test_lowered_operations.cpp
 * @brief Tests the fourteen lowerings the SushiBLAS migration added against host values.
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
#include <SushiAI/autograd/backward.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every lowered-operation case records on. */
    class Integration_LoweredOperations : public SaTest::EngineTest
    {

    };

    /** @brief Holds the bound |got - expected| <= absolute + relative * |expected|. */
    struct Tolerance
    {
        double absolute; /**< @brief Holds the term that bounds a result near zero. */
        double relative; /**< @brief Holds the factor applied to the expected magnitude. */
    };

    /** @brief Bounds a FLOAT32 result: a few units in the last place of a device kernel. */
    constexpr Tolerance K_FLOAT32_TOLERANCE = {1e-6, 1e-5};

    /** @brief Bounds a FLOAT64 result. */
    constexpr Tolerance K_FLOAT64_TOLERANCE = {1e-12, 1e-9};

    /** @brief Names a builder emitter that takes one operand. */
    using UnaryEmitter = ValueId (GraphBuilder::*)(ValueId) const;

    /** @brief Names a builder emitter that takes two operands. */
    using BinaryEmitter = ValueId (GraphBuilder::*)(ValueId, ValueId) const;

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
        const ValueId x = builder.input(Shape{1, columns}, "x");
        const ValueId y = (builder.*emit)(x);

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
        const ValueId a = builder.input(Shape{1, columns}, "a");
        const ValueId b = builder.input(Shape{1, columns}, "b");
        const ValueId y = (builder.*emit)(a, b);

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

        const ValueId x = builder.parameter(Shape{2, 3}, "x", DType::Float64);
        const ValueId weights = builder.input(Shape{2, 3}, "targets", DType::Float64);
        const ValueId loss = builder.nll_loss((builder.*activation)(x), weights);

        const std::array<ValueId, 1> wrt = {x};
        const std::vector<ValueId> gradients = Autograd::differentiate(
            graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

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
}

/** @brief Tests that DIV divides elementwise over both signs of each operand. */
TEST_F(Integration_LoweredOperations, DivDividesElementwise)
{
    const std::vector<double> dividend = {-6.0, 0.0, 3.0, 1.0, -7.5, 9.0};
    const std::vector<double> divisor = {2.0, -4.0, -1.5, 8.0, -2.5, 3.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::div, dividend, divisor, result));

    std::vector<double> expected(dividend.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = dividend[i] / divisor[i];

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that MINIMUM takes the smaller element whichever operand holds it. */
TEST_F(Integration_LoweredOperations, MinimumTakesTheSmallerOperand)
{
    const std::vector<double> left = {-2.0, 0.0, 3.0, 1.5, -1.0, 4.0};
    const std::vector<double> right = {1.0, 0.0, -3.0, 1.5, -4.0, 5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::minimum, left, right, result));

    std::vector<double> expected(left.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::min(left[i], right[i]);

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that MAXIMUM takes the larger element whichever operand holds it. */
TEST_F(Integration_LoweredOperations, MaximumTakesTheLargerOperand)
{
    const std::vector<double> left = {-2.0, 0.0, 3.0, 1.5, -1.0, 4.0};
    const std::vector<double> right = {1.0, 0.0, -3.0, 1.5, -4.0, 5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_binary(*engine, &GraphBuilder::maximum, left, right, result));

    std::vector<double> expected(left.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::max(left[i], right[i]);

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that ABS returns the magnitude of negative, zero and positive elements. */
TEST_F(Integration_LoweredOperations, AbsReturnsTheMagnitude)
{
    const std::vector<double> input = {-2.5, -0.125, 0.0, 0.125, 3.0, 1024.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::abs, input, result));

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::abs(input[i]);

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that ATAN returns the arctangent on both sides of zero and near its limit. */
TEST_F(Integration_LoweredOperations, AtanReturnsTheArctangent)
{
    const std::vector<double> input = {-64.0, -1.0, -0.25, 0.0, 0.25, 1.0, 64.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::atan, input, result));

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::atan(input[i]);

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that LESS_EQUAL writes 1 for less and for equal, and 0 for greater. */
TEST_F(Integration_LoweredOperations, LessEqualWritesOneForLessAndForEqual)
{
    const std::vector<double> left = {-1.0, 0.0, 2.0, 3.0, -2.0, 5.0};
    const std::vector<double> right = {0.0, 0.0, 2.0, 1.0, -2.0, -5.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(
        run_binary(*engine, &GraphBuilder::less_equal, left, right, result));

    std::vector<double> expected(left.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = left[i] <= right[i] ? 1.0 : 0.0;

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SELECT takes the first branch where the mask is 1 and the second at 0. */
TEST_F(Integration_LoweredOperations, SelectTakesEachBranchWhereTheMaskSaysSo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId condition = builder.input(Shape{1, 6}, "condition");
    const ValueId a = builder.input(Shape{1, 6}, "a");
    const ValueId b = builder.input(Shape{1, 6}, "b");
    const ValueId y = builder.select(condition, a, b);

    const std::vector<double> mask = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const std::vector<double> taken = {-1.0, 2.0, 0.0, 4.0, -5.0, 6.0};
    const std::vector<double> other = {10.0, -20.0, 30.0, 0.0, 50.0, -60.0};

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(condition), mask);
    write_tensor(run.pool.tensor(a), taken);
    write_tensor(run.pool.tensor(b), other);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

    std::vector<double> expected(mask.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = mask[i] != 0.0 ? taken[i] : other[i];

    expect_elements(read_tensor(run.pool.tensor(y)), expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that CONSTANT fills every element with its attribute, on each run. */
TEST_F(Integration_LoweredOperations, ConstantFillsEveryElementWithItsValue)
{
    constexpr double K_FILL = -1.75;
    constexpr double K_LIKE_FILL = 0.5;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 3}, "x");
    const ValueId filled = builder.constant(Shape{4}, K_FILL);
    const ValueId like = builder.constant_like(x, K_LIKE_FILL);
    const ValueId shifted = builder.add(x, like);

    const std::vector<double> input = {-2.0, -0.5, 0.0, 0.5, 2.0, 8.0};

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), input);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(filled)), std::vector<double>(4, K_FILL),
                    K_FLOAT32_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(like)), std::vector<double>(6, K_LIKE_FILL),
                    K_FLOAT32_TOLERANCE);

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = input[i] + K_LIKE_FILL;

    expect_elements(read_tensor(run.pool.tensor(shifted)), expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SILU returns x * sigmoid(x) on both sides of zero. */
TEST_F(Integration_LoweredOperations, SiluMatchesItsDefinition)
{
    const std::vector<double> input = {-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::silu, input, result));

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = input[i] * host_sigmoid(input[i]);

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that SOFTPLUS returns log(1 + exp(x)), including where it nears 0 and x. */
TEST_F(Integration_LoweredOperations, SoftplusMatchesItsDefinition)
{
    const std::vector<double> input = {-20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 20.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::softplus, input, result));

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = std::log1p(std::exp(input[i]));

    expect_elements(result, expected, K_FLOAT32_TOLERANCE);
}

/** @brief Tests SILU_BACKWARD against sigmoid(x) * (1 + x * (1 - sigmoid(x))). */
TEST_F(Integration_LoweredOperations, SiluBackwardMatchesTheClosedFormDerivative)
{
    const std::vector<double> point = {-2.0, -0.5, 0.0, 0.5, 2.0, 4.0};
    const std::vector<double> targets = {1.0, -2.0, 0.5, 3.0, -1.0, 2.0};

    std::vector<double> gradient;
    ASSERT_NO_FATAL_FAILURE(
        run_activation_gradient(*engine, &GraphBuilder::silu, point, targets, gradient));

    std::vector<double> expected(point.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double s = host_sigmoid(point[i]);
        expected[i] = -targets[i] / 2.0 * s * (1.0 + point[i] * (1.0 - s));
    }

    expect_elements(gradient, expected, K_FLOAT64_TOLERANCE);
}

/** @brief Tests SOFTPLUS_BACKWARD against sigmoid(x), the derivative of log(1 + exp(x)). */
TEST_F(Integration_LoweredOperations, SoftplusBackwardMatchesTheClosedFormDerivative)
{
    const std::vector<double> point = {-2.0, -0.5, 0.0, 0.5, 2.0, 4.0};
    const std::vector<double> targets = {1.0, -2.0, 0.5, 3.0, -1.0, 2.0};

    std::vector<double> gradient;
    ASSERT_NO_FATAL_FAILURE(
        run_activation_gradient(*engine, &GraphBuilder::softplus, point, targets, gradient));

    std::vector<double> expected(point.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = -targets[i] / 2.0 * host_sigmoid(point[i]);

    expect_elements(gradient, expected, K_FLOAT64_TOLERANCE);
}

/** @brief Tests that STOP_GRADIENT copies its operand's elements unchanged. */
TEST_F(Integration_LoweredOperations, StopGradientCopiesItsOperand)
{
    const std::vector<double> input = {-3.5, -0.25, 0.0, 0.25, 7.0, 4096.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_unary(*engine, &GraphBuilder::stop_gradient, input, result));

    expect_elements(result, input, K_FLOAT32_TOLERANCE);
}

/** @brief Tests that BATCHNORM_FOLD writes gamma / sqrt(variance + eps) and its shift. */
TEST_F(Integration_LoweredOperations, BatchNormFoldWritesTheScaleAndTheShift)
{
    constexpr double K_EPSILON = 1e-3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId gamma = builder.input(Shape{4}, "gamma");
    const ValueId beta = builder.input(Shape{4}, "beta");
    const ValueId mean = builder.input(Shape{4}, "mean");
    const ValueId variance = builder.input(Shape{4}, "variance");
    const Graph::BatchNormFold fold =
        builder.batchnorm_fold(gamma, beta, mean, variance, K_EPSILON);

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

    std::vector<double> expected_scale(gammas.size());
    std::vector<double> expected_shift(gammas.size());
    for (std::size_t i = 0; i < gammas.size(); ++i)
    {
        expected_scale[i] = gammas[i] / std::sqrt(variances[i] + K_EPSILON);
        expected_shift[i] = betas[i] - means[i] * expected_scale[i];
    }

    expect_elements(read_tensor(run.pool.tensor(fold.scale)), expected_scale,
                    K_FLOAT32_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(fold.shift)), expected_shift,
                    K_FLOAT32_TOLERANCE);
}

/** @brief Tests that DEPTHWISE_CONV2D matches three nested loops over a padded 3x3 window. */
TEST_F(Integration_LoweredOperations, DepthwiseConv2dMatchesTheWindowSums)
{
    constexpr int64_t K_HEIGHT = 4;
    constexpr int64_t K_WIDTH = 4;
    constexpr int64_t K_CHANNELS = 2;
    constexpr int64_t K_KERNEL = 3;
    constexpr int64_t K_PAD = 1;

    // Integer-valued data, so every product and sum is exact in FLOAT32.
    std::vector<double> image(static_cast<std::size_t>(K_HEIGHT * K_WIDTH * K_CHANNELS));
    for (std::size_t i = 0; i < image.size(); ++i)
    {
        image[i] = static_cast<double>((i * 7) % 11) - 5.0;
    }

    std::vector<double> taps(static_cast<std::size_t>(K_KERNEL * K_KERNEL * K_CHANNELS));
    for (std::size_t i = 0; i < taps.size(); ++i)
    {
        taps[i] = static_cast<double>((i * 5) % 7) - 3.0;
    }

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, K_HEIGHT, K_WIDTH, K_CHANNELS}, "x");
    const ValueId weights = builder.input(Shape{K_KERNEL, K_KERNEL, K_CHANNELS}, "taps");

    Graph::ConvGeometry g;
    g.kernel_h = K_KERNEL;
    g.kernel_w = K_KERNEL;
    g.pad_top = K_PAD;
    g.pad_left = K_PAD;
    const ValueId y = builder.depthwise_conv2d(x, weights, g);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), image);
    write_tensor(run.pool.tensor(weights), taps);

    ASSERT_NO_FATAL_FAILURE(run.compile_and_run());

    const auto pixel = [](int64_t row, int64_t col, int64_t c)
    {
        return static_cast<std::size_t>((row * K_WIDTH + col) * K_CHANNELS + c);
    };
    const auto tap = [](int64_t r, int64_t s, int64_t c)
    {
        return static_cast<std::size_t>((r * K_KERNEL + s) * K_CHANNELS + c);
    };

    // Each output sums the taps whose input pixel lies inside the image; the rest add nothing.
    std::vector<double> expected(image.size(), 0.0);
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
                        if (in_row < 0 || in_row >= K_HEIGHT || in_col < 0 || in_col >= K_WIDTH)
                        {
                            continue;
                        }

                        expected[pixel(row, col, c)] +=
                            image[pixel(in_row, in_col, c)] * taps[tap(r, s, c)];
                    }
                }
            }
        }
    }

    const Tolerance window_tolerance = {1e-6, 0.0};
    expect_elements(read_tensor(run.pool.tensor(y)), expected, window_tolerance);
}

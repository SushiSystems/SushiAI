/**************************************************************************/
/* test_layer_norm.cpp                                                    */
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
 * @file test_layer_norm.cpp
 * @brief Checks LAYER_NORM and the LayerNorm layer on an engine against a host loop, and the
 *        gradients of x, gamma and beta against central differences.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every layer-normalisation case records on. */
    class Integration_LayerNorm : public SaTest::EngineTest
    {

    };

    /** @brief Holds the epsilon every case passes. */
    constexpr double K_EPSILON = 1e-5;

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_FLOAT32_TOLERANCE = 2e-5;

    /** @brief Bounds a FLOAT64 result computed on the device against a double host value. */
    constexpr double K_FLOAT64_TOLERANCE = 1e-12;

    /** @brief Holds the row length of the value the gradient cases normalise. */
    constexpr int64_t K_WIDTH = 4;

    /** @brief Holds the row count of the value the gradient cases normalise, 2 by 3. */
    constexpr int64_t K_ROWS = 6;

    /**
     * @brief Returns the layer normalisation of @p x on the host, in double.
     * @param width The row length; @p x holds whole rows, and gamma and beta one row each.
     */
    std::vector<double> normalised_on_host(const std::vector<double>& x,
                                           const std::vector<double>& gamma,
                                           const std::vector<double>& beta, std::size_t width)
    {
        std::vector<double> out(x.size());
        for (std::size_t base = 0; base < x.size(); base += width)
        {
            double mean = 0.0;
            for (std::size_t c = 0; c < width; ++c)
                mean += x[base + c];
            mean /= static_cast<double>(width);

            double variance = 0.0;
            for (std::size_t c = 0; c < width; ++c)
                variance += (x[base + c] - mean) * (x[base + c] - mean);
            variance /= static_cast<double>(width);

            const double inv_std = 1.0 / std::sqrt(variance + K_EPSILON);
            for (std::size_t c = 0; c < width; ++c)
                out[base + c] = (x[base + c] - mean) * inv_std * gamma[c] + beta[c];
        }

        return out;
    }

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at element " << i;
    }

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     *
     * The loss is the cross-entropy of what the build function returns, an [N, C] value, and
     * the variable is the one leaf the gradient is taken at.
     */
    class Probe
    {
        public:
            /** @brief Builds the graph over a variable of @p variable_shape and compiles it. */
            template<typename BuildFn>
            Probe(SushiBLAS::Engine& engine, Shape variable_shape, BuildFn&& build)
                : pool_(engine)
            {
                const GraphBuilder builder(graph_);
                variable_ = builder.parameter(variable_shape, "variable", DType::Float64);

                const ValueID logits = build(builder, variable_);
                const Shape logit_shape = graph_.value(logits).shape;
                targets_ = builder.input(logit_shape, "targets", DType::Float64);
                loss_ = builder.nll_loss(builder.log_softmax(logits), targets_);

                const std::vector<ValueID> wrt = {variable_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();

                // Row r of the targets names class r mod C, so every class is a target.
                std::vector<double> one_hot(
                    static_cast<std::size_t>(logit_shape.num_elements()), 0.0);
                for (int64_t row = 0; row < logit_shape[0]; ++row)
                    one_hot[static_cast<std::size_t>(row * logit_shape[1]
                                                     + row % logit_shape[1])] = 1.0;
                write_tensor(pool_.tensor(targets_), one_hot);
            }

            /** @brief Writes the values of a leaf that is not the variable. */
            void fill(ValueID id, const std::vector<double>& values)
            {
                write_tensor(pool_.tensor(id), values);
            }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID variable_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Runs the full-Jacobian check of @p probe at @p point and reports a failure. */
    void expect_gradient_matches(Probe& probe, std::vector<double>& point)
    {
        const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at)
            {
                return probe.value(at);
            },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            {
                probe.gradient(at, out);
            },
            point);

        EXPECT_TRUE(result.passed) << result.to_string();
    }

    /** @brief Returns the [K_ROWS, K_WIDTH] rows of a normalised [2, 3, K_WIDTH] value. */
    ValueID normalised_rows(const GraphBuilder& builder, ValueID x, ValueID gamma, ValueID beta)
    {
        const Graph::LayerNormResult out = builder.layer_norm(x, gamma, beta, K_EPSILON).value();
        return builder.reshape(out.output, Shape{K_ROWS, K_WIDTH});
    }
}

/** @brief Tests that a FLOAT32 [2, 3, 5] value is normalised row by row, as the host does. */
TEST_F(Integration_LayerNorm, EachRowIsNormalisedByItsOwnStatistics)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3, 5}, "x");
    const ValueID gamma = builder.input(Shape{5}, "gamma");
    const ValueID beta = builder.input(Shape{5}, "beta");
    const Result<Graph::LayerNormResult> out = builder.layer_norm(x, gamma, beta, K_EPSILON);
    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());

    const std::vector<double> values = noise(2 * 3 * 5, 5101);
    const std::vector<double> gain = noise(5, 5102);
    const std::vector<double> bias = noise(5, 5103);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(gamma), gain);
    write_tensor(pool.tensor(beta), bias);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));

    // The second run proves a replay starts from the operand and not from the first result.
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());

    expect_elements(read_tensor(pool.tensor(out.value().output)),
                    normalised_on_host(values, gain, bias, 5), K_FLOAT32_TOLERANCE);
}

/** @brief Tests that the saved statistics are each row's mean and 1 / sqrt(variance + eps). */
TEST_F(Integration_LayerNorm, TheSavedStatisticsAreTheRowMeanAndInverseDeviation)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 4}, "x", DType::Float64);
    const ValueID gamma = builder.input(Shape{4}, "gamma", DType::Float64);
    const ValueID beta = builder.input(Shape{4}, "beta", DType::Float64);
    const Result<Graph::LayerNormResult> out = builder.layer_norm(x, gamma, beta, K_EPSILON);
    ASSERT_TRUE(out.has_value());

    // Row 0 has mean 2.5 and variance 1.25; row 1 has mean -1 and variance 5.
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, -4.0, -2.0, 0.0, 2.0};

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(gamma), std::vector<double>(4, 1.0));
    write_tensor(pool.tensor(beta), std::vector<double>(4, 0.0));

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    expect_elements(read_tensor(pool.tensor(out.value().saved_mean)), {2.5, -1.0},
                    K_FLOAT64_TOLERANCE);
    expect_elements(read_tensor(pool.tensor(out.value().saved_inv_std)),
                    {1.0 / std::sqrt(1.25 + K_EPSILON), 1.0 / std::sqrt(5.0 + K_EPSILON)},
                    K_FLOAT64_TOLERANCE);
}

/** @brief Tests the gradient of the input against central differences, in FLOAT64. */
TEST_F(Integration_LayerNorm, TheInputGradientMatchesCentralDifferences)
{
    ValueID gamma = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{2, 3, K_WIDTH}, [&](const GraphBuilder& builder, ValueID x)
    {
        gamma = builder.parameter(Shape{K_WIDTH}, "gamma", DType::Float64);
        beta = builder.parameter(Shape{K_WIDTH}, "beta", DType::Float64);
        return normalised_rows(builder, x, gamma, beta);
    });

    probe.fill(gamma, noise(K_WIDTH, 5201));
    probe.fill(beta, noise(K_WIDTH, 5202));

    std::vector<double> point = noise(K_ROWS * K_WIDTH, 5203);
    expect_gradient_matches(probe, point);
}

/** @brief Tests the gradient of the gain against central differences, in FLOAT64. */
TEST_F(Integration_LayerNorm, TheGainGradientMatchesCentralDifferences)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{K_WIDTH}, [&](const GraphBuilder& builder, ValueID gamma)
    {
        x = builder.input(Shape{2, 3, K_WIDTH}, "x", DType::Float64);
        beta = builder.parameter(Shape{K_WIDTH}, "beta", DType::Float64);
        return normalised_rows(builder, x, gamma, beta);
    });

    probe.fill(x, noise(K_ROWS * K_WIDTH, 5301));
    probe.fill(beta, noise(K_WIDTH, 5302));

    std::vector<double> point = noise(K_WIDTH, 5303);
    expect_gradient_matches(probe, point);
}

/** @brief Tests the gradient of the bias against central differences, in FLOAT64. */
TEST_F(Integration_LayerNorm, TheBiasGradientMatchesCentralDifferences)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID gamma = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{K_WIDTH}, [&](const GraphBuilder& builder, ValueID beta)
    {
        x = builder.input(Shape{2, 3, K_WIDTH}, "x", DType::Float64);
        gamma = builder.parameter(Shape{K_WIDTH}, "gamma", DType::Float64);
        return normalised_rows(builder, x, gamma, beta);
    });

    probe.fill(x, noise(K_ROWS * K_WIDTH, 5401));
    probe.fill(gamma, noise(K_WIDTH, 5402));

    std::vector<double> point = noise(K_WIDTH, 5403);
    expect_gradient_matches(probe, point);
}

/** @brief Tests that a LayerNorm layer over a [2, 3, 4] value yields the host's rows. */
TEST_F(Integration_LayerNorm, ALayerNormalisesWithItsGainAndItsBias)
{
    NN::LayerNorm layer{};
    layer.dim = 4;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3, 4}, "x");
    const NN::Trace trace = NN::Trace::evaluation(builder);
    const ValueID y = layer.forward(trace, x);
    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(graph.value(y).shape, (Shape{2, 3, 4}));

    const std::vector<double> values = noise(2 * 3 * 4, 5501);
    const std::vector<double> gain = noise(4, 5502);
    const std::vector<double> bias = noise(4, 5503);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(layer.weight.id_in(graph)), gain);
    write_tensor(pool.tensor(layer.bias.id_in(graph)), bias);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    expect_elements(read_tensor(pool.tensor(y)), normalised_on_host(values, gain, bias, 4),
                    K_FLOAT32_TOLERANCE);
}

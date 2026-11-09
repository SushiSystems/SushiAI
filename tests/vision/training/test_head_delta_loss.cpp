/**************************************************************************/
/* test_head_delta_loss.cpp                                               */
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
 * @file test_head_delta_loss.cpp
 * @brief Tests the head delta loss: its graph, and on an engine the gradient -scale * delta at the
 *        head.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;
using Vision::Training::emit_head_delta_loss;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::TensorPool;

namespace
{
    /** @brief Holds the factor the tests pass, one over a batch of four. */
    constexpr double K_SCALE = 0.25;

    /** @brief Returns a head of two samples on a 3x3 grid with seven channels, 126 elements. */
    Shape head_shape()
    {
        return Shape{2, 3, 3, 7};
    }

} // namespace

/** @brief Tests that the result is one element and its root scales by minus the factor. */
TEST(Small_HeadDeltaLoss, TheResultIsOneElementScaledByMinusTheFactor)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            emit_head_delta_loss(builder, head, delta, K_SCALE));

    EXPECT_EQ(graph.value(loss).shape, (Shape{1}));
    EXPECT_EQ(graph.value(loss).dtype, DType::Float32);
    EXPECT_EQ(graph.nodes().back().outputs, (std::vector<ValueID>{loss}));
    EXPECT_EQ(graph.nodes().back().attributes.get(0), -K_SCALE);
}

/** @brief Tests that the value on the host is minus the factor times the sum of products. */
TEST(Small_HeadDeltaLoss, TheValueOnTheHostIsMinusTheFactorTimesTheSumOfProducts)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            emit_head_delta_loss(builder, head, delta, K_SCALE));

    // 126 elements of head 2 times delta 3 sum to 756; times -1/4 is -189, exact in double.
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(head, std::vector<double>(126, 2.0)));
    SA_EXPECT_OK(evaluator.write(delta, std::vector<double>(126, 3.0)));
    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> value, evaluator.read(loss));
    EXPECT_EQ(value, (std::vector<double>{-189.0}));
}

/** @brief Tests that the result carries the element type of the head. */
TEST(Small_HeadDeltaLoss, TheResultHasTheElementTypeOfTheHead)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head", DType::Float64);
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            emit_head_delta_loss(builder, head, delta, K_SCALE));

    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that autograd takes the result as a root and returns a head-shaped gradient. */
TEST(Small_HeadDeltaLoss, TheResultIsARootAutogradDifferentiates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            emit_head_delta_loss(builder, head, delta, K_SCALE));

    const std::vector<ValueID> wrt = {head};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> gradients, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).shape, head_shape());
    EXPECT_FALSE(graph.value(delta).requires_grad);
}

/** @brief Tests that a delta of another shape is refused, naming MUL and both shapes. */
TEST(Small_HeadDeltaLoss, ADeltaOfAnotherShapeIsRefusedNamingBothShapes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(Shape{2, 3, 3, 6}, "delta");
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_FAILURE(emit_head_delta_loss(builder, head, delta, K_SCALE),
                      StatusCode::InvalidDescription, "MUL needs equally shaped operands",
                      "[2, 3, 3, 6]", "[2, 3, 3, 7]");
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that a delta of another element type is refused, naming MUL and both types. */
TEST(Small_HeadDeltaLoss, ADeltaOfAnotherElementTypeIsRefusedNamingBothTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_FAILURE(emit_head_delta_loss(builder, head, delta, K_SCALE),
                      StatusCode::InvalidDescription, "MUL needs operands of one element type",
                      "float32", "float64");
    EXPECT_EQ(graph.node_count(), leaves);
}

namespace
{
    class Medium_HeadDeltaLossEndToEnd : public SaTest::EngineTest {};

    /** @brief Holds the factor under test, one over the batch of two. */
    constexpr double K_ENGINE_SCALE = 0.5;

    /** @brief Seeds the head's values. */
    constexpr std::uint64_t K_HEAD_SEED = 20261009;

    /** @brief Seeds the delta's values. */
    constexpr std::uint64_t K_DELTA_SEED = 20261010;

    /**
     * @brief Holds the Bounds the absolute error of each float32 gradient element: 32 times the
     *        float unit roundoff, the smallest power of two not tighter than the earlier 1e-6.
     */
    constexpr double K_GRADIENT_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Holds the Bounds the absolute error of the float32 sum of 126 products below 1: 2048
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-4.
     */
    constexpr double K_VALUE_TOLERANCE = 2048.0 * TestSupport::unit_roundoff<float>();

    /** @brief Expects each element of @p actual within @p bound of the same element of @p expected. */
    void expect_elements_within(const std::vector<double>& actual,
                                const std::vector<double>& expected, double bound)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], bound) << "element " << i;
    }

    /** @brief Returns @p values each multiplied by -@p factor. */
    std::vector<double> negated_scaled(const std::vector<double>& values, double factor)
    {
        std::vector<double> scaled;
        scaled.reserve(values.size());
        for (const double value : values)
            scaled.push_back(-factor * value);
        return scaled;
    }

    /** @brief Holds the graph of one head and its delta, compiled, with host copies of both. */
    struct DeltaLossCase
    {
        /**
         * @brief Builds, differentiates and compiles the loss, then writes the seeded leaves.
         * @param engine Must outlive this object.
         */
        explicit DeltaLossCase(SushiAI::ExecutionBackend& backend) : pool(backend.engine())
        {
            const GraphBuilder builder(graph);
            const Shape shape{2, 3, 3, 7};

            head = builder.parameter(shape, "head");
            delta = builder.input(shape, "delta");
            // The head and the delta share one shape and type, so the builder accepts both.
            loss = emit_head_delta_loss(builder, head, delta, K_ENGINE_SCALE).value();

            const std::vector<ValueID> wrt = {head};
            // The graph is fixed here, so a refusal is a defect: value() stops the test.
            gradient = Autograd::differentiate(
                           graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()))
                           .value()
                           .at(0);

            SA_EXPECT_OK(pool.materialise_leaves(graph));
            Result<std::unique_ptr<CompiledStep>> compiled =
                CompiledStep::compile(backend, graph, pool);
            SA_EXPECT_OK(compiled);
            if (compiled.has_value())
                step = std::move(compiled).value();

            const std::size_t count = static_cast<std::size_t>(shape.num_elements());
            std::vector<double> head_noise(count);
            std::vector<double> delta_noise(count);
            SaTest::HostNoise(K_HEAD_SEED).fill(head_noise);
            SaTest::HostNoise(K_DELTA_SEED).fill(delta_noise);

            // Every third delta is zero, as most entries of a real head's delta are.
            for (std::size_t i = 0; i < count; i += 3)
                delta_noise[i] = 0.0;

            SaTest::write_tensor(pool.tensor(head), head_noise);
            SaTest::write_tensor(pool.tensor(delta), delta_noise);

            // The tensors hold float32, so the expectation starts from the rounded values.
            for (std::size_t i = 0; i < count; ++i)
            {
                head_values.push_back(static_cast<double>(static_cast<float>(head_noise[i])));
                delta_values.push_back(static_cast<double>(static_cast<float>(delta_noise[i])));
            }
        }

        Graph::Graph graph;                      /**< @brief Holds the graph under test. */
        TensorPool pool;                         /**< @brief Holds the tensors of the graph. */
        std::unique_ptr<CompiledStep> step;      /**< @brief Holds the step, null when refused. */
        ValueID head = Graph::K_NO_VALUE;        /**< @brief Holds the head, a parameter. */
        ValueID delta = Graph::K_NO_VALUE;       /**< @brief Holds the delta, an input leaf. */
        ValueID loss = Graph::K_NO_VALUE;        /**< @brief Holds the emitted root. */
        ValueID gradient = Graph::K_NO_VALUE;    /**< @brief Holds the gradient at the head. */
        std::vector<double> head_values;         /**< @brief Holds the head as the tensor has it. */
        std::vector<double> delta_values;        /**< @brief Holds the delta the tensor holds. */
    };
} // namespace

/** @brief Tests that the gradient at the head is -scale * delta, element by element, to 1e-6. */
TEST_F(Medium_HeadDeltaLossEndToEnd, TheGradientAtTheHeadIsMinusScaleTimesDelta)
{
    DeltaLossCase model(backend());
    ASSERT_NE(model.step, nullptr);
    ASSERT_NE(model.gradient, Graph::K_NO_VALUE);

    SA_EXPECT_OK(model.step->run());

    const std::vector<double> gradient = SaTest::read_tensor(model.pool.tensor(model.gradient));
    ASSERT_EQ(gradient.size(), model.delta_values.size());

    expect_elements_within(gradient, negated_scaled(model.delta_values, K_ENGINE_SCALE),
                           K_GRADIENT_TOLERANCE);
}

/** @brief Tests that the root's value is -scale * sum(delta * head), summed in double, to 1e-4. */
TEST_F(Medium_HeadDeltaLossEndToEnd, TheValueIsMinusScaleTimesTheSumOfProducts)
{
    DeltaLossCase model(backend());
    ASSERT_NE(model.step, nullptr);

    SA_EXPECT_OK(model.step->run());

    const double sum = std::inner_product(model.delta_values.begin(), model.delta_values.end(),
                                          model.head_values.begin(), 0.0);

    const std::vector<double> value = SaTest::read_tensor(model.pool.tensor(model.loss));
    ASSERT_EQ(value.size(), 1u);
    EXPECT_NEAR(value[0], -K_ENGINE_SCALE * sum, K_VALUE_TOLERANCE);
}

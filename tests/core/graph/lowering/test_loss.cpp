/**************************************************************************/
/* test_loss.cpp                                                          */
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
 * @file test_loss.cpp
 * @brief Tests the loss lowerings: log-softmax and cross-entropy stay finite and exact far below
 *        the batch maximum.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /**
     * @brief Bounds the absolute error of a float result; 2048 unit roundoffs of float, the
     * smallest power of two that covers 1e-4.
     */
    constexpr double K_FLOAT_LOOSE_TOLERANCE = 2048.0 * TestSupport::unit_roundoff<float>();

    /** @brief Runs once per gap of K_GAPS, on the engine every case records on. */
    class Medium_LogSoftmaxStability : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<double>
    {

    };

    /** @brief Classes per row in every case here. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief `ln(1 + e^-1 + e^-2)`, the log-sum-exp of {a, a+1, a+2} minus `a + 2`. */
    const double K_LSE = std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));

    /** @brief The log-softmax of {a, a+1, a+2}, which does not depend on `a`. */
    const std::vector<double> K_EXPECTED_ROW = {-2.0 - K_LSE, -1.0 - K_LSE, -K_LSE};

    /** @brief Every gap this file checks, in units of the logits themselves. */
    const std::vector<double> K_GAPS = {0.0, 90.0, 100.0, 104.0, 200.0};

    /**
     * @brief Two rows of logits whose maxima sit @p gap apart.
     *
     * Row 0 is {0, 1, 2} and row 1 is {-gap, 1 - gap, 2 - gap}, so the batch
     * maximum is 2, row 1's own maximum is `2 - gap`, and the distance between
     * them is exactly @p gap.
     */
    std::vector<double> logits_with_gap(double gap)
    {
        return {0.0, 1.0, 2.0, -gap, 1.0 - gap, 2.0 - gap};
    }

    /** @brief The log-softmax of both rows of logits_with_gap, which no gap changes. */
    std::vector<double> expected_log_softmax()
    {
        std::vector<double> rows = K_EXPECTED_ROW;
        rows.insert(rows.end(), K_EXPECTED_ROW.begin(), K_EXPECTED_ROW.end());
        return rows;
    }

    /** @brief The cross-entropy gradient of both rows, (softmax - one-hot) / 2. */
    std::vector<double> expected_cross_entropy_gradient()
    {
        const double mass = 1.0 + std::exp(-1.0) + std::exp(-2.0);
        const double first = std::exp(-2.0) / mass;
        const double second = std::exp(-1.0) / mass;
        const double third = 1.0 / mass;

        return {(first - 1.0) / 2.0, second / 2.0, third / 2.0,
                (first - 1.0) / 2.0, second / 2.0, third / 2.0};
    }

    /** @brief Expects each element of @p got to be finite and within @p tolerance of @p expected. */
    void expect_finite_and_near(const std::vector<double>& got, const std::vector<double>& expected,
                                double tolerance)
    {
        ASSERT_EQ(expected.size(), got.size());

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(got[i])) << "element " << i << " is " << got[i];
            EXPECT_NEAR(got[i], expected[i], tolerance) << "element " << i;
        }
    }
}

/**
 * @brief Tests that log-softmax of a row far below the batch maximum is finite and within
 *        K_FLOAT_LOOSE_TOLERANCE.
 *
 * A row whose exponentials all underflow under the batch's global maximum has a row sum of
 * exactly zero, and log(0) propagates as +inf through the final subtraction.
 * @see tests/core/graph/README.md
 */
TEST_P(Medium_LogSoftmaxStability, ARowFarBelowTheBatchMaximumStaysFiniteAndCorrect)
{
    const double gap = GetParam();
    SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID l, builder.log_softmax(x));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), logits_with_gap(gap));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    expect_finite_and_near(read_tensor(pool.tensor(l)), expected_log_softmax(),
                           K_FLOAT_LOOSE_TOLERANCE);
}

/**
 * @brief Tests that cross-entropy over such a row is finite and equals 2 + k.
 * @see tests/core/graph/README.md
 */
TEST_P(Medium_LogSoftmaxStability, CrossEntropyOverSuchARowIsFiniteAndCorrect)
{
    const double gap = GetParam();
    SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
    const ValueID targets = builder.input(Shape{2, K_CLASSES}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            builder.nll_loss(builder.log_softmax(x), targets));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), logits_with_gap(gap));
    write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    expect_finite_and_near(read_tensor(pool.tensor(loss)), {2.0 + K_LSE},
                           K_FLOAT_LOOSE_TOLERANCE);
}

/**
 * @brief Tests that the cross-entropy gradient over such a row is finite and matches the softmax.
 *
 * A non-finite gradient is the failure that ends a run silently: the optimizer writes it into
 * the parameters and every later step trains on NaN.
 * @see tests/core/graph/README.md
 */
TEST_P(Medium_LogSoftmaxStability, TheCrossEntropyGradientOverSuchARowIsFinite)
{
    const double gap = GetParam();
    SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.parameter(Shape{2, K_CLASSES}, "x");
    const ValueID targets = builder.input(Shape{2, K_CLASSES}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            builder.nll_loss(builder.log_softmax(x), targets));

    const std::vector<ValueID> wrt = {x};
    const std::vector<ValueID> grads = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();
    ASSERT_NE(grads.at(0), Graph::K_NO_VALUE);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), logits_with_gap(gap));
    write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    expect_finite_and_near(read_tensor(pool.tensor(grads[0])),
                           expected_cross_entropy_gradient(), K_FLOAT_LOOSE_TOLERANCE);
}

/**
 * @brief Tests that the log-probabilities at a gap are exactly equal to those at no gap.
 *
 * Log-softmax is invariant to a per-row offset, so any difference is entirely the stabilising
 * shift's doing.
 * @see tests/core/graph/README.md
 */
TEST_P(Medium_LogSoftmaxStability, TheAnswerDoesNotDependOnTheGapAtAll)
{
    const double gap = GetParam();
    SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID l, builder.log_softmax(x));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend(), graph, pool));

    write_tensor(pool.tensor(x), logits_with_gap(0.0));
    SA_EXPECT_OK(step->run());
    const std::vector<double> baseline = read_tensor(pool.tensor(l));

    write_tensor(pool.tensor(x), logits_with_gap(gap));
    SA_EXPECT_OK(step->run());

    EXPECT_EQ(read_tensor(pool.tensor(l)), baseline);
}

INSTANTIATE_TEST_SUITE_P(, Medium_LogSoftmaxStability, ::testing::ValuesIn(K_GAPS),
                         [](const ::testing::TestParamInfo<double>& info)
                         {
                             return "Gap" + std::to_string(static_cast<int>(info.param));
                         });

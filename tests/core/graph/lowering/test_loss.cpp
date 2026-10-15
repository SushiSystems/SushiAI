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
 * @file test_loss.cpp
 * @brief Tests the loss lowerings: log-softmax and cross-entropy stay finite and exact far below
 *        the batch maximum.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

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
    class Medium_LogSoftmaxStability : public SaTest::EngineTest {};

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

    /**
     * @brief The largest absolute deviation of @p got from the per-row expectation.
     *
     * Infinite when any element is not finite, so a `+inf` row reports as `inf`
     * rather than as a NaN comparison that quietly comes out false.
     */
    double worst_error(const std::vector<double>& got)
    {
        double worst = 0.0;
        for (std::size_t row = 0; row < got.size() / static_cast<std::size_t>(K_CLASSES); ++row)
        {
            for (std::size_t c = 0; c < static_cast<std::size_t>(K_CLASSES); ++c)
            {
                const double value = got[row * static_cast<std::size_t>(K_CLASSES) + c];
                if (!std::isfinite(value))
                    return std::numeric_limits<double>::infinity();

                worst = std::max(worst, std::fabs(value - K_EXPECTED_ROW[c]));
            }
        }
        return worst;
    }
}

/**
 * @brief Tests that log-softmax of a row far below the batch maximum is finite and within 1e-4.
 * @see tests/core/graph/README.md
 */
TEST_F(Medium_LogSoftmaxStability, ARowFarBelowTheBatchMaximumStaysFiniteAndCorrect)
{
    for (double gap : K_GAPS)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
        const ValueID l = builder.log_softmax(x);

        TensorPool pool(*engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), logits_with_gap(gap));

        SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                                CompiledStep::compile(*engine, graph, pool));
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(l));
        ASSERT_EQ(6u, got.size());

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(K_CLASSES);
            EXPECT_TRUE(std::isfinite(got[i]))
                << "element " << i << " is " << got[i] << " at a gap of " << gap
                << ". A row whose exponentials all underflow under the batch's "
                   "global maximum has a row sum of exactly zero, and log(0) "
                   "propagates as +inf through the final subtraction.";
            EXPECT_NEAR(got[i], K_EXPECTED_ROW[c], 1e-4)
                << "element " << i << " at a gap of " << gap
                << "; worst error over the batch was " << worst_error(got);
        }
    }
}

/**
 * @brief Tests that cross-entropy over such a row is finite and equals 2 + k.
 * @see tests/core/graph/README.md
 */
TEST_F(Medium_LogSoftmaxStability, CrossEntropyOverSuchARowIsFiniteAndCorrect)
{
    for (double gap : K_GAPS)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
        const ValueID targets = builder.input(Shape{2, K_CLASSES}, "targets");
        const ValueID loss = builder.nll_loss(builder.log_softmax(x), targets);

        TensorPool pool(*engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), logits_with_gap(gap));
        write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

        SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                                CompiledStep::compile(*engine, graph, pool));
        SA_EXPECT_OK(step->run());

        const double got = read_tensor(pool.tensor(loss))[0];
        EXPECT_TRUE(std::isfinite(got))
            << "loss is " << got << " at a gap of " << gap;
        EXPECT_NEAR(got, 2.0 + K_LSE, 1e-4) << "at a gap of " << gap;
    }
}

/**
 * @brief Tests that the cross-entropy gradient over such a row is finite and matches the softmax.
 * @see tests/core/graph/README.md
 */
TEST_F(Medium_LogSoftmaxStability, TheCrossEntropyGradientOverSuchARowIsFinite)
{
    const double mass = 1.0 + std::exp(-1.0) + std::exp(-2.0);
    const std::vector<double> softmax_row =
        {std::exp(-2.0) / mass, std::exp(-1.0) / mass, 1.0 / mass};

    for (double gap : K_GAPS)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.parameter(Shape{2, K_CLASSES}, "x");
        const ValueID targets = builder.input(Shape{2, K_CLASSES}, "targets");
        const ValueID loss = builder.nll_loss(builder.log_softmax(x), targets);

        const std::vector<ValueID> wrt = {x};
        const std::vector<ValueID> grads = Autograd::differentiate(
            graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
        ASSERT_NE(grads.at(0), Graph::K_NO_VALUE);

        TensorPool pool(*engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), logits_with_gap(gap));
        write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

        SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                                CompiledStep::compile(*engine, graph, pool));
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(grads[0]));
        ASSERT_EQ(6u, got.size());

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(K_CLASSES);
            const double target = (c == 0) ? 1.0 : 0.0;
            const double expected = (softmax_row[c] - target) / 2.0;

            EXPECT_TRUE(std::isfinite(got[i]))
                << "gradient element " << i << " is " << got[i] << " at a gap of "
                << gap << ". A non-finite gradient here is the failure that ends a "
                   "run silently: the optimizer writes it into the parameters and "
                   "every later step trains on NaN.";
            EXPECT_NEAR(got[i], expected, 1e-4)
                << "gradient element " << i << " at a gap of " << gap;
        }
    }
}

/**
 * @brief Tests that the log-probabilities are exactly equal at every gap.
 * @see tests/core/graph/README.md
 */
TEST_F(Medium_LogSoftmaxStability, TheAnswerDoesNotDependOnTheGapAtAll)
{
    std::vector<double> baseline;

    for (double gap : K_GAPS)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{2, K_CLASSES}, "x");
        const ValueID l = builder.log_softmax(x);

        TensorPool pool(*engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), logits_with_gap(gap));

        SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                                CompiledStep::compile(*engine, graph, pool));
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(l));
        if (baseline.empty())
        {
            baseline = got;
            continue;
        }

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            EXPECT_EQ(got[i], baseline[i])
                << "element " << i << " moved by " << (got[i] - baseline[i])
                << " when row 1 was offset by " << gap
                << ". Log-softmax is invariant to a per-row offset, so this "
                   "difference is entirely the stabilising shift's doing.";
        }
    }
}

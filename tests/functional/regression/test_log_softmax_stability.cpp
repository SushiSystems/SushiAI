/**************************************************************************/
/* test_log_softmax_stability.cpp                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// Log-softmax stabilised by the WRONG maximum.
//
// The lowering used to shift the whole batch by the batch's global maximum,
// on the argument that log-softmax is invariant to subtracting any single
// scalar from a row. That argument is sound in real arithmetic and false in
// binary32, because the invariance is destroyed by the *exponential*, not by
// the subtraction: a row sitting a long way below the batch maximum has every
// `exp(x - global_max)` collapse into the subnormals and then to zero, its row
// sum becomes exactly zero, `log(0)` is `-inf`, and the row's log-probabilities
// come back `+inf`.
//
// Measured on this project's OpenCL CPU device, FLOAT32, on the [2, 3] input
// below, whose true log-probabilities are about -0.41 to -2.41.
//
// The quantity is the worst absolute deviation of row 1's log-probabilities
// from the answer the same code gives at a gap of 0. That is the one figure
// directly measured on BOTH sides of the fix, and the assertion that produces
// it is TheAnswerDoesNotDependOnTheGapAtAll's EXPECT_EQ:
//
//   | row-to-batch-max gap | global shift (before) | per-row shift (after) |
//   |---------------------:|----------------------:|----------------------:|
//   |                   90 |              2.15e-06 |           0 (exactly) |
//   |                  100 |              2.70e-02 |           0 (exactly) |
//   |                  104 |                   inf |           0 (exactly) |
//   |                  200 |                   inf |           0 (exactly) |
//
// Measured against the hand-computed expectation directly, rather than against
// the gap-0 answer, the gap-100 case came out 2.7033e-02 off - which is 6.6% of
// the largest true log-probability (-0.40761), and is the "about 7% low at 100"
// SushiBLAS's GPU_VALIDATION.md recorded for this defect before it was fixed.
//
// The residual at the gaps that PASS is deliberately absent from both columns,
// on both sides, because it was never measured: a passing EXPECT_NEAR bounds it
// under 1e-4 and says nothing more. Do not quote a figure for it - an earlier
// draft of this header did, and the invented numbers survived a full run
// unchallenged because nothing here fails when a comment is wrong.
//
// This is a regression test rather than an integration one because of how it
// fails. `+inf` log-probabilities do not stop a run: the loss becomes `inf` or
// `nan`, the optimizer writes `nan` parameters, and the training loop reports
// completion having learnt nothing. Nothing throws and no test that only checks
// "did it run" notices. That is exactly the failure mode this tier exists for.
//
// The expectations are hand-computed, not read off another path in this
// repository. Each row is {a, a+1, a+2}, whose log-softmax is
// {-2 - k, -1 - k, -k} with k = ln(1 + e^-1 + e^-2), for *any* a - so the
// expected values are the same for every gap under test, and any dependence on
// `a` at all is the defect showing.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    class Regression_LogSoftmaxStability : public SaTest::EngineTest {};

    /** @brief Classes per row in every case here. */
    constexpr int64_t kClasses = 3;

    /** @brief `ln(1 + e^-1 + e^-2)`, the log-sum-exp of {a, a+1, a+2} minus `a + 2`. */
    const double kLse = std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));

    /**
     * @brief The log-softmax of {a, a+1, a+2}, which does not depend on `a`.
     */
    const std::vector<double> kExpectedRow = {-2.0 - kLse, -1.0 - kLse, -kLse};

    /** @brief Every gap this file checks, in units of the logits themselves. */
    const std::vector<double> kGaps = {0.0, 90.0, 100.0, 104.0, 200.0};

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
        for (std::size_t row = 0; row < got.size() / static_cast<std::size_t>(kClasses); ++row)
        {
            for (std::size_t c = 0; c < static_cast<std::size_t>(kClasses); ++c)
            {
                const double value = got[row * static_cast<std::size_t>(kClasses) + c];
                if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
                worst = std::max(worst, std::fabs(value - kExpectedRow[c]));
            }
        }
        return worst;
    }
}

// The defect itself. At a gap of 104 this returned `+inf` for row 1; at 100 it
// was 6.6% low on the largest log-probability (-0.43464 measured against a true
// -0.40761); at 90 it had drifted 2.15e-06 off its own gap-0 answer. The per-row
// maximum removes the gap from the arithmetic entirely, so the answer is the
// same at every gap - which is what the single shared expectation asserts.
//
// The tolerance here is 1e-4, which is wide enough to pass at gap 90 both before
// and after. That is on purpose: this case exists to catch `+inf` and gross
// error, and the fine-grained claim - that the gap changes *nothing* - is
// TheAnswerDoesNotDependOnTheGapAtAll's to make, at exact equality.
TEST_F(Regression_LogSoftmaxStability, ARowFarBelowTheBatchMaximumStaysFiniteAndCorrect)
{
    for (double gap : kGaps)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{2, kClasses}, "x");
        const ValueId l = builder.log_softmax(x);

        TensorPool pool(*engine);
        pool.materialise_leaves(graph);
        write_tensor(pool.tensor(x), logits_with_gap(gap));

        CompiledStep step(*engine, graph, pool);
        step.run();

        const std::vector<double> got = read_tensor(pool.tensor(l));
        ASSERT_EQ(6u, got.size());

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(kClasses);
            EXPECT_TRUE(std::isfinite(got[i]))
                << "element " << i << " is " << got[i] << " at a gap of " << gap
                << ". A row whose exponentials all underflow under the batch's "
                   "global maximum has a row sum of exactly zero, and log(0) "
                   "propagates as +inf through the final subtraction.";
            EXPECT_NEAR(got[i], kExpectedRow[c], 1e-4)
                << "element " << i << " at a gap of " << gap
                << "; worst error over the batch was " << worst_error(got);
        }
    }
}

// The same defect one node downstream. Cross-entropy is `-sum(L * Y) / N` over
// the log-probabilities, so an infinite row makes the loss infinite - and the
// loss is the number the training loop prints and the number every stopping
// rule reads.
//
// Hand-computed: with the target on class 0 of both rows, the loss is
// `-(2 * (-2 - k)) / 2 = 2 + k`, whatever the gap.
TEST_F(Regression_LogSoftmaxStability, CrossEntropyOverSuchARowIsFiniteAndCorrect)
{
    for (double gap : kGaps)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{2, kClasses}, "x");
        const ValueId targets = builder.input(Shape{2, kClasses}, "targets");
        const ValueId loss = builder.nll_loss(builder.log_softmax(x), targets);

        TensorPool pool(*engine);
        pool.materialise_leaves(graph);
        write_tensor(pool.tensor(x), logits_with_gap(gap));
        write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

        CompiledStep step(*engine, graph, pool);
        step.run();

        const double got = read_tensor(pool.tensor(loss))[0];
        EXPECT_TRUE(std::isfinite(got))
            << "loss is " << got << " at a gap of " << gap;
        EXPECT_NEAR(got, 2.0 + kLse, 1e-4) << "at a gap of " << gap;
    }
}

// And the failure that actually destroys a run rather than merely reporting a
// bad number. The fused cross-entropy gradient is `(exp(L) - Y) / N`, so an
// infinite `L` makes the gradient infinite, the optimizer writes non-finite
// parameters, and every subsequent step trains on NaN while the loop reports
// success.
//
// Hand-computed: `exp(L)` is the softmax of {a, a+1, a+2}, which is
// {e^-2, e^-1, 1} / (1 + e^-1 + e^-2), the same for both rows; subtract the
// one-hot target and divide by the batch size of 2.
TEST_F(Regression_LogSoftmaxStability, TheCrossEntropyGradientOverSuchARowIsFinite)
{
    const double mass = 1.0 + std::exp(-1.0) + std::exp(-2.0);
    const std::vector<double> softmax_row =
        {std::exp(-2.0) / mass, std::exp(-1.0) / mass, 1.0 / mass};

    for (double gap : kGaps)
    {
        SCOPED_TRACE("row-to-batch-maximum gap of " + std::to_string(gap));

        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.parameter(Shape{2, kClasses}, "x");
        const ValueId targets = builder.input(Shape{2, kClasses}, "targets");
        const ValueId loss = builder.nll_loss(builder.log_softmax(x), targets);

        const std::vector<ValueId> wrt = {x};
        const std::vector<ValueId> grads = Autograd::differentiate(
            graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));
        ASSERT_NE(grads.at(0), Graph::NO_VALUE);

        TensorPool pool(*engine);
        pool.materialise_leaves(graph);
        write_tensor(pool.tensor(x), logits_with_gap(gap));
        write_tensor(pool.tensor(targets), std::vector<double>{1, 0, 0, 1, 0, 0});

        CompiledStep step(*engine, graph, pool);
        step.run();

        const std::vector<double> got = read_tensor(pool.tensor(grads[0]));
        ASSERT_EQ(6u, got.size());

        for (std::size_t i = 0; i < got.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(kClasses);
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

// The property that makes the fix a fix rather than a wider threshold: nothing
// in the answer depends on the gap at all. Two batches differing only by a
// constant offset applied to one row must produce identical log-probabilities
// for the row that did not move, and identical ones for the row that did.
//
// A shift by the global maximum fails this at any gap large enough to matter; a
// shift by each row's own maximum passes it at every gap, because the gap never
// enters the arithmetic.
//
// Asserted as **equality**, not as a tolerance, and that is not optimism: every
// logit here is a small integer, so each row's own maximum is exact, the shift
// `x - row_max` is exact, and both rows are therefore reduced to literally the
// same three floats {-2, -1, 0} before anything transcendental happens. The gap
// cannot reach the answer, so any difference at all is a real one - and a
// tolerance here would have been the same mistake in miniature, a threshold
// wide enough to hide the thing under test.
TEST_F(Regression_LogSoftmaxStability, TheAnswerDoesNotDependOnTheGapAtAll)
{
    std::vector<double> baseline;

    for (double gap : kGaps)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{2, kClasses}, "x");
        const ValueId l = builder.log_softmax(x);

        TensorPool pool(*engine);
        pool.materialise_leaves(graph);
        write_tensor(pool.tensor(x), logits_with_gap(gap));

        CompiledStep step(*engine, graph, pool);
        step.run();

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

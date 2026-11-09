/**************************************************************************/
/* test_dense_losses.cpp                                                  */
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
 * @file test_dense_losses.cpp
 * @brief Tests the loss graphs that compare an output with a target element by element.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The rounded operations of the mean of eight squared differences. */
    constexpr std::size_t K_SQUARED_ERROR_OPERATIONS = 10;

    /** @brief The rounded operations of the mean of twelve elements: the sum and the scale. */
    constexpr std::size_t K_MEAN_OPERATIONS = 14;

    /** @brief The rounded operations of the logistic loss of a few elements in float32. */
    constexpr std::size_t K_LOGISTIC_OPERATIONS = 16;

    /** @brief Returns the cross entropy of sigmoid(x) against y, averaged: the closed form. */
    double logistic_mean_of(const std::vector<double>& logits, const std::vector<double>& targets)
    {
        double total = 0.0;
        for (std::size_t i = 0; i < logits.size(); ++i)
        {
            const double x = logits[i];
            total += std::max(x, 0.0) - x * targets[i] + std::log1p(std::exp(-std::abs(x)));
        }

        return total / static_cast<double>(logits.size());
    }

    /** @brief Holds logits and targets of one shape for the logistic loss. */
    struct LogisticRow
    {
        std::string name;               /**< The row's name. */
        Shape shape;                    /**< The shape of the logits and of the targets. */
        std::vector<double> logits;     /**< The logits. */
        std::vector<double> targets;    /**< The targets, each in [0, 1]. */
    };

    /** @brief Runs the logistic loss of each row on the device. */
    class Medium_DenseLossesLogistic : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<LogisticRow>
    {
    };
} // namespace

/** @brief Tests that the mean over every element is the sum of the elements over their count. */
TEST(Small_DenseLosses, TheMeanIsOneColumnSummedAndScaled)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID values = builder.input(Shape{4, 3}, "values");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mean,
                            Train::emit_mean_of_elements(builder, graph, values));
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(values, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                                          12.0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(mean));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<double>(result.front(), 6.5, K_MEAN_OPERATIONS)));
}

/** @brief Tests that the logistic loss is the cross entropy of the logit, finite at +-100. */
TEST_P(Medium_DenseLossesLogistic, TheLogisticLossReadsTheLogitInItsStableForm)
{
    const LogisticRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(row.shape, "logits");
    const Graph::ValueID targets = builder.input(row.shape, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            Train::emit_logistic_loss(builder, graph, logits, targets));
    Graph::TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(logits), row.logits);
    SaTest::write_tensor(pool.tensor(targets), row.targets);
    std::unique_ptr<Graph::CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(backend(), graph, pool));

    SA_EXPECT_OK(step->run());

    const std::vector<double> result = SaTest::read_tensor(pool.tensor(loss));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        result.front(), logistic_mean_of(row.logits, row.targets), K_LOGISTIC_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_DenseLossesLogistic,
    ::testing::Values(
        LogisticRow{"ModerateLogits", Shape{2, 2}, {0.5, -1.5, 2.0, -0.25}, {1.0, 0.0, 0.5, 1.0}},
        LogisticRow{"LargeLogitsStayFinite", Shape{1, 2}, {100.0, -100.0}, {0.0, 1.0}}),
    [](const ::testing::TestParamInfo<LogisticRow>& info) { return info.param.name; });

/** @brief Tests that the squared error is the mean of the squared differences. */
TEST(Small_DenseLosses, TheSquaredErrorSquaresTheDifference)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID outputs = builder.input(Shape{4, 2}, "outputs");
    const Graph::ValueID targets = builder.input(Shape{4, 2}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            Train::emit_squared_error(builder, graph, outputs, targets));
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(outputs, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
    SA_EXPECT_OK(evaluator.write(targets, {0.0, 2.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0}));

    SA_EXPECT_OK(evaluator.run());

    // The differences are 1, 0, -2, 0, 0, 0, 0, 2; their squares sum to 9.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(loss));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result.front(), 9.0 / 8.0, K_SQUARED_ERROR_OPERATIONS)));
}

/** @brief Tests that two values of different shapes stop the process, naming both. */
TEST(Medium_DenseLossesDeathTest, OperandsOfTwoShapesAreFatal)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID outputs = builder.input(Shape{4, 2}, "outputs");
    const Graph::ValueID targets = builder.input(Shape{4, 3}, "targets");

    SA_EXPECT_FATAL((void)Train::emit_squared_error(builder, graph, outputs, targets),
                    "[4, 2]", "[4, 3]");
}

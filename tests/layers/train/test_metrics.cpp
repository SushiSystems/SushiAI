/**************************************************************************/
/* test_metrics.cpp                                                       */
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
 * @file test_metrics.cpp
 * @brief Tests the named numbers of a step and the lookup of a step's one output.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;
using Train::Metric;
using Train::Metrics;

static_assert(Train::K_METRIC_CAPACITY == 8, "the overflow row fills eight names");

namespace
{
    /** @brief Returns the view of a step over @p batch that published @p outputs. */
    Train::StepView view_of(const Data::Batch& batch,
                            const std::vector<Train::StepOutput>& outputs)
    {
        return Train::StepView{
            batch, 0.0,
            SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};
    }

    /** @brief Returns a Metrics holding one metric of value 1 for each of @p names. */
    Metrics holding(const std::vector<std::string_view>& names)
    {
        Metrics metrics;
        for (const std::string_view name : names)
            SA_EXPECT_OK(metrics.add(Metric{name, 1.0, false}));

        return metrics;
    }

    /** @brief Holds the names a set already holds, the metric added and the refusal. */
    struct AddRow
    {
        std::string name;                   /**< The row's name. */
        std::vector<std::string_view> existing; /**< The metrics the set holds first. */
        std::string_view added;                 /**< The name of the metric added. */
        std::string first;                  /**< A text the message holds. */
        std::string second;                 /**< Another text the message holds. */
    };

    /** @brief Adds a metric a set refuses, for each row. */
    class Small_MetricsRefusal : public ::testing::TestWithParam<AddRow>
    {

    };

    /** @brief Holds a metric and the text it prints as. */
    struct DescribeRow
    {
        std::string name;       /**< The row's name. */
        Metric metric;          /**< The metric described. */
        std::string expected;   /**< The text it prints as. */
    };

    /** @brief Prints the metric of each row. */
    class Small_MetricsDescribe : public ::testing::TestWithParam<DescribeRow>
    {

    };

    /** @brief Holds the outputs a step published, each as a name and whether it holds a tensor. */
    struct OutputsRow
    {
        std::string name;                                   /**< The row's name. */
        std::vector<std::pair<std::string, bool>> outputs;  /**< The outputs, in order. */
        std::string first;                                  /**< A text the message holds. */
        std::string second;                                 /**< Another text the message holds. */
    };

    /** @brief Asks the only output of the step of each row. */
    class Small_MetricsOnlyOutput : public ::testing::TestWithParam<OutputsRow>
    {

    };
} // namespace

/** @brief Tests that a new set holds no metric. */
TEST(Small_Metrics, ANewSetIsEmpty)
{
    const Metrics metrics;

    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that metrics are kept in the order they were added. */
TEST(Small_Metrics, MetricsAreKeptInTheOrderTheyWereAdded)
{
    Metrics metrics;

    SA_EXPECT_OK(metrics.add(Metric{"accuracy", 0.75, true}));
    SA_EXPECT_OK(metrics.add(Metric{"perplexity", 12.5, false}));

    ASSERT_EQ(metrics.all().size(), 2u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[1].name, "perplexity");
}

/** @brief Tests that find returns the metric of a name and null for another. */
TEST(Small_Metrics, AMetricIsFoundByName)
{
    Metrics metrics;
    SA_EXPECT_OK(metrics.add(Metric{"perplexity", 12.5, false}));

    ASSERT_NE(metrics.find("perplexity"), nullptr);
    EXPECT_EQ(metrics.find("perplexity")->value, 12.5);
    EXPECT_EQ(metrics.find("recall"), nullptr);
}

/** @brief Tests that value_or returns the value of a name and the fallback for another. */
TEST(Small_Metrics, ValueOrFallsBackForAMissingName)
{
    Metrics metrics;
    SA_EXPECT_OK(metrics.add(Metric{"accuracy", 0.75, true}));

    EXPECT_EQ(metrics.value_or("accuracy", -1.0), 0.75);
    EXPECT_EQ(metrics.value_or("recall", -1.0), -1.0);
}

/** @brief Tests that clear removes every metric. */
TEST(Small_Metrics, ClearRemovesEveryMetric)
{
    Metrics metrics = holding({"accuracy"});

    metrics.clear();

    EXPECT_TRUE(metrics.all().empty());
    EXPECT_EQ(metrics.find("accuracy"), nullptr);
}

/** @brief Tests that an empty name, a second metric of one name and a ninth are refused. */
TEST_P(Small_MetricsRefusal, AMetricTheSetCannotHoldIsRefused)
{
    const AddRow& row = GetParam();
    Metrics metrics = holding(row.existing);

    SA_EXPECT_FAILURE(metrics.add(Metric{row.added, 2.0, false}), StatusCode::InvalidDescription,
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MetricsRefusal,
    ::testing::Values(
        AddRow{"AnEmptyName", {}, "", "non-empty name", "non-empty name"},
        AddRow{"ASecondMetricOfOneName", {"a"}, "a", "'a'", "twice"},
        AddRow{"AnOverflow", {"a", "b", "c", "d", "e", "f", "g", "h"}, "i", "at most 8", "'i'"}),
    [](const ::testing::TestParamInfo<AddRow>& info) { return info.param.name; });

/** @brief Tests that a refused second metric of one name leaves the first value alone. */
TEST(Small_Metrics, ARefusedDuplicateKeepsTheFirstValue)
{
    Metrics metrics = holding({"a"});

    SA_EXPECT_FAILURE(metrics.add(Metric{"a", 2.0, false}), StatusCode::InvalidDescription,
                      "twice");

    EXPECT_EQ(metrics.find("a")->value, 1.0);
}

/** @brief Tests that a refused ninth metric leaves the set at its capacity. */
TEST(Small_Metrics, ARefusedOverflowLeavesTheSetFull)
{
    Metrics metrics = holding({"a", "b", "c", "d", "e", "f", "g", "h"});

    SA_EXPECT_FAILURE(metrics.add(Metric{"i", 0.0, false}), StatusCode::InvalidDescription,
                      "at most 8");

    EXPECT_EQ(metrics.all().size(), Train::K_METRIC_CAPACITY);
}

/** @brief Tests that a fraction prints as a percentage and any other number as it is. */
TEST_P(Small_MetricsDescribe, AFractionPrintsAsAPercentage)
{
    const DescribeRow& row = GetParam();

    EXPECT_EQ(Train::describe_metric_value(row.metric), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MetricsDescribe,
    ::testing::Values(DescribeRow{"AFraction", Metric{"accuracy", 0.9240, true}, "92.40%"},
                      DescribeRow{"ABareNumber", Metric{"perplexity", 12.5, false}, "12.5"}),
    [](const ::testing::TestParamInfo<DescribeRow>& info) { return info.param.name; });

/** @brief Tests that only_output returns the tensor of a step that published one output. */
TEST(Small_Metrics, OnlyOutputReturnsTheOneTensor)
{
    const SushiBLAS::Tensor tensor{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> one{Train::StepOutput{"output", &tensor}};

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const found,
                            Train::only_output(view_of(batch, one)));
    EXPECT_EQ(found, &tensor);
}

/** @brief Tests that only_output refuses no output, two outputs and an output with no tensor. */
TEST_P(Small_MetricsOnlyOutput, OnlyOutputRefusesAnyOtherCount)
{
    const OutputsRow& row = GetParam();
    const SushiBLAS::Tensor tensor{};
    const Data::Batch batch{};
    std::vector<Train::StepOutput> outputs;
    for (const std::pair<std::string, bool>& output : row.outputs)
        outputs.push_back(Train::StepOutput{output.first, output.second ? &tensor : nullptr});

    SA_EXPECT_FAILURE(Train::only_output(view_of(batch, outputs)),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MetricsOnlyOutput,
    ::testing::Values(
        OutputsRow{"NoOutput", {}, "published 0", "published 0"},
        OutputsRow{"TwoOutputs", {{"p3", true}, {"p4", true}}, "published 2", "published 2"},
        OutputsRow{"AnOutputWithNoTensor", {{"output", false}}, "'output'", "no tensor"}),
    [](const ::testing::TestParamInfo<OutputsRow>& info) { return info.param.name; });

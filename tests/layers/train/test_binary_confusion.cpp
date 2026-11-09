/**************************************************************************/
/* test_binary_confusion.cpp                                              */
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
 * @file test_binary_confusion.cpp
 * @brief Tests the count of yes-or-no predictions, its ratios and the decision threshold.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_confusion.hpp>
#include <SushiAI/train/decision_threshold.hpp>
#include <SushiAI/train/host_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Train::BinaryConfusion;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The rounded operations on the longest path from a count to a ratio. */
    constexpr std::size_t K_RATIO_OPERATIONS = 4;

    /** @brief The rounded operations of the logit of a probability and the sigmoid of that logit. */
    constexpr std::size_t K_SIGMOID_ROUND_TRIP_OPERATIONS = 8;

    /** @brief Provides the host engine the logits and the truth are allocated on. */
    class HostTensors
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape, DType dtype,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                              allocate_tensor(backend.engine(), shape, dtype));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Counts the rows of one cut over the host engine. */
    class Small_BinaryConfusion : public HostTensors, public ::testing::Test
    {
    };

    /** @brief Holds a column of a [2, 2] matrix of logits and the counts that column gives. */
    struct ColumnRow
    {
        std::string name;                /**< The row's name. */
        std::int64_t column;             /**< The column counted. */
        std::int64_t true_positives;     /**< The expected true positives. */
        std::int64_t false_positives;    /**< The expected false positives. */
        std::int64_t false_negatives;    /**< The expected false negatives. */
    };

    /** @brief Counts each column of one matrix over the host engine. */
    class Small_BinaryConfusionColumn : public HostTensors,
                                        public ::testing::TestWithParam<ColumnRow>
    {
    };

    /** @brief Holds a threshold and the count the one row of logit 1 lands in. */
    struct CutRow
    {
        std::string name;                /**< The row's name. */
        double probability;              /**< The threshold, as a probability. */
        std::int64_t true_positives;     /**< The expected true positives. */
        std::int64_t false_negatives;    /**< The expected false negatives. */
    };

    /** @brief Counts one row at each cut over the host engine. */
    class Small_BinaryConfusionCut : public HostTensors, public ::testing::TestWithParam<CutRow>
    {
    };
} // namespace

/** @brief Tests that each of the four counts is the number of rows in its cell. */
TEST_F(Small_BinaryConfusion, EachRowLandsInOneOfTheFourCounts)
{
    // Rows: hit, false alarm, miss, correct rejection, hit at the cut itself.
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{5, 1}, DType::Float32,
                                      {2.0, 1.0, -1.0, -3.0, 0.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor truth,
                            tensor_of(Shape{5}, DType::Float32, {1.0, 0.0, 1.0, 0.0, 1.0}));

    const BinaryConfusion counts = Train::count_confusion(
        logits, truth, Train::ColumnOfLogits{5, 1, 0}, Train::logit_of(0.5));

    EXPECT_EQ(counts.true_positives, 2);
    EXPECT_EQ(counts.false_positives, 1);
    EXPECT_EQ(counts.false_negatives, 1);
    EXPECT_EQ(counts.true_negatives, 1);
}

/** @brief Tests that a count reads its own column of a matrix and no other. */
TEST_P(Small_BinaryConfusionColumn, AColumnIsCountedAlone)
{
    const ColumnRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{2, 2}, DType::Float64, {3.0, -3.0, 3.0, 3.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor truth,
                            tensor_of(Shape{2, 2}, DType::Float32, {1.0, 1.0, 1.0, 0.0}));

    const BinaryConfusion counts =
        Train::count_confusion(logits, truth, Train::ColumnOfLogits{2, 2, row.column}, 0.0);

    EXPECT_EQ(counts.true_positives, row.true_positives);
    EXPECT_EQ(counts.false_positives, row.false_positives);
    EXPECT_EQ(counts.false_negatives, row.false_negatives);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BinaryConfusionColumn,
    ::testing::Values(ColumnRow{"TheFirstColumn", 0, 2, 0, 0},
                      ColumnRow{"TheSecondColumn", 1, 0, 1, 1}),
    [](const ::testing::TestParamInfo<ColumnRow>& info) { return info.param.name; });

/** @brief Tests that a raised threshold moves the cut above a logit that passed one half. */
TEST_P(Small_BinaryConfusionCut, TheThresholdMovesTheCut)
{
    const CutRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{1, 1}, DType::Float32, {1.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor truth,
                            tensor_of(Shape{1}, DType::Float32, {1.0}));

    // sigmoid(1) is 0.731, between the two thresholds.
    const BinaryConfusion counts = Train::count_confusion(
        logits, truth, Train::ColumnOfLogits{1, 1, 0}, Train::logit_of(row.probability));

    EXPECT_EQ(counts.true_positives, row.true_positives);
    EXPECT_EQ(counts.false_negatives, row.false_negatives);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BinaryConfusionCut,
    ::testing::Values(CutRow{"AtOneHalf", 0.5, 1, 0}, CutRow{"AtNineTenths", 0.9, 0, 1}),
    [](const ::testing::TestParamInfo<CutRow>& info) { return info.param.name; });

/** @brief Tests that every ratio of an empty count is 0 and none divides by zero. */
TEST(Small_BinaryConfusionRatios, ARatioOverNothingIsZero)
{
    const BinaryConfusion none{};
    EXPECT_EQ(Train::accuracy_of(none), 0.0);
    EXPECT_EQ(Train::precision_of(none), 0.0);
    EXPECT_EQ(Train::recall_of(none), 0.0);
    EXPECT_EQ(Train::f1_of(none), 0.0);
}

/** @brief Tests that true negatives alone give an accuracy of 1 and an F1 of 0. */
TEST(Small_BinaryConfusionRatios, RejectionsAloneGiveFullAccuracyAndNoF1)
{
    BinaryConfusion rejections{};
    rejections.true_negatives = 4;

    EXPECT_EQ(Train::accuracy_of(rejections), 1.0);
    EXPECT_EQ(Train::f1_of(rejections), 0.0);
}

/** @brief Tests that the four ratios of the counts 2, 1, 1, 1 are their quotients. */
TEST(Small_BinaryConfusionRatios, TheRatiosAreTheQuotientsOfTheCounts)
{
    BinaryConfusion counts{};
    counts.true_positives = 2;
    counts.false_positives = 1;
    counts.false_negatives = 1;
    counts.true_negatives = 1;

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Train::accuracy_of(counts), 3.0 / 5.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Train::precision_of(counts), 2.0 / 3.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Train::recall_of(counts), 2.0 / 3.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Train::f1_of(counts), 4.0 / 6.0, K_RATIO_OPERATIONS)));
}

namespace
{
    /** @brief Runs the inverse of the sigmoid over each probability. */
    class Small_DecisionThresholdLogit : public ::testing::TestWithParam<double>
    {
    };
} // namespace

/** @brief Tests that logit_of at one half is the logit 0. */
TEST(Small_DecisionThreshold, LogitOfOneHalfIsZero)
{
    EXPECT_EQ(Train::logit_of(0.5), 0.0);
}

/** @brief Tests that logit_of inverts the sigmoid above one half and below it. */
TEST_P(Small_DecisionThresholdLogit, LogitOfInvertsTheSigmoid)
{
    const double probability = GetParam();
    const double logit = Train::logit_of(probability);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        1.0 / (1.0 + std::exp(-logit)), probability, K_SIGMOID_ROUND_TRIP_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(Probabilities, Small_DecisionThresholdLogit,
                         ::testing::Values(0.1, 0.731, 0.9));

namespace
{
    /** @brief Parses @p text, a JSON document the test states; a refusal stops the test. */
    Config::Value parsed(std::string_view text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Holds a JSON document and the threshold the reader returns for it. */
    struct ReadRow
    {
        std::string name;       /**< The row's name. */
        std::string json;       /**< The document read. */
        double threshold;       /**< The threshold the reader returns. */
    };

    /** @brief Runs the threshold reader over each accepted document. */
    class Small_DecisionThresholdRead : public ::testing::TestWithParam<ReadRow>
    {
    };

    /** @brief Holds a JSON document and the words of the refusal it gets. */
    struct RefusedRow
    {
        std::string name;       /**< The row's name. */
        std::string json;       /**< The document read. */
        std::string first;      /**< A text the message holds. */
        std::string second;     /**< Another text the message holds. */
        std::string third;      /**< A third text the message holds. */
    };

    /** @brief Runs the threshold reader over each refused document. */
    class Small_DecisionThresholdRefused : public ::testing::TestWithParam<RefusedRow>
    {
    };
} // namespace

/** @brief Tests that the reader returns the stated threshold and one half for none. */
TEST_P(Small_DecisionThresholdRead, TheReaderDefaultsToOneHalf)
{
    const ReadRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const double threshold,
                            Train::read_threshold(parsed(row.json), "objective"));
    EXPECT_EQ(threshold, row.threshold);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DecisionThresholdRead,
    ::testing::Values(ReadRow{"AbsentKey", "{}", Train::K_DEFAULT_THRESHOLD},
                      ReadRow{"StatedQuarter", R"({"threshold": 0.25})", 0.25}),
    [](const ::testing::TestParamInfo<ReadRow>& info) { return info.param.name; });

/** @brief Tests that another key, a value that is no number and a bound are each refused. */
TEST_P(Small_DecisionThresholdRefused, TheReaderRefusesWhatIsNoProbability)
{
    const RefusedRow& row = GetParam();

    SA_EXPECT_FAILURE(Train::read_threshold(parsed(row.json), "objective"),
                      StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DecisionThresholdRefused,
    ::testing::Values(
        RefusedRow{"AnotherKey", R"({"cut": 0.5})", "objective", "'cut'", "'threshold'"},
        RefusedRow{"ANonNumber", R"({"threshold": "high"})", "objective", "number", ""},
        RefusedRow{"Zero", R"({"threshold": 0})", "above 0 and below 1", "", ""},
        RefusedRow{"One", R"({"threshold": 1})", "above 0 and below 1", "", ""},
        RefusedRow{"BelowZero", R"({"threshold": -0.5})", "above 0 and below 1", "", ""},
        RefusedRow{"AboveOne", R"({"threshold": 1.5})", "above 0 and below 1", "", ""}),
    [](const ::testing::TestParamInfo<RefusedRow>& info) { return info.param.name; });

/** @brief Tests that the host read accepts a float tensor of the asked count. */
TEST_F(Small_BinaryConfusion, TheHostReadAcceptsAFloatTensorOfTheAskedCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor reals,
                            tensor_of(Shape{2, 2}, DType::Float64, {0.5, 1.5, 2.5, 3.5}));

    SA_EXPECT_OK(Train::require_host_reals(reals, 4, "the logits"));
    EXPECT_EQ(Train::host_real(reals, 2), 2.5);
}

/** @brief Tests that the host read refuses an integer tensor. */
TEST_F(Small_BinaryConfusion, TheHostReadRefusesAnIntegerTensor)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor integers,
                            allocate_tensor(backend.engine(), Shape{4}, DType::Int32));

    SA_EXPECT_FAILURE(Train::require_host_reals(integers, 4, "the logits"),
                      StatusCode::InvalidDescription, "host read of the logits", "int32");
}

/** @brief Tests that the host read refuses another element count. */
TEST_F(Small_BinaryConfusion, TheHostReadRefusesAnotherElementCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor reals,
                            tensor_of(Shape{2, 2}, DType::Float64, {0.5, 1.5, 2.5, 3.5}));

    SA_EXPECT_FAILURE(Train::require_host_reals(reals, 6, "the logits"),
                      StatusCode::Incompatible, "6 values", "holds 4");
}

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
 * @file test_binary_confusion.cpp
 * @brief Tests the count of yes-or-no predictions, its ratios and the decision threshold.
 * @author Mustafa Garip
 */

#include <cmath>
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
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_confusion.hpp>
#include <SushiAI/train/decision_threshold.hpp>
#include <SushiAI/train/host_values.hpp>

using namespace SushiAI;
using Train::BinaryConfusion;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the difference between a ratio of small counts and its exact value. */
    constexpr double K_RATIO_TOLERANCE = 1e-12;

    /** @brief Provides the engine the logits and the truth are allocated on. */
    class Medium_BinaryConfusion : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape, DType dtype,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(*engine, shape, dtype));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }
    };
} // namespace

/** @brief Tests that each of the four counts is the number of rows in its cell. */
TEST_F(Medium_BinaryConfusion, EachRowLandsInOneOfTheFourCounts)
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
    EXPECT_NEAR(Train::accuracy_of(counts), 3.0 / 5.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(Train::precision_of(counts), 2.0 / 3.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(Train::recall_of(counts), 2.0 / 3.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(Train::f1_of(counts), 4.0 / 6.0, K_RATIO_TOLERANCE);
}

/** @brief Tests that a count reads its own column of a matrix and no other. */
TEST_F(Medium_BinaryConfusion, AColumnIsCountedAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{2, 2}, DType::Float64, {3.0, -3.0, 3.0, 3.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor truth,
                            tensor_of(Shape{2, 2}, DType::Float32, {1.0, 1.0, 1.0, 0.0}));

    const BinaryConfusion first =
        Train::count_confusion(logits, truth, Train::ColumnOfLogits{2, 2, 0}, 0.0);
    const BinaryConfusion second =
        Train::count_confusion(logits, truth, Train::ColumnOfLogits{2, 2, 1}, 0.0);

    EXPECT_EQ(first.true_positives, 2);
    EXPECT_EQ(second.true_positives, 0);
    EXPECT_EQ(second.false_negatives, 1);
    EXPECT_EQ(second.false_positives, 1);
}

/** @brief Tests that a raised threshold moves the cut above a logit that passed one half. */
TEST_F(Medium_BinaryConfusion, TheThresholdMovesTheCut)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{1, 1}, DType::Float32, {1.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor truth,
                            tensor_of(Shape{1}, DType::Float32, {1.0}));
    const Train::ColumnOfLogits where{1, 1, 0};

    // sigmoid(1) is 0.731, between the two thresholds.
    EXPECT_EQ(Train::count_confusion(logits, truth, where, Train::logit_of(0.5))
                  .true_positives, 1);
    EXPECT_EQ(Train::count_confusion(logits, truth, where, Train::logit_of(0.9))
                  .false_negatives, 1);
}

/** @brief Tests that every ratio of an empty count is 0 and none divides by zero. */
TEST(Small_BinaryConfusionRatios, ARatioOverNothingIsZero)
{
    const BinaryConfusion none{};
    EXPECT_EQ(Train::accuracy_of(none), 0.0);
    EXPECT_EQ(Train::precision_of(none), 0.0);
    EXPECT_EQ(Train::recall_of(none), 0.0);
    EXPECT_EQ(Train::f1_of(none), 0.0);

    BinaryConfusion rejections{};
    rejections.true_negatives = 4;
    EXPECT_EQ(Train::accuracy_of(rejections), 1.0);
    EXPECT_EQ(Train::f1_of(rejections), 0.0);
}

/** @brief Tests that logit_of inverts the sigmoid at one half, above it and below it. */
TEST(Small_DecisionThreshold, LogitOfInvertsTheSigmoid)
{
    EXPECT_EQ(Train::logit_of(0.5), 0.0);
    for (const double probability : {0.1, 0.731, 0.9})
    {
        const double logit = Train::logit_of(probability);
        EXPECT_NEAR(1.0 / (1.0 + std::exp(-logit)), probability, K_RATIO_TOLERANCE);
    }
}

namespace
{
    /** @brief Parses @p text, a JSON document the test states; a refusal stops the test. */
    Config::Value parsed(std::string_view text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that the reader returns the stated threshold and one half for none. */
TEST(Small_DecisionThreshold, TheReaderDefaultsToOneHalf)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const double absent,
        Train::read_threshold(parsed("{}"), "objective"));
    EXPECT_EQ(absent, Train::K_DEFAULT_THRESHOLD);

    SA_ASSERT_OK_AND_ASSIGN(
        const double stated,
        Train::read_threshold(parsed(R"({"threshold": 0.25})"),
                              "objective"));
    EXPECT_EQ(stated, 0.25);
}

/** @brief Tests that another key, a value that is no number and a bound are each refused. */
TEST(Small_DecisionThreshold, TheReaderRefusesWhatIsNoProbability)
{
    SA_EXPECT_FAILURE(
        Train::read_threshold(parsed(R"({"cut": 0.5})"), "objective"),
        StatusCode::InvalidDescription, "objective", "'cut'", "'threshold'");
    SA_EXPECT_FAILURE(
        Train::read_threshold(parsed(R"({"threshold": "high"})"),
                              "objective"),
        StatusCode::InvalidDescription, "objective", "number");

    for (const char* const text : {R"({"threshold": 0})", R"({"threshold": 1})",
                                   R"({"threshold": -0.5})", R"({"threshold": 1.5})"})
    {
        SA_EXPECT_FAILURE(Train::read_threshold(parsed(text), "objective"),
                          StatusCode::InvalidDescription, "above 0 and below 1");
    }
}

/** @brief Tests that the host read refuses an integer tensor and another element count. */
TEST_F(Medium_BinaryConfusion, TheHostReadRefusesWhatItCannotRead)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor reals,
                            tensor_of(Shape{2, 2}, DType::Float64, {0.5, 1.5, 2.5, 3.5}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor integers,
                            allocate_tensor(*engine, Shape{4}, DType::Int32));

    SA_EXPECT_OK(Train::require_host_reals(reals, 4, "the logits"));
    EXPECT_EQ(Train::host_real(reals, 2), 2.5);
    SA_EXPECT_FAILURE(Train::require_host_reals(integers, 4, "the logits"),
                      StatusCode::InvalidDescription, "host read of the logits", "int32");
    SA_EXPECT_FAILURE(Train::require_host_reals(reals, 6, "the logits"),
                      StatusCode::Incompatible, "6 values", "holds 4");
}

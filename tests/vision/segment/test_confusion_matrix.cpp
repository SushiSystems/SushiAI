/**************************************************************************/
/* test_confusion_matrix.cpp                                              */
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
 * @file test_confusion_matrix.cpp
 * @brief Tests the pixel counts of a confusion matrix, its ignore index and its refusals.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Vision::Segment::check_confusion_class_count;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Holds the index the tests mark a pixel of no class with. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Views @p classes as the span add() reads. */
    SushiRuntime::span<const std::int32_t> view(const Classes& classes)
    {
        return SushiRuntime::span<const std::int32_t>(classes.data(), classes.size());
    }

    /** @brief Names a class count make refuses and what its refusal holds. */
    struct RefusedClassCount
    {
        std::int32_t classes;     /**< The class count given to make. */
        std::string_view first;   /**< A text the refusal holds. */
        std::string_view second;  /**< Another text the refusal holds. */
    };

    /** @brief Gives each class count outside the bounds to a test. */
    class Small_SegmentConfusionMatrixClassCount
        : public ::testing::TestWithParam<RefusedClassCount>
    {
    };

    /** @brief Gives each class count at the edge of the bounds to a test. */
    class Small_SegmentConfusionMatrixEdge : public ::testing::TestWithParam<std::int32_t> {};

    /** @brief Names a batch add cannot count, and what its refusal holds. */
    struct RefusedBatch
    {
        std::string_view name;    /**< The name of the row. */
        Classes predicted;        /**< The predicted class of each pixel. */
        Classes truth;            /**< The true class of each pixel. */
        std::string_view first;   /**< A text the refusal holds. */
        std::string_view second;  /**< Another text the refusal holds. */
        std::string_view third;   /**< Another text the refusal holds, or empty. */
        std::string_view fourth;  /**< Another text the refusal holds, or empty. */
    };

    /** @brief Gives each batch add refuses to a test. */
    class Small_SegmentConfusionMatrixBatch : public ::testing::TestWithParam<RefusedBatch> {};

    /** @brief Names a class count row in the name of its test. */
    std::string count_name(const ::testing::TestParamInfo<RefusedClassCount>& info)
    {
        return "classes_" + std::to_string(info.param.classes);
    }

    /** @brief Names an edge class count in the name of its test. */
    std::string edge_name(const ::testing::TestParamInfo<std::int32_t>& info)
    {
        return "classes_" + std::to_string(info.param);
    }

    /** @brief Names a refused batch in the name of its test. */
    std::string batch_name(const ::testing::TestParamInfo<RefusedBatch>& info)
    {
        return Test::parameter_name(info.param.name);
    }
} // namespace

/** @brief Tests that a new matrix holds its class count and no pixel. */
TEST(Small_SegmentConfusionMatrix, ANewMatrixCountsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, ConfusionMatrix::make(3));

    EXPECT_EQ(matrix.class_count(), 3);
    EXPECT_EQ(matrix.pixel_count(), 0);
    EXPECT_EQ(matrix.count(2, 1), 0);
}

/** @brief Tests that each pixel is counted at its true row and its predicted column. */
TEST(Small_SegmentConfusionMatrix, AddCountsByTruthAndPrediction)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(3));
    const Classes predicted{0, 1, 1, 2, 2, 2};
    const Classes truth{0, 0, 1, 2, 1, 2};

    SA_EXPECT_OK(matrix.add(view(predicted), view(truth), K_IGNORE));

    EXPECT_EQ(matrix.count(0, 0), 1);
    EXPECT_EQ(matrix.count(0, 1), 1);
    EXPECT_EQ(matrix.count(1, 1), 1);
    EXPECT_EQ(matrix.count(1, 2), 1);
    EXPECT_EQ(matrix.count(2, 2), 2);
    EXPECT_EQ(matrix.count(1, 0), 0);
    EXPECT_EQ(matrix.truth_pixels(1), 2);
    EXPECT_EQ(matrix.predicted_pixels(2), 3);
    EXPECT_EQ(matrix.pixel_count(), 6);
}

/** @brief Tests that an ignored pixel is counted nowhere, whatever was predicted for it. */
TEST(Small_SegmentConfusionMatrix, AnIgnoredPixelIsNotCounted)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));
    const Classes predicted{0, 1, 7};
    const Classes truth{0, K_IGNORE, K_IGNORE};

    SA_EXPECT_OK(matrix.add(view(predicted), view(truth), K_IGNORE));

    EXPECT_EQ(matrix.pixel_count(), 1);
    EXPECT_EQ(matrix.count(0, 0), 1);
    EXPECT_EQ(matrix.predicted_pixels(1), 0);
}

/** @brief Tests that two batches added one after the other give the sum of their counts. */
TEST(Small_SegmentConfusionMatrix, CountsAccumulateOverBatches)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));
    const Classes predicted{0, 1, 1};
    const Classes truth{0, 0, 1};

    SA_EXPECT_OK(matrix.add(view(predicted), view(truth), K_IGNORE));
    SA_EXPECT_OK(matrix.add(view(predicted), view(truth), K_IGNORE));

    EXPECT_EQ(matrix.count(0, 0), 2);
    EXPECT_EQ(matrix.count(0, 1), 2);
    EXPECT_EQ(matrix.count(1, 1), 2);
    EXPECT_EQ(matrix.pixel_count(), 6);
}

/** @brief Tests that merge adds another matrix count by count. */
TEST(Small_SegmentConfusionMatrix, MergeAddsAnotherMatrixCountByCount)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix first, ConfusionMatrix::make(2));
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix second, ConfusionMatrix::make(2));
    const Classes predicted{1, 1};
    const Classes truth{0, 1};
    SA_EXPECT_OK(first.add(view(predicted), view(truth), K_IGNORE));
    SA_EXPECT_OK(second.add(view(predicted), view(truth), K_IGNORE));

    SA_EXPECT_OK(first.merge(second));

    EXPECT_EQ(first.count(0, 1), 2);
    EXPECT_EQ(first.count(1, 1), 2);
}

/** @brief Tests that clear sets every count to zero. */
TEST(Small_SegmentConfusionMatrix, ClearSetsEveryCountToZero)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));
    const Classes predicted{1, 1};
    const Classes truth{0, 1};
    SA_EXPECT_OK(matrix.add(view(predicted), view(truth), K_IGNORE));

    matrix.clear();

    EXPECT_EQ(matrix.count(0, 1), 0);
    EXPECT_EQ(matrix.count(1, 1), 0);
    EXPECT_EQ(matrix.pixel_count(), 0);
}

/** @brief Tests that make refuses a class count outside [1, K_MAX_CONFUSION_CLASSES]. */
TEST_P(Small_SegmentConfusionMatrixClassCount, MakeRefusesAClassCountOutsideItsBounds)
{
    const RefusedClassCount& row = GetParam();

    SA_EXPECT_FAILURE(ConfusionMatrix::make(row.classes), StatusCode::InvalidDescription,
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Counts, Small_SegmentConfusionMatrixClassCount,
    ::testing::Values(RefusedClassCount{0, "ConfusionMatrix", "0 classes"},
                      RefusedClassCount{Vision::Segment::K_MAX_CONFUSION_CLASSES + 1,
                                        "4097 classes", "4096"}),
    count_name);

/**
 * @brief Tests that the check accepts a class count at either edge of its bounds.
 *
 * The check allocates nothing, so the largest count is tried without building the 128 MiB matrix.
 */
TEST_P(Small_SegmentConfusionMatrixEdge, TheCheckAcceptsAClassCountAtTheEdgeOfItsBounds)
{
    SA_EXPECT_OK(check_confusion_class_count(GetParam()));
}

/** @brief Tests that make builds a matrix over the class count it is given. */
TEST(Small_SegmentConfusionMatrixMake, MakeBuildsAMatrixOverTheGivenClassCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, ConfusionMatrix::make(1));

    EXPECT_EQ(matrix.class_count(), 1);
}

INSTANTIATE_TEST_SUITE_P(Counts, Small_SegmentConfusionMatrixEdge,
                         ::testing::Values(1, Vision::Segment::K_MAX_CONFUSION_CLASSES),
                         edge_name);

/** @brief Tests that add refuses a batch it cannot count and changes no count. */
TEST_P(Small_SegmentConfusionMatrixBatch, AddRefusesWithoutCounting)
{
    const RefusedBatch& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));

    SA_EXPECT_FAILURE(matrix.add(view(row.predicted), view(row.truth), K_IGNORE),
                      StatusCode::Incompatible, row.first, row.second, row.third, row.fourth);

    // The first pixel of each refused batch was valid and must not have been counted.
    EXPECT_EQ(matrix.pixel_count(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Batches, Small_SegmentConfusionMatrixBatch,
    ::testing::Values(
        RefusedBatch{"UnequalSpans", {0, 1}, {0, 1, 1}, "2 predicted", "3 true", "", ""},
        RefusedBatch{"TruthOutOfBounds", {0, 1}, {0, 2}, "true class of pixel 1", "is 2",
                     "below 2", "255"},
        RefusedBatch{"PredictionOutOfBounds", {0, -1}, {0, 1}, "predicted class of pixel 1",
                     "is -1", "", ""}),
    batch_name);

/** @brief Tests that merge refuses a matrix of another class count. */
TEST(Small_SegmentConfusionMatrix, MergeRefusesAnotherClassCount)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix two, ConfusionMatrix::make(2));
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix three, ConfusionMatrix::make(3));

    SA_EXPECT_FAILURE(two.merge(three), StatusCode::Incompatible, "2 classes", "of 3");
}

/** @brief Tests that a refused merge leaves every count of the receiving matrix as it was. */
TEST(Small_SegmentConfusionMatrix, ARefusedMergeChangesNoCount)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix two, ConfusionMatrix::make(2));
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix three, ConfusionMatrix::make(3));
    const Classes predicted{0, 1, 1};
    const Classes truth{0, 0, 1};
    SA_EXPECT_OK(two.add(view(predicted), view(truth), K_IGNORE));

    EXPECT_FALSE(two.merge(three).is_ok());

    EXPECT_EQ(two.count(0, 0), 1);
    EXPECT_EQ(two.count(0, 1), 1);
    EXPECT_EQ(two.count(1, 1), 1);
    EXPECT_EQ(two.count(1, 0), 0);
    EXPECT_EQ(two.pixel_count(), 3);
}

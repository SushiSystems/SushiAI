/**************************************************************************/
/* test_intersection_over_union.cpp                                       */
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
 * @file test_intersection_over_union.cpp
 * @brief Tests intersection over union and its mean.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Holds the index the tests mark a pixel of no class with. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Holds the rounded operations of one ratio of two whole counts: one division. */
    constexpr std::size_t K_RATIO_OPERATIONS = 1;

    /** @brief Holds the rounded operations of a mean of three ratios: three divisions, two sums, one division. */
    constexpr std::size_t K_MEAN_OPERATIONS = 6;

    /** @brief Names a class, its hand-counted intersection and its hand-counted union. */
    struct ClassRatio
    {
        std::int32_t class_index; /**< The class. */
        double intersection;      /**< The pixels predicted and true as the class. */
        double union_pixels;      /**< The pixels predicted or true as the class. */
    };

    /** @brief Gives each class of the counted matrix to a test. */
    class Small_SegmentClassIoU : public ::testing::TestWithParam<ClassRatio> {};

    /** @brief Names a class in the name of its test. */
    std::string class_name(const ::testing::TestParamInfo<ClassRatio>& info)
    {
        return "class_" + std::to_string(info.param.class_index);
    }

    /** @brief Returns a matrix of @p classes classes that counted the two lists. */
    Result<ConfusionMatrix> counted(std::int32_t classes, const Classes& predicted,
                                    const Classes& truth)
    {
        SA_TRY_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(classes));
        SA_TRY(matrix.add(
            SushiRuntime::span<const std::int32_t>(predicted.data(), predicted.size()),
            SushiRuntime::span<const std::int32_t>(truth.data(), truth.size()), K_IGNORE));

        return Result<ConfusionMatrix>(std::move(matrix));
    }

} // namespace

/**
 * @brief Tests each class against counts worked by hand: 1/2, 1/3 and 2/3. A ratio of two whole
 *        counts is one rounded division, so it is held to one unit roundoff of the exact ratio
 *        times the helper's threshold.
 */
TEST_P(Small_SegmentClassIoU, IsIntersectionOverUnion)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 1, 1, 2, 2, 2}, {0, 0, 1, 2, 1, 2}));
    const ClassRatio& row = GetParam();

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Vision::Segment::class_iou(matrix, row.class_index), row.intersection / row.union_pixels,
        K_RATIO_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(Classes, Small_SegmentClassIoU,
                         ::testing::Values(ClassRatio{0, 1.0, 2.0}, ClassRatio{1, 1.0, 3.0},
                                           ClassRatio{2, 2.0, 3.0}),
                         class_name);

/** @brief Tests that the mean of the class ratios 1/2, 1/3 and 2/3 is their mean. */
TEST(Small_SegmentMetrics, MeanIoUIsTheMeanOfTheClassRatios)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 1, 1, 2, 2, 2}, {0, 0, 1, 2, 1, 2}));

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Vision::Segment::mean_iou(matrix), (1.0 / 2.0 + 1.0 / 3.0 + 2.0 / 3.0) / 3.0,
        K_MEAN_OPERATIONS)));
}

/** @brief Tests that a class no true pixel holds is left out of the mean. */
TEST(Small_SegmentMetrics, TheMeanSkipsAClassThatDoesNotOccur)
{
    // Class 2 is predicted once and never true: its IoU is 0 and it does not occur.
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 0, 1, 2}, {0, 0, 1, 1}));

    EXPECT_TRUE(Vision::Segment::class_occurs(matrix, 0));
    EXPECT_FALSE(Vision::Segment::class_occurs(matrix, 2));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Vision::Segment::mean_iou(matrix), (1.0 + 1.0 / 2.0) / 2.0, K_MEAN_OPERATIONS)));
}

/** @brief Tests that the IoU of a class neither true nor predicted is 0. */
TEST(Small_SegmentMetrics, AClassNeitherTrueNorPredictedHasAnIoUOfZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, counted(3, {0, 1}, {0, 1}));

    EXPECT_EQ(Vision::Segment::class_iou(matrix, 2), 0.0);
}

/** @brief Tests that a matrix that counted nothing gives zero for both means. */
TEST(Small_SegmentMetrics, AnEmptyMatrixGivesZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, counted(2, {0}, {K_IGNORE}));

    EXPECT_EQ(Vision::Segment::mean_iou(matrix), 0.0);
    EXPECT_EQ(Vision::Segment::pixel_accuracy(matrix), 0.0);
}

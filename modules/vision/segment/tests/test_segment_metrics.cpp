/**************************************************************************/
/* test_segment_metrics.cpp                                               */
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
 * @file test_segment_metrics.cpp
 * @brief Tests intersection over union, its mean, pixel accuracy and the class pick by hand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Holds the index the tests mark a pixel of no class with. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Bounds the absolute error of a ratio of two counts below ten, in double. */
    constexpr double K_TOLERANCE = 1e-15;

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

/** @brief Tests each class against counts worked by hand: 1/2, 1/3 and 2/3. */
TEST(Unit_SegmentMetrics, ClassIoUIsIntersectionOverUnion)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 1, 1, 2, 2, 2}, {0, 0, 1, 2, 1, 2}));

    EXPECT_NEAR(Vision::Segment::class_iou(matrix, 0), 1.0 / 2.0, K_TOLERANCE);
    EXPECT_NEAR(Vision::Segment::class_iou(matrix, 1), 1.0 / 3.0, K_TOLERANCE);
    EXPECT_NEAR(Vision::Segment::class_iou(matrix, 2), 2.0 / 3.0, K_TOLERANCE);
    EXPECT_NEAR(Vision::Segment::mean_iou(matrix), (1.0 / 2.0 + 1.0 / 3.0 + 2.0 / 3.0) / 3.0,
                K_TOLERANCE);
}

/** @brief Tests that a class no true pixel holds is left out of the mean. */
TEST(Unit_SegmentMetrics, TheMeanSkipsAClassThatDoesNotOccur)
{
    // Class 2 is predicted once and never true: its IoU is 0 and it does not occur.
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 0, 1, 2}, {0, 0, 1, 1}));

    EXPECT_TRUE(Vision::Segment::class_occurs(matrix, 0));
    EXPECT_FALSE(Vision::Segment::class_occurs(matrix, 2));
    EXPECT_NEAR(Vision::Segment::class_iou(matrix, 2), 0.0, K_TOLERANCE);
    EXPECT_NEAR(Vision::Segment::mean_iou(matrix), (1.0 + 1.0 / 2.0) / 2.0, K_TOLERANCE);
}

/** @brief Tests that pixel accuracy is the diagonal over the counted pixels. */
TEST(Unit_SegmentMetrics, PixelAccuracyIsTheDiagonalOverTheCountedPixels)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 1, 1, 2, 2, 2, 0}, {0, 0, 1, 2, 1, 2, K_IGNORE}));

    EXPECT_NEAR(Vision::Segment::pixel_accuracy(matrix), 4.0 / 6.0, K_TOLERANCE);
}

/** @brief Tests that a matrix that counted nothing gives zero for both means. */
TEST(Unit_SegmentMetrics, AnEmptyMatrixGivesZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, counted(2, {0}, {K_IGNORE}));

    EXPECT_EQ(Vision::Segment::mean_iou(matrix), 0.0);
    EXPECT_EQ(Vision::Segment::pixel_accuracy(matrix), 0.0);
}

/** @brief Tests that the pick is the largest score of each pixel, the lowest class on a tie. */
TEST(Unit_SegmentMetrics, ArgmaxPicksTheLargestScoreAndTheLowestClassOnATie)
{
    const std::vector<float> scores{0.1f, 0.9f, 0.0f, 2.0f, -1.0f, 2.0f, -3.0f, -2.0f, -1.0f};
    Classes picked(3, -1);

    Vision::Segment::argmax_classes(
        SushiRuntime::span<const float>(scores.data(), scores.size()), 3,
        SushiRuntime::span<std::int32_t>(picked.data(), picked.size()));

    EXPECT_EQ(picked, (Classes{1, 0, 2}));
}

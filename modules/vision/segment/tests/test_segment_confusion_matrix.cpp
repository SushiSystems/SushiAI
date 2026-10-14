/**************************************************************************/
/* test_segment_confusion_matrix.cpp                                      */
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
 * @file test_segment_confusion_matrix.cpp
 * @brief Tests the pixel counts of a confusion matrix, its ignore index and its refusals.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
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
} // namespace

/** @brief Tests that a new matrix holds its class count and no pixel. */
TEST(Unit_SegmentConfusionMatrix, ANewMatrixCountsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix, ConfusionMatrix::make(3));

    EXPECT_EQ(matrix.class_count(), 3);
    EXPECT_EQ(matrix.pixel_count(), 0);
    EXPECT_EQ(matrix.count(2, 1), 0);
}

/** @brief Tests that each pixel is counted at its true row and its predicted column. */
TEST(Unit_SegmentConfusionMatrix, AddCountsByTruthAndPrediction)
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
TEST(Unit_SegmentConfusionMatrix, AnIgnoredPixelIsNotCounted)
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
TEST(Unit_SegmentConfusionMatrix, CountsAccumulateOverBatches)
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

/** @brief Tests that merge adds another matrix count by count and clear empties one. */
TEST(Unit_SegmentConfusionMatrix, MergeAddsAndClearEmpties)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix first, ConfusionMatrix::make(2));
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix second, ConfusionMatrix::make(2));
    const Classes predicted{1, 1};
    const Classes truth{0, 1};
    SA_EXPECT_OK(first.add(view(predicted), view(truth), K_IGNORE));
    SA_EXPECT_OK(second.add(view(predicted), view(truth), K_IGNORE));

    SA_EXPECT_OK(first.merge(second));
    second.clear();

    EXPECT_EQ(first.count(0, 1), 2);
    EXPECT_EQ(first.count(1, 1), 2);
    EXPECT_EQ(second.pixel_count(), 0);
}

/** @brief Tests that make refuses a class count outside [1, K_MAX_CONFUSION_CLASSES]. */
TEST(Unit_SegmentConfusionMatrix, MakeRefusesAClassCountOutsideItsBounds)
{
    SA_EXPECT_FAILURE(ConfusionMatrix::make(0), StatusCode::InvalidDescription,
                      "ConfusionMatrix", "0 classes");
    SA_EXPECT_FAILURE(ConfusionMatrix::make(Vision::Segment::K_MAX_CONFUSION_CLASSES + 1),
                      StatusCode::InvalidDescription, "4097 classes", "4096");
}

/** @brief Tests that add refuses unequal spans and a class out of bounds, changing nothing. */
TEST(Unit_SegmentConfusionMatrix, AddRefusesWithoutCounting)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));
    const Classes two{0, 1};
    const Classes three{0, 1, 1};
    const Classes bad_truth{0, 2};
    const Classes bad_prediction{0, -1};

    SA_EXPECT_FAILURE(matrix.add(view(two), view(three), K_IGNORE), StatusCode::Incompatible,
                      "2 predicted", "3 true");
    SA_EXPECT_FAILURE(matrix.add(view(two), view(bad_truth), K_IGNORE),
                      StatusCode::Incompatible, "true class of pixel 1", "is 2", "below 2",
                      "255");
    SA_EXPECT_FAILURE(matrix.add(view(bad_prediction), view(two), K_IGNORE),
                      StatusCode::Incompatible, "predicted class of pixel 1", "is -1");

    // The first pixel of each refused batch was valid and must not have been counted.
    EXPECT_EQ(matrix.pixel_count(), 0);
}

/** @brief Tests that merge refuses a matrix of another class count. */
TEST(Unit_SegmentConfusionMatrix, MergeRefusesAnotherClassCount)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix two, ConfusionMatrix::make(2));
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix three, ConfusionMatrix::make(3));

    SA_EXPECT_FAILURE(two.merge(three), StatusCode::Incompatible, "2 classes", "of 3");
}

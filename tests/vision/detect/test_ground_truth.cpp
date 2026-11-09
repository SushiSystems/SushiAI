/**************************************************************************/
/* test_ground_truth.cpp                                                  */
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
 * @file test_ground_truth.cpp
 * @brief Tests the refill of a used batch, the scaling of the corners and the refused rows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::GroundTruthBatch;
    using SushiAI::Vision::Detect::pad_ground_truth;

    /**
     * @brief Pads a batch of two images with three rows in the first, then pads one image of one
     *        box into the same object, and returns it.
     */
    GroundTruthBatch pad_into_a_used_batch()
    {
        const std::vector<std::int64_t> wide_images{0, 0, 0, 1};
        const std::vector<std::int64_t> wide_classes{1, 2, 3, 4};
        const std::vector<float> wide_boxes(16, 0.5f);
        GroundTruthBatch padded;
        SA_EXPECT_OK(pad_ground_truth({wide_images, wide_classes, wide_boxes}, 2, 32.0f, 32.0f,
                                      padded));

        const std::vector<std::int64_t> images{0};
        const std::vector<std::int64_t> classes{6};
        const std::vector<float> boxes{0.5f, 0.5f, 0.5f, 0.5f};
        SA_EXPECT_OK(pad_ground_truth({images, classes, boxes}, 1, 32.0f, 32.0f, padded));
        return padded;
    }
} // namespace

/** @brief Tests that padding into a used batch keeps no row of the earlier batch. */
TEST(Small_DetectGroundTruth, PaddingIntoAUsedBatchKeepsNoEarlierRow)
{
    const GroundTruthBatch padded = pad_into_a_used_batch();

    EXPECT_EQ(padded.batch, 1);
    EXPECT_EQ(padded.max_boxes, 1);
    EXPECT_EQ(padded.labels, std::vector<std::int64_t>{6});
    EXPECT_EQ(padded.valid, std::vector<std::uint8_t>{1});
    EXPECT_EQ(padded.boxes.size(), 4u);
}

/**
 * @brief Tests the corners of the refilled batch.
 *
 * One image of one box: the centre (16, 16) and the size (16, 16) give corners 8 and 24, exact in
 * float32.
 */
TEST(Small_DetectGroundTruth, PaddingIntoAUsedBatchWritesTheCornersOfTheNewRow)
{
    const GroundTruthBatch padded = pad_into_a_used_batch();

    EXPECT_EQ(padded.boxes, (std::vector<float>{8.0f, 8.0f, 24.0f, 24.0f}));
}

/**
 * @brief Tests that x and width scale with the image width and y and height with its height.
 *
 * On a 64 by 32 input the centre (0.5, 0.25) and the size (0.25, 0.5) give the centre (32, 8) and
 * the size (16, 16), so the corners are 24, 0, 40 and 16. Every other case passes 32 for both.
 */
TEST(Small_DetectGroundTruth, ScalesXByTheWidthAndYByTheHeight)
{
    const std::vector<std::int64_t> images{0};
    const std::vector<std::int64_t> classes{1};
    const std::vector<float> boxes{0.5f, 0.25f, 0.25f, 0.5f};
    GroundTruthBatch padded;

    SA_EXPECT_OK(pad_ground_truth({images, classes, boxes}, 1, 64.0f, 32.0f, padded));

    EXPECT_EQ(padded.boxes, (std::vector<float>{24.0f, 0.0f, 40.0f, 16.0f}));
}

/** @brief Tests that a batch without images is refused. */
TEST(Small_DetectGroundTruth, RefusesNoImages)
{
    const std::vector<std::int64_t> images{0, 0};
    const std::vector<float> boxes(8, 0.5f);
    GroundTruthBatch padded;

    SA_EXPECT_FAILURE(pad_ground_truth({images, images, boxes}, 0, 32.0f, 32.0f, padded),
                      StatusCode::InvalidDescription, "positive batch", "batch 0");
}

/** @brief Tests that row arrays of unequal lengths are refused. */
TEST(Small_DetectGroundTruth, RefusesRowArraysOfUnequalLengths)
{
    const std::vector<std::int64_t> images{0, 0};
    const std::vector<std::int64_t> classes{1};
    const std::vector<float> boxes(8, 0.5f);
    GroundTruthBatch padded;

    SA_EXPECT_FAILURE(pad_ground_truth({images, classes, boxes}, 1, 32.0f, 32.0f, padded),
                      StatusCode::InvalidDescription, "[n], [n] and [n, 4] rows",
                      "2, 1 and 8 values");
}

/** @brief Tests that a row naming a negative image index is refused, as a row past the batch is. */
TEST(Small_DetectGroundTruth, RefusesARowThatNamesANegativeImage)
{
    const std::vector<std::int64_t> images{-1};
    const std::vector<std::int64_t> classes{1};
    const std::vector<float> boxes{0.5f, 0.5f, 0.5f, 0.5f};
    GroundTruthBatch padded;

    SA_EXPECT_FAILURE(pad_ground_truth({images, classes, boxes}, 2, 32.0f, 32.0f, padded),
                      StatusCode::InvalidDescription, "image -1", "batch of 2");
}

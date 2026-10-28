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
 * @file test_ground_truth.cpp
 * @brief Tests the empty ground truth, the refill of a used batch and the refused row arrays.
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
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::GroundTruthBatch;
    using SushiAI::Vision::Detect::pad_ground_truth;

    /** @brief Holds the tolerance of a corner that is a multiple of 8, which float32 holds. */
    constexpr float K_EXACT = 0.0f;
} // namespace

/** @brief Tests that a default batch has no images, no rows and no arrays. */
TEST(Small_DetectGroundTruth, ADefaultBatchIsEmpty)
{
    const GroundTruthBatch padded;

    EXPECT_EQ(padded.batch, 0);
    EXPECT_EQ(padded.max_boxes, 0);
    EXPECT_TRUE(padded.labels.empty());
    EXPECT_TRUE(padded.boxes.empty());
    EXPECT_TRUE(padded.valid.empty());
}

/** @brief Tests that padding into a used batch keeps no row of the earlier batch. */
TEST(Small_DetectGroundTruth, PaddingIntoAUsedBatchKeepsNoEarlierRow)
{
    const std::vector<std::int64_t> wide_images{0, 0, 0, 1};
    const std::vector<std::int64_t> wide_classes{1, 2, 3, 4};
    const std::vector<float> wide_boxes(16, 0.5f);
    GroundTruthBatch padded;
    SA_EXPECT_OK(pad_ground_truth({wide_images, wide_classes, wide_boxes}, 2, 32.0f, 32.0f,
                                  padded));
    ASSERT_EQ(padded.max_boxes, 3);

    const std::vector<std::int64_t> images{0};
    const std::vector<std::int64_t> classes{6};
    const std::vector<float> boxes{0.5f, 0.5f, 0.5f, 0.5f};
    SA_EXPECT_OK(pad_ground_truth({images, classes, boxes}, 1, 32.0f, 32.0f, padded));

    // One image of one box: the centre (16, 16) and the size (16, 16) give corners 8 and 24.
    EXPECT_EQ(padded.batch, 1);
    EXPECT_EQ(padded.max_boxes, 1);
    EXPECT_EQ(padded.labels, std::vector<std::int64_t>{6});
    EXPECT_EQ(padded.valid, std::vector<std::uint8_t>{1});
    const std::vector<float> expected{8.0f, 8.0f, 24.0f, 24.0f};
    ASSERT_EQ(padded.boxes.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(padded.boxes[i], expected[i], K_EXACT) << "value " << i;
}

/** @brief Tests that a batch without images and row arrays of unequal lengths are refused. */
TEST(Small_DetectGroundTruth, RefusesNoImagesAndRowArraysOfUnequalLengths)
{
    const std::vector<std::int64_t> images{0, 0};
    const std::vector<std::int64_t> classes{1};
    const std::vector<float> boxes(8, 0.5f);
    GroundTruthBatch padded;

    SA_EXPECT_FAILURE(pad_ground_truth({images, images, boxes}, 0, 32.0f, 32.0f, padded),
                      StatusCode::InvalidDescription, "positive batch", "batch 0");
    SA_EXPECT_FAILURE(pad_ground_truth({images, classes, boxes}, 1, 32.0f, 32.0f, padded),
                      StatusCode::InvalidDescription, "[n], [n] and [n, 4] rows",
                      "2, 1 and 8 values");
}

/**************************************************************************/
/* test_anchors.cpp                                                       */
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
 * @file test_anchors.cpp
 * @brief Tests the anchor grid and its leaves against hand cases.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Free::AnchorLeaves;
    using SushiAI::Vision::Detect::Free::FeatureLevel;
    using SushiAI::Vision::Detect::Free::LevelPlacement;

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_REFERENCE_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f},
                                                       {20, 20, 32.0f}};

    /** @brief Holds a two-level grid small enough to write out: 2 by 3 cells, then 1 by 2. */
    const std::vector<FeatureLevel> K_SMALL_LEVELS{{2, 3, 8.0f}, {1, 2, 16.0f}};
}

/** @brief Tests that the anchor count is the sum of height times width over the levels. */
TEST(Small_DetectAnchors, CountsOneAnchorPerCell)
{
    EXPECT_EQ(require_value(
                  SushiAI::Vision::Detect::Free::count_anchors(K_SMALL_LEVELS)),
              8);
    EXPECT_EQ(require_value(
                  SushiAI::Vision::Detect::Free::count_anchors(K_REFERENCE_LEVELS)),
              8400);
}

/** @brief Tests that points are cell centres, levels in order and row-major inside a level. */
TEST(Small_DetectAnchors, OrdersPointsByLevelThenRowThenColumn)
{
    std::vector<float> points(16);
    std::vector<float> strides(8);
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides));

    const std::vector<float> expected_points{0.5f, 0.5f, 1.5f, 0.5f, 2.5f, 0.5f, 0.5f, 1.5f,
                                             1.5f, 1.5f, 2.5f, 1.5f, 0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> expected_strides{8.0f, 8.0f, 8.0f,  8.0f,
                                              8.0f, 8.0f, 16.0f, 16.0f};
    EXPECT_EQ(points, expected_points);
    EXPECT_EQ(strides, expected_strides);
}

/** @brief Tests that scaling multiplies both coordinates of a point by its anchor's stride. */
TEST(Small_DetectAnchors, ScalesEachPointByItsStride)
{
    std::vector<float> points(16);
    std::vector<float> strides(8);
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides));

    std::vector<float> pixels(16);
    SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels);

    // 0.5 * 8 = 4, 1.5 * 8 = 12, 2.5 * 8 = 20; on the second level 0.5 * 16 = 8, 1.5 * 16 = 24.
    const std::vector<float> expected{4.0f,  4.0f,  12.0f, 4.0f,  20.0f, 4.0f, 4.0f,  12.0f,
                                      12.0f, 12.0f, 20.0f, 12.0f, 8.0f,  8.0f, 24.0f, 8.0f};
    EXPECT_EQ(pixels, expected);
}

/** @brief Tests that the size check refuses strides that do not number one per point. */
TEST(Small_DetectAnchors, ScaleRejectsStridesThatDoNotMatchThePoints)
{
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::check_scale_sizes(16, 7, 16),
                      StatusCode::InvalidDescription, "[A] strides", "got 16, 7 and 16 values");
}

/** @brief Tests that scaling stops the process on strides that do not number one per point. */
TEST(Medium_DetectAnchorsFatal, ScaleStopsOnStridesThatDoNotMatchThePoints)
{
    const std::vector<float> points(16, 0.5f);
    const std::vector<float> strides(7, 8.0f);
    std::vector<float> pixels(16);
    SA_EXPECT_FATAL(SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels),
                    "[A] strides", "got 16, 7 and 16 values");
}

/** @brief Tests that a level without cells or without a stride is refused by name. */
TEST(Small_DetectAnchors, RejectsALevelWithoutCellsOrStride)
{
    const std::vector<FeatureLevel> no_rows{{0, 3, 8.0f}};
    const std::vector<FeatureLevel> no_stride{{2, 3, 0.0f}};
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::count_anchors(no_rows),
                      StatusCode::InvalidDescription, "height 0", "width 3");
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::count_anchors(no_stride),
                      StatusCode::InvalidDescription, "stride 0");
}

/** @brief Tests that output arrays of the wrong size are refused with both sizes named. */
TEST(Small_DetectAnchors, RejectsOutputsOfTheWrongSize)
{
    std::vector<float> points(15);
    std::vector<float> strides(8);
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides),
                      StatusCode::InvalidDescription, "16 point values", "got 15");
}

/** @brief Tests that the pixel anchor points equal the reference's "anchors.points" exactly. */
TEST(Medium_DetectAnchors, MatchesTheReferenceAnchorPoints)
{
    SA_OPEN_REFERENCE_OR_SKIP(loss, "yolo26/loss.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, loss, "anchors.points");

    std::vector<float> points(8400 * 2);
    std::vector<float> strides(8400);
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::make_anchors(K_REFERENCE_LEVELS, points, strides));
    std::vector<float> pixels(points.size());
    SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels);

    ASSERT_EQ(reference.shape, (SushiAI::Shape{8400, 2}));
    EXPECT_EQ(pixels, reference.floats());
}

/** @brief Tests that each level's strides fill the anchor range the reference manifest records. */
TEST(Medium_DetectAnchors, GivesEachLevelItsStrideOverItsAnchorRange)
{
    std::vector<float> points(8400 * 2);
    std::vector<float> strides(8400);
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::make_anchors(K_REFERENCE_LEVELS, points, strides));

    // Offsets 0, 6400 and 8000 are "anchor_offset" of the three levels in MANIFEST.json.
    const std::vector<float> first(strides.begin(), strides.begin() + 6400);
    const std::vector<float> second(strides.begin() + 6400, strides.begin() + 8000);
    const std::vector<float> third(strides.begin() + 8000, strides.end());
    EXPECT_EQ(first, std::vector<float>(6400, 8.0f));
    EXPECT_EQ(second, std::vector<float>(1600, 16.0f));
    EXPECT_EQ(third, std::vector<float>(400, 32.0f));
}

/** @brief Tests that the leaves hold the grid-unit points and the strides once per image. */
TEST(Small_DetectAnchorLeaves, RepeatsThePointsAndTheStridesForEveryImage)
{
    AnchorLeaves leaves;
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 2, leaves));

    // One image is the eight points of OrdersPointsByLevelThenRowThenColumn; the second repeats it.
    const std::vector<float> image_points{0.5f, 0.5f, 1.5f, 0.5f, 2.5f, 0.5f, 0.5f, 1.5f,
                                          1.5f, 1.5f, 2.5f, 1.5f, 0.5f, 0.5f, 1.5f, 0.5f};
    std::vector<float> expected_points(image_points);
    expected_points.insert(expected_points.end(), image_points.begin(), image_points.end());

    // Six anchors of stride 8 and two of stride 16, four values each, in both images.
    std::vector<float> image_strides(24, 8.0f);
    image_strides.insert(image_strides.end(), 8, 16.0f);
    std::vector<float> expected_strides(image_strides);
    expected_strides.insert(expected_strides.end(), image_strides.begin(), image_strides.end());

    EXPECT_EQ(leaves.points, expected_points);
    EXPECT_EQ(leaves.strides, expected_strides);
}

/** @brief Tests that split_level cuts the leaves into one level's grid of points and strides. */
TEST(Small_DetectAnchorLeaves, SplitsIntoALevelsPointsAndStrides)
{
    AnchorLeaves leaves;
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 2, leaves));

    // The second level is anchors 6 and 7 of 8: a 1 by 2 grid, so [2, 1, 2, 2] and [2, 1, 2, 4].
    std::vector<float> points(2 * 2 * 2);
    std::vector<float> strides(2 * 2 * 4);
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::split_level(leaves.points,
                                                            LevelPlacement{2, 8, 2, 6, 2},
                                                            points));
    SA_EXPECT_OK(SushiAI::Vision::Detect::Free::split_level(leaves.strides,
                                                            LevelPlacement{2, 8, 4, 6, 2},
                                                            strides));

    const std::vector<float> expected_points{0.5f, 0.5f, 1.5f, 0.5f, 0.5f, 0.5f, 1.5f, 0.5f};
    EXPECT_EQ(points, expected_points);
    EXPECT_EQ(strides, std::vector<float>(16, 16.0f));
}

/** @brief Tests that leaves are refused for a batch without images. */
TEST(Small_DetectAnchorLeaves, RejectsABatchWithoutImages)
{
    AnchorLeaves leaves;
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 0, leaves),
                      StatusCode::InvalidDescription, "positive batch", "batch 0 and 2 levels");
}

/** @brief Tests that leaves are refused for a model without levels. */
TEST(Small_DetectAnchorLeaves, RejectsAModelWithoutLevels)
{
    const std::vector<FeatureLevel> none;
    AnchorLeaves leaves;
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::Free::build_anchor_leaves(none, 1, leaves),
                      StatusCode::InvalidDescription, "at least one level", "batch 1 and 0 levels");
}

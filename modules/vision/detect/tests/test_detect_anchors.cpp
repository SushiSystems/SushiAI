/**************************************************************************/
/* test_detect_anchors.cpp                                                */
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
 * @file test_detect_anchors.cpp
 * @brief Tests the anchor grid, its leaves and the level placement against hand cases.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::AnchorLeaves;
    using SushiAI::Vision::Detect::Free::FeatureLevel;
    using SushiAI::Vision::Detect::Free::LevelPlacement;
    using SushiAI::Test::near_reference;

    /** @brief Holds the tolerance of values that are multiples of 0.5 and so exact. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_REFERENCE_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f},
                                                       {20, 20, 32.0f}};

    /** @brief Holds a two-level grid small enough to write out: 2 by 3 cells, then 1 by 2. */
    const std::vector<FeatureLevel> K_SMALL_LEVELS{{2, 3, 8.0f}, {1, 2, 16.0f}};
}

/** @brief Tests that the anchor count is the sum of height times width over the levels. */
TEST(Unit_DetectAnchors, CountsOneAnchorPerCell)
{
    EXPECT_EQ(SushiAI::Vision::Detect::Free::count_anchors(K_SMALL_LEVELS), 8);
    EXPECT_EQ(SushiAI::Vision::Detect::Free::count_anchors(K_REFERENCE_LEVELS), 8400);
}

/** @brief Tests that points are cell centres, levels in order and row-major inside a level. */
TEST(Unit_DetectAnchors, OrdersPointsByLevelThenRowThenColumn)
{
    std::vector<float> points(16);
    std::vector<float> strides(8);
    SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides);

    const std::vector<float> expected_points{0.5f, 0.5f, 1.5f, 0.5f, 2.5f, 0.5f, 0.5f, 1.5f,
                                             1.5f, 1.5f, 2.5f, 1.5f, 0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> expected_strides{8.0f, 8.0f, 8.0f,  8.0f,
                                              8.0f, 8.0f, 16.0f, 16.0f};
    EXPECT_TRUE(near_reference(points, expected_points, K_EXACT));
    EXPECT_TRUE(near_reference(strides, expected_strides, K_EXACT));
}

/** @brief Tests that scaling multiplies both coordinates of a point by its anchor's stride. */
TEST(Unit_DetectAnchors, ScalesEachPointByItsStride)
{
    std::vector<float> points(16);
    std::vector<float> strides(8);
    SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides);

    std::vector<float> pixels(16);
    SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels);

    // 0.5 * 8 = 4, 1.5 * 8 = 12, 2.5 * 8 = 20; on the second level 0.5 * 16 = 8, 1.5 * 16 = 24.
    const std::vector<float> expected{4.0f,  4.0f,  12.0f, 4.0f,  20.0f, 4.0f, 4.0f,  12.0f,
                                      12.0f, 12.0f, 20.0f, 12.0f, 8.0f,  8.0f, 24.0f, 8.0f};
    EXPECT_TRUE(near_reference(pixels, expected, K_EXACT));
}

/** @brief Tests that scaling refuses strides that do not number one per point. */
TEST(Unit_DetectAnchors, ScaleRejectsStridesThatDoNotMatchThePoints)
{
    const std::vector<float> points(16, 0.5f);
    const std::vector<float> strides(7, 8.0f);
    std::vector<float> pixels(16);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels),
                    "[A] strides", "got 16, 7 and 16 values");
}

/** @brief Tests that a level without cells or without a stride is refused by name. */
TEST(Unit_DetectAnchors, RejectsALevelWithoutCellsOrStride)
{
    const std::vector<FeatureLevel> no_rows{{0, 3, 8.0f}};
    const std::vector<FeatureLevel> no_stride{{2, 3, 0.0f}};
    SA_EXPECT_ERROR((void)SushiAI::Vision::Detect::Free::count_anchors(no_rows), "height 0",
                    "width 3");
    SA_EXPECT_ERROR((void)SushiAI::Vision::Detect::Free::count_anchors(no_stride), "stride 0");
}

/** @brief Tests that output arrays of the wrong size are refused with both sizes named. */
TEST(Unit_DetectAnchors, RejectsOutputsOfTheWrongSize)
{
    std::vector<float> points(15);
    std::vector<float> strides(8);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::make_anchors(K_SMALL_LEVELS, points, strides),
                    "16 point values", "got 15");
}

/** @brief Tests that the pixel anchor points equal the reference's "anchors.points" exactly. */
TEST(Unit_DetectAnchors, MatchesTheReferenceAnchorPoints)
{
    SA_OPEN_REFERENCE_OR_SKIP(loss, "yolo26/loss.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, loss, "anchors.points");

    std::vector<float> points(8400 * 2);
    std::vector<float> strides(8400);
    SushiAI::Vision::Detect::Free::make_anchors(K_REFERENCE_LEVELS, points, strides);
    std::vector<float> pixels(points.size());
    SushiAI::Vision::Detect::Free::scale_points(points, strides, pixels);

    ASSERT_EQ(reference.shape, (SushiAI::Shape{8400, 2}));
    EXPECT_TRUE(near_reference(pixels, reference.floats(), K_EXACT));
}

/** @brief Tests that each level's strides fill the anchor range the reference manifest records. */
TEST(Unit_DetectAnchors, GivesEachLevelItsStrideOverItsAnchorRange)
{
    std::vector<float> points(8400 * 2);
    std::vector<float> strides(8400);
    SushiAI::Vision::Detect::Free::make_anchors(K_REFERENCE_LEVELS, points, strides);

    // Offsets 0, 6400 and 8000 are "anchor_offset" of the three levels in MANIFEST.json.
    const std::vector<float> first(strides.begin(), strides.begin() + 6400);
    const std::vector<float> second(strides.begin() + 6400, strides.begin() + 8000);
    const std::vector<float> third(strides.begin() + 8000, strides.end());
    EXPECT_TRUE(near_reference(first, std::vector<float>(6400, 8.0f), K_EXACT));
    EXPECT_TRUE(near_reference(second, std::vector<float>(1600, 16.0f), K_EXACT));
    EXPECT_TRUE(near_reference(third, std::vector<float>(400, 32.0f), K_EXACT));
}

/** @brief Tests that the leaves hold the grid-unit points and the strides once per image. */
TEST(Unit_DetectAnchorLeaves, RepeatsThePointsAndTheStridesForEveryImage)
{
    AnchorLeaves leaves;
    SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 2, leaves);

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

    EXPECT_TRUE(near_reference(leaves.points, expected_points, K_EXACT));
    EXPECT_TRUE(near_reference(leaves.strides, expected_strides, K_EXACT));
}

/** @brief Tests that split_level cuts the leaves into one level's grid of points and strides. */
TEST(Unit_DetectAnchorLeaves, SplitsIntoALevelsPointsAndStrides)
{
    AnchorLeaves leaves;
    SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 2, leaves);

    // The second level is anchors 6 and 7 of 8: a 1 by 2 grid, so [2, 1, 2, 2] and [2, 1, 2, 4].
    std::vector<float> points(2 * 2 * 2);
    std::vector<float> strides(2 * 2 * 4);
    SushiAI::Vision::Detect::Free::split_level(leaves.points, LevelPlacement{2, 8, 2, 6, 2},
                                               points);
    SushiAI::Vision::Detect::Free::split_level(leaves.strides, LevelPlacement{2, 8, 4, 6, 2},
                                               strides);

    const std::vector<float> expected_points{0.5f, 0.5f, 1.5f, 0.5f, 0.5f, 0.5f, 1.5f, 0.5f};
    EXPECT_TRUE(near_reference(points, expected_points, K_EXACT));
    EXPECT_TRUE(near_reference(strides, std::vector<float>(16, 16.0f), K_EXACT));
}

/** @brief Tests that leaves are refused for a batch without images and for no levels. */
TEST(Unit_DetectAnchorLeaves, RejectsABatchWithoutImagesAndAModelWithoutLevels)
{
    const std::vector<FeatureLevel> none;
    AnchorLeaves leaves;
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::build_anchor_leaves(K_SMALL_LEVELS, 0, leaves),
                    "positive batch", "batch 0 and 2 levels");
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::build_anchor_leaves(none, 1, leaves),
                    "at least one level", "batch 1 and 0 levels");
}

/** @brief Tests that joining writes a level's rows at its anchor offset in every image. */
TEST(Unit_DetectLevelRows, JoinPlacesALevelAtItsAnchorOffset)
{
    // Two images, five anchors of two values; the level holds anchors 3 and 4.
    const LevelPlacement placement{2, 5, 2, 3, 2};
    const std::vector<float> level{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> joined(20, 0.0f);
    SushiAI::Vision::Detect::Free::join_level(level, placement, joined);

    const std::vector<float> expected{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    EXPECT_TRUE(near_reference(joined, expected, K_EXACT));
}

/** @brief Tests that splitting returns the rows that joining placed. */
TEST(Unit_DetectLevelRows, SplitReturnsTheRowsJoinPlaced)
{
    const LevelPlacement placement{2, 5, 2, 1, 3};
    std::vector<float> joined(20);
    for (std::size_t i = 0; i < joined.size(); ++i)
        joined[i] = static_cast<float>(i);

    std::vector<float> level(12);
    SushiAI::Vision::Detect::Free::split_level(joined, placement, level);

    // Image 0 holds values 0..9 and its anchors 1..3 are values 2..7; image 1 adds 10.
    const std::vector<float> expected{2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                      12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
    EXPECT_TRUE(near_reference(level, expected, K_EXACT));
}

/** @brief Tests that a level reaching past the last anchor is refused. */
TEST(Unit_DetectLevelRows, RejectsALevelPastTheLastAnchor)
{
    const LevelPlacement placement{1, 5, 2, 4, 2};
    const std::vector<float> level(4, 0.0f);
    std::vector<float> joined(10, 0.0f);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::join_level(level, placement, joined),
                    "2 anchors", "offset 4", "5 anchors");
}

/** @brief Tests that splitting into a level array of the wrong size is refused with both sizes. */
TEST(Unit_DetectLevelRows, SplitRejectsALevelArrayOfTheWrongSize)
{
    // Two images of three anchors and two values need 12 level values out of 20 joined ones.
    const LevelPlacement placement{2, 5, 2, 1, 3};
    const std::vector<float> joined(20, 0.0f);
    std::vector<float> level(10);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::Free::split_level(joined, placement, level),
                    "12 level values", "20 joined values", "got 10 and 20");
}

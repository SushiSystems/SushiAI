/**************************************************************************/
/* test_iou.cpp                                                           */
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
 * @file test_iou.cpp
 * @brief Tests intersection over union and its variants against hand cases.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/iou.hpp>

namespace
{
    using SushiAI::Test::near_reference;

    /** @brief Holds the tolerance of an IoU value, eight float32 steps at 1. */
    constexpr float K_IOU_TOLERANCE = 1e-6f;

} // namespace

/** @brief Tests that a box has IoU 1 and complete IoU 1 with itself. */
TEST(Unit_DetectBoxes, IdenticalBoxesScoreOne)
{
    const std::vector<float> box{0.0f, 0.0f, 2.0f, 2.0f};

    // intersection 2 * 2 = 4 and union 4 + 4 - 4 = 4; no centre shift and no aspect gap.
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(box, box), 4.0f / 4.0f,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(box, box), 1.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that disjoint boxes have IoU 0 and a complete IoU lowered by their distance. */
TEST(Unit_DetectBoxes, DisjointBoxesScoreZeroAndPayForDistance)
{
    const std::vector<float> left{0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> right{2.0f, 0.0f, 3.0f, 1.0f};

    const float hull_diagonal = 3.0f * 3.0f + 1.0f * 1.0f;
    const float centre_shift = 2.0f + 3.0f - 0.0f - 1.0f;
    const float centre_distance = (centre_shift * centre_shift) / 4.0f;
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(left, right), 0.0f,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(left, right),
                0.0f - centre_distance / hull_diagonal, K_IOU_TOLERANCE);
}

/** @brief Tests that partly overlapping squares give 1/7 and lose the centre term only. */
TEST(Unit_DetectBoxes, OverlappingSquaresLoseTheCentreTerm)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f};
    const std::vector<float> second{1.0f, 1.0f, 3.0f, 3.0f};

    const float iou = (1.0f * 1.0f) / (4.0f + 4.0f - 1.0f);
    const float hull_diagonal = 3.0f * 3.0f + 3.0f * 3.0f;
    const float centre_distance = (2.0f * 2.0f + 2.0f * 2.0f) / 4.0f;
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(first, second), iou,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(first, second),
                iou - centre_distance / hull_diagonal, K_IOU_TOLERANCE);
}

/** @brief Tests that boxes of different aspect ratio also lose the aspect term. */
TEST(Unit_DetectBoxes, DifferentAspectRatiosLoseTheAspectTerm)
{
    const std::vector<float> wide{0.0f, 0.0f, 4.0f, 2.0f};
    const std::vector<float> square{0.0f, 0.0f, 2.0f, 2.0f};

    const double pi = std::acos(-1.0);
    const double iou = (2.0 * 2.0) / (8.0 + 4.0 - 4.0);
    const double hull_diagonal = 4.0 * 4.0 + 2.0 * 2.0;
    const double centre_distance = ((0.0 + 2.0 - 0.0 - 4.0) * (0.0 + 2.0 - 0.0 - 4.0)) / 4.0;
    const double angle_gap = std::atan(2.0 / 2.0) - std::atan(4.0 / 2.0);
    const double aspect = 4.0 / (pi * pi) * angle_gap * angle_gap;
    const double trade_off = aspect / (1.0 - iou + aspect);
    const double expected = iou - (centre_distance / hull_diagonal + aspect * trade_off);

    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(wide, square), iou,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(wide, square), expected, K_IOU_TOLERANCE);
}

/** @brief Tests that a zero-size box gives finite values: IoU 0 and complete IoU -0.05. */
TEST(Unit_DetectBoxes, ADegenerateBoxStaysFinite)
{
    const std::vector<float> point{1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> box{0.0f, 0.0f, 2.0f, 2.0f};

    // The point's width over its height is 0 / eps = 0, so the angle gap is atan(1) = pi / 4.
    const double pi = std::acos(-1.0);
    const double aspect = 4.0 / (pi * pi) * (pi / 4.0) * (pi / 4.0);
    const double trade_off = aspect / (1.0 - 0.0 + aspect);
    const float iou = SushiAI::Vision::Detect::intersection_over_union(point, box);
    const float complete = SushiAI::Vision::Detect::complete_iou(point, box);

    EXPECT_NEAR(iou, 0.0f, K_IOU_TOLERANCE);
    EXPECT_NEAR(complete, 0.0 - aspect * trade_off, K_IOU_TOLERANCE);
    EXPECT_TRUE(std::isfinite(SushiAI::Vision::Detect::complete_iou(point, point)));
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(point, point), 0.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that the elementwise forms pair row i with row i. */
TEST(Unit_DetectBoxes, ElementwiseFormsPairRows)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> second{1.0f, 1.0f, 3.0f, 3.0f, 2.0f, 0.0f, 3.0f, 1.0f};
    std::vector<float> ious(2);
    std::vector<float> completes(2);
    SushiAI::Vision::Detect::elementwise_iou(first, second, ious);
    SushiAI::Vision::Detect::elementwise_complete_iou(first, second, completes);

    const std::vector<float> expected_ious{1.0f / 7.0f, 0.0f};
    const std::vector<float> expected_completes{1.0f / 7.0f - 2.0f / 18.0f, -4.0f / 10.0f};
    EXPECT_TRUE(near_reference(ious, expected_ious, K_IOU_TOLERANCE));
    EXPECT_TRUE(near_reference(completes, expected_completes, K_IOU_TOLERANCE));
}

/** @brief Tests that the pairwise form fills row n, column m with IoU of box n and box m. */
TEST(Unit_DetectBoxes, PairwiseFormFillsEveryPair)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> second{0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f,
                                    5.0f, 5.0f, 6.0f, 6.0f};
    std::vector<float> ious(6);
    SushiAI::Vision::Detect::pairwise_iou(first, second, ious);

    const std::vector<float> expected{4.0f / (4.0f + 4.0f - 4.0f), 1.0f / (4.0f + 4.0f - 1.0f),
                                      0.0f, 1.0f / (1.0f + 4.0f - 1.0f), 0.0f, 0.0f};
    EXPECT_TRUE(near_reference(ious, expected, K_IOU_TOLERANCE));
}

/** @brief Tests that an elementwise IoU over arrays of different length is refused. */
TEST(Unit_DetectBoxes, RejectsElementwiseInputsOfDifferentLength)
{
    const std::vector<float> first(8, 0.0f);
    const std::vector<float> second(4, 0.0f);
    std::vector<float> ious(2);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::elementwise_iou(first, second, ious),
                    "got 8, 4 and 2");
}

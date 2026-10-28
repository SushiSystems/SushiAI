/**************************************************************************/
/* test_target_leaves.cpp                                                 */
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
 * @file test_target_leaves.cpp
 * @brief Tests the leaf constants, the extents of every leaf and the refused geometry.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/free/assignment.hpp>
#include <SushiAI/vision/detect/free/target_leaves.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Free::Assignment;
    using SushiAI::Vision::Detect::Free::build_target_leaves;
    using SushiAI::Vision::Detect::Free::K_MIN_TARGET_SCORES_SUM;
    using SushiAI::Vision::Detect::Free::K_SAFE_BOX;
    using SushiAI::Vision::Detect::Free::TargetLeaves;

    /** @brief Holds the tolerance of a constant or of a value the leaves copy. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Returns an assignment of the given extents in which no anchor is assigned. */
    Assignment make_idle_assignment(std::int64_t batch, std::int64_t anchors,
                                    std::int64_t classes)
    {
        const std::size_t rows = static_cast<std::size_t>(batch * anchors);
        Assignment assignment;
        assignment.batch = batch;
        assignment.anchors = anchors;
        assignment.num_classes = classes;
        assignment.foreground_mask.assign(rows, 0);
        assignment.target_boxes.assign(rows * 4, 0.0f);
        assignment.target_scores.assign(rows * static_cast<std::size_t>(classes), 0.0f);
        assignment.target_labels.assign(rows, 0);
        assignment.target_indices.assign(rows, 0);
        return assignment;
    }

    /** @brief Returns @p anchors points at the centre of one row of cells, x then y. */
    std::vector<float> make_points(std::size_t anchors)
    {
        std::vector<float> points;
        for (std::size_t anchor = 0; anchor < anchors; ++anchor)
        {
            points.push_back(static_cast<float>(anchor) + 0.5f);
            points.push_back(0.5f);
        }
        return points;
    }
} // namespace

/** @brief Tests the safe box, the floor of the score sum and the sum a default leaf set holds. */
TEST(Small_DetectTargetLeaves, TheSafeBoxIsOneCellAndTheScoreSumStartsAtItsFloor)
{
    const TargetLeaves leaves;

    EXPECT_NEAR(K_SAFE_BOX[0], 0.0f, K_EXACT);
    EXPECT_NEAR(K_SAFE_BOX[1], 0.0f, K_EXACT);
    EXPECT_NEAR(K_SAFE_BOX[2], 1.0f, K_EXACT);
    EXPECT_NEAR(K_SAFE_BOX[3], 1.0f, K_EXACT);
    EXPECT_NEAR(K_MIN_TARGET_SCORES_SUM, 1.0f, K_EXACT);
    EXPECT_NEAR(leaves.target_scores_sum[0], K_MIN_TARGET_SCORES_SUM, K_EXACT);
}

/** @brief Tests that every leaf has the row count of the assignment it was built from. */
TEST(Small_DetectTargetLeaves, EveryLeafHasTheRowsOfTheAssignment)
{
    const Assignment assignment = make_idle_assignment(2, 3, 2);
    const std::vector<float> points = make_points(3);
    const std::vector<float> strides(3, 8.0f);
    TargetLeaves leaves;

    SA_EXPECT_OK(build_target_leaves(assignment, {points, strides, 32.0f, 32.0f}, leaves));

    // Two images of three anchors: six rows, four values per box row.
    EXPECT_EQ(leaves.box_mask.size(), 24u);
    EXPECT_EQ(leaves.weight.size(), 6u);
    EXPECT_EQ(leaves.normalized_weight.size(), 6u);
    EXPECT_EQ(leaves.target_boxes.size(), 24u);
    EXPECT_EQ(leaves.target_distances.size(), 24u);
    EXPECT_EQ(leaves.l1_scale.size(), 24u);
    EXPECT_EQ(leaves.safe_boxes.size(), 24u);
}

/** @brief Tests that rebuilding used leaves from a smaller assignment keeps none of the first. */
TEST(Small_DetectTargetLeaves, RebuildingFromASmallerAssignmentKeepsNoEarlierRow)
{
    Assignment first = make_idle_assignment(2, 3, 2);
    first.foreground_mask[4] = 1;
    first.target_scores[4 * 2] = 1.0f;
    first.target_boxes = std::vector<float>(24, 8.0f);
    const std::vector<float> points = make_points(3);
    const std::vector<float> strides(3, 8.0f);
    TargetLeaves leaves;
    SA_EXPECT_OK(build_target_leaves(first, {points, strides, 32.0f, 32.0f}, leaves));

    const Assignment second = make_idle_assignment(1, 3, 2);
    SA_EXPECT_OK(build_target_leaves(second, {points, strides, 32.0f, 32.0f}, leaves));

    ASSERT_EQ(leaves.box_mask.size(), 12u);
    for (std::size_t i = 0; i < leaves.box_mask.size(); ++i)
        EXPECT_NEAR(leaves.box_mask[i], 0.0f, K_EXACT) << "value " << i;
    EXPECT_NEAR(leaves.target_scores_sum[0], K_MIN_TARGET_SCORES_SUM, K_EXACT);
}

/** @brief Tests that geometry of another anchor count and an image without size are refused. */
TEST(Small_DetectTargetLeaves, RefusesGeometryOfAnotherAnchorCountAndNoImageSize)
{
    const Assignment assignment = make_idle_assignment(1, 3, 2);
    const std::vector<float> points = make_points(3);
    const std::vector<float> strides(3, 8.0f);
    const std::vector<float> short_strides(2, 8.0f);
    TargetLeaves leaves;

    SA_EXPECT_FAILURE(build_target_leaves(assignment, {points, short_strides, 32.0f, 32.0f},
                                          leaves),
                      StatusCode::InvalidDescription, "[3, 2] anchor points and [3] strides",
                      "got 6 and 2 values");
    SA_EXPECT_FAILURE(build_target_leaves(assignment, {points, strides, 0.0f, 32.0f}, leaves),
                      StatusCode::InvalidDescription, "positive image size");
}

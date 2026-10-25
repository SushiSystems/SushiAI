/**************************************************************************/
/* test_shape_ratio_assignment.cpp                                        */
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
 * @file test_shape_ratio_assignment.cpp
 * @brief Tests the cells, slots and box targets the shape-ratio assignment gives truths.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::GroundTruthBatch;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AssignmentInput;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioAssignment;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioOptions;
    using SushiAI::Vision::Detect::Anchor::TargetAssignment;
    using SushiAI::Vision::Detect::Anchor::validate_shape_ratio_options;

    /** @brief Holds the tolerance of a box target; every value below is exact in float. */
    constexpr float K_TOLERANCE = 1e-6f;

    /** @brief Holds the network input of 64 pixels; the 8 by 8 grid has a stride of 8. */
    constexpr NetworkSize K_NETWORK{64, 64};

    /** @brief Holds one image's head of 8 by 8 cells. */
    constexpr HeadShape K_SHAPE{1, 8, 8};

    /** @brief Returns a layout of one head over anchors 0.25 and 1.0 of the input. */
    AnchorLayout make_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.25f, 0.25f}, {1.0f, 1.0f}};
        layout.head_anchors = {{0, 1}};
        return layout;
    }

    /** @brief Returns a one-image truth of square boxes, each `{centre, side}` in pixels. */
    GroundTruthBatch make_truth(const std::vector<std::vector<float>>& boxes)
    {
        GroundTruthBatch truth;
        truth.batch = 1;
        truth.max_boxes = static_cast<std::int64_t>(boxes.size());
        for (const std::vector<float>& box : boxes)
        {
            const float half = box[1] / 2.0f;
            truth.boxes.insert(truth.boxes.end(),
                               {box[0] - half, box[0] - half, box[0] + half, box[0] + half});
            truth.labels.push_back(1);
            truth.valid.push_back(1);
        }
        return truth;
    }

    /** @brief Runs the default rule over @p truth and returns its assignments. */
    std::vector<TargetAssignment> assign(const GroundTruthBatch& truth)
    {
        const AnchorLayout layout = make_layout();
        AssignmentInput input;
        input.layout = &layout;
        input.head_index = 0;
        input.shape = K_SHAPE;
        input.classes = 3;
        input.truth = &truth;
        input.network = K_NETWORK;

        std::vector<TargetAssignment> result;
        SA_EXPECT_OK(ShapeRatioAssignment(ShapeRatioOptions()).assign(input, result));
        return result;
    }

    /** @brief Checks the place and the box target of @p entry. */
    void expect_entry(const TargetAssignment& entry, std::int64_t slot, std::int64_t row,
                      std::int64_t col, float box_x, float box_y, std::int64_t truth_row)
    {
        EXPECT_EQ(entry.image, 0);
        EXPECT_EQ(entry.slot, slot);
        EXPECT_EQ(entry.row, row);
        EXPECT_EQ(entry.col, col);
        EXPECT_NEAR(entry.box_x, box_x, K_TOLERANCE);
        EXPECT_NEAR(entry.box_y, box_y, K_TOLERANCE);
        EXPECT_EQ(entry.label, 1);
        EXPECT_EQ(entry.truth_row, truth_row);
    }
} // namespace

/** @brief Tests that a centre on a cell's middle takes that cell alone. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACentreOnTheMiddleTakesItsCellAlone)
{
    const std::vector<TargetAssignment> result = assign(make_truth({{36.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 1u);
    expect_entry(result[0], 0, 4, 4, 0.5f, 0.5f, 0);
    EXPECT_NEAR(result[0].box_width, 2.0f, K_TOLERANCE);
    EXPECT_NEAR(result[0].box_height, 2.0f, K_TOLERANCE);
}

/** @brief Tests that a centre near the top-left corner of its cell gains those neighbours. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACentreNearTheTopLeftGainsTheLeftAndTopCells)
{
    const std::vector<TargetAssignment> result = assign(make_truth({{34.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 3u);
    expect_entry(result[0], 0, 3, 4, 0.25f, 1.25f, 0);
    expect_entry(result[1], 0, 4, 3, 1.25f, 0.25f, 0);
    expect_entry(result[2], 0, 4, 4, 0.25f, 0.25f, 0);
}

/** @brief Tests that a centre near the bottom-right gains the right and bottom cells. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACentreNearTheBottomRightGainsThoseCells)
{
    const std::vector<TargetAssignment> result = assign(make_truth({{38.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 3u);
    expect_entry(result[0], 0, 4, 4, 0.75f, 0.75f, 0);
    expect_entry(result[1], 0, 4, 5, -0.25f, 0.75f, 0);
    expect_entry(result[2], 0, 5, 4, 0.75f, -0.25f, 0);
}

/** @brief Tests that a centre in the first cell gains no neighbour beyond the border. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACentreAtTheBorderGainsNoNeighbour)
{
    const std::vector<TargetAssignment> result = assign(make_truth({{2.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 1u);
    expect_entry(result[0], 0, 0, 0, 0.25f, 0.25f, 0);
}

/** @brief Tests that an anchor whose side ratio reaches the limit is not assigned. */
TEST(Small_DetectAnchorShapeRatioAssignment, AnAnchorAtTheRatioLimitIsRejected)
{
    // The truth spans 2 cells; anchor 1 spans 8, a ratio of exactly 4.
    const std::vector<TargetAssignment> result = assign(make_truth({{36.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].slot, 0);
}

/** @brief Tests that two truths on one slot leave the one of higher shape IoU there. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACollisionKeepsTheTruthOfHigherShapeIoU)
{
    // Row 0 spans 3 cells (IoU 4/9 on slot 0, alone on slot 1); row 1 spans 2 (IoU 1).
    const std::vector<TargetAssignment> result =
        assign(make_truth({{36.0f, 24.0f}, {36.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 2u);
    expect_entry(result[0], 0, 4, 4, 0.5f, 0.5f, 1);
    expect_entry(result[1], 1, 4, 4, 0.5f, 0.5f, 0);
}

/** @brief Tests that two equal truths on one slot leave the lower row there. */
TEST(Small_DetectAnchorShapeRatioAssignment, ACollisionOfEqualTruthsKeepsTheLowerRow)
{
    const std::vector<TargetAssignment> result =
        assign(make_truth({{36.0f, 16.0f}, {36.0f, 16.0f}}));

    ASSERT_EQ(result.size(), 1u);
    expect_entry(result[0], 0, 4, 4, 0.5f, 0.5f, 0);
}

/** @brief Tests that bad settings are refused naming their key. */
TEST(Small_DetectAnchorShapeRatioAssignment, RefusesBadSettingsNamingTheKey)
{
    ShapeRatioOptions ratio;
    ratio.ratio_limit = 1.0f;
    SA_EXPECT_FAILURE(validate_shape_ratio_options(ratio), StatusCode::InvalidDescription,
                      "'ratio_limit'");

    ShapeRatioOptions offset;
    offset.neighbour_offset = 0.75f;
    SA_EXPECT_FAILURE(validate_shape_ratio_options(offset), StatusCode::InvalidDescription,
                      "'neighbour_offset'");

    SA_EXPECT_OK(validate_shape_ratio_options(ShapeRatioOptions()));
}

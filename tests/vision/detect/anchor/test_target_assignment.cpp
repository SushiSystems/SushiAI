/**************************************************************************/
/* test_target_assignment.cpp                                             */
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
 * @file test_target_assignment.cpp
 * @brief Tests the assignment structs and the AssignmentRule interface through one rule.
 * @author Mustafa Garip
 */

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
    using SushiAI::Vision::Detect::Anchor::AssignmentRule;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioAssignment;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioOptions;
    using SushiAI::Vision::Detect::Anchor::TargetAssignment;

    /** @brief Holds the tolerance of a box target; every value below is exact in float. */
    constexpr float K_TOLERANCE = 1e-6f;

    /** @brief Holds the network input of 64 pixels; the 8 by 8 grid has a stride of 8. */
    constexpr NetworkSize K_NETWORK{64, 64};

    /** @brief Returns a layout of one head over anchors 0.25 and 1.0 of the input. */
    AnchorLayout make_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.25f, 0.25f}, {1.0f, 1.0f}};
        layout.head_anchors = {{0, 1}};
        return layout;
    }

    /**
     * @brief Returns a truth of two images with one valid box each, in pixel corners.
     *
     * Image 0 holds a 16 pixel box centred at (36, 36) with the label 1 and image 1 a 16 pixel box
     * centred at (20, 12) with the label 2. On the 8 by 8 grid each centre lies in the middle of
     * its cell, so each box takes that cell alone.
     */
    GroundTruthBatch make_two_images()
    {
        GroundTruthBatch truth;
        truth.batch = 2;
        truth.max_boxes = 1;
        truth.boxes = {28.0f, 28.0f, 44.0f, 44.0f, 12.0f, 4.0f, 28.0f, 20.0f};
        truth.labels = {1, 2};
        truth.valid = {1, 1};
        return truth;
    }

    /** @brief Returns the input of the head 0 of @p layout over @p truth on an 8 by 8 grid. */
    AssignmentInput make_input(const AnchorLayout& layout, const GroundTruthBatch& truth)
    {
        AssignmentInput input;
        input.layout = &layout;
        input.head_index = 0;
        input.shape = HeadShape{truth.batch, 8, 8};
        input.classes = 3;
        input.truth = &truth;
        input.network = K_NETWORK;
        return input;
    }
} // namespace

/** @brief Tests that an assignment starts at zero in every field. */
TEST(Small_DetectAnchorTargetAssignment, ADefaultAssignmentIsZero)
{
    const TargetAssignment entry;

    EXPECT_EQ(entry.image, 0);
    EXPECT_EQ(entry.slot, 0);
    EXPECT_EQ(entry.row, 0);
    EXPECT_EQ(entry.col, 0);
    EXPECT_EQ(entry.box_x, 0.0f);
    EXPECT_EQ(entry.box_y, 0.0f);
    EXPECT_EQ(entry.box_width, 0.0f);
    EXPECT_EQ(entry.box_height, 0.0f);
    EXPECT_EQ(entry.label, 0);
    EXPECT_EQ(entry.truth_row, 0);
}

/** @brief Tests that an assignment input starts without a layout, a truth or a class. */
TEST(Small_DetectAnchorTargetAssignment, ADefaultInputNamesNoLayoutAndNoTruth)
{
    const AssignmentInput input;

    EXPECT_EQ(input.layout, nullptr);
    EXPECT_EQ(input.truth, nullptr);
    EXPECT_EQ(input.head_index, 0);
    EXPECT_EQ(input.classes, 0);
    EXPECT_EQ(input.shape.batch, 0);
    EXPECT_EQ(input.network.height, 0);
}

/** @brief Tests that assign through the interface orders the entries by image and place. */
TEST(Small_DetectAnchorTargetAssignment, ARuleOrdersTheEntriesByImageThroughTheInterface)
{
    const AnchorLayout layout = make_layout();
    const GroundTruthBatch truth = make_two_images();
    const ShapeRatioAssignment concrete{ShapeRatioOptions()};
    const AssignmentRule& rule = concrete;

    std::vector<TargetAssignment> result;
    SA_EXPECT_OK(rule.assign(make_input(layout, truth), result));

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].image, 0);
    EXPECT_EQ(result[0].slot, 0);
    EXPECT_EQ(result[0].row, 4);
    EXPECT_EQ(result[0].col, 4);
    EXPECT_EQ(result[0].label, 1);
    EXPECT_EQ(result[0].truth_row, 0);
    EXPECT_NEAR(result[0].box_x, 0.5f, K_TOLERANCE);
    EXPECT_NEAR(result[0].box_y, 0.5f, K_TOLERANCE);
    EXPECT_NEAR(result[0].box_width, 2.0f, K_TOLERANCE);
    EXPECT_NEAR(result[0].box_height, 2.0f, K_TOLERANCE);
    EXPECT_EQ(result[1].image, 1);
    EXPECT_EQ(result[1].slot, 0);
    EXPECT_EQ(result[1].row, 1);
    EXPECT_EQ(result[1].col, 2);
    EXPECT_EQ(result[1].label, 2);
    EXPECT_EQ(result[1].truth_row, 0);
}

/** @brief Tests that the entries a caller left in the result are replaced, not extended. */
TEST(Small_DetectAnchorTargetAssignment, ARuleReplacesWhatTheResultHeld)
{
    const AnchorLayout layout = make_layout();
    const GroundTruthBatch truth = make_two_images();
    const ShapeRatioAssignment concrete{ShapeRatioOptions()};
    const AssignmentRule& rule = concrete;

    std::vector<TargetAssignment> result(5);
    result[0].label = 99;
    SA_EXPECT_OK(rule.assign(make_input(layout, truth), result));

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].label, 1);
}

/** @brief Tests that a truth with no valid row leaves the result empty. */
TEST(Small_DetectAnchorTargetAssignment, ABatchWithoutAValidRowAssignsNothing)
{
    const AnchorLayout layout = make_layout();
    GroundTruthBatch truth = make_two_images();
    truth.valid = {0, 0};
    const ShapeRatioAssignment concrete{ShapeRatioOptions()};
    const AssignmentRule& rule = concrete;

    std::vector<TargetAssignment> result(3);
    SA_EXPECT_OK(rule.assign(make_input(layout, truth), result));

    EXPECT_TRUE(result.empty());
}

/** @brief Tests that a head the layout does not hold is refused naming both counts. */
TEST(Small_DetectAnchorTargetAssignment, AHeadOutsideTheLayoutIsRefused)
{
    const AnchorLayout layout = make_layout();
    const GroundTruthBatch truth = make_two_images();
    AssignmentInput input = make_input(layout, truth);
    input.head_index = 1;
    const ShapeRatioAssignment concrete{ShapeRatioOptions()};
    const AssignmentRule& rule = concrete;

    std::vector<TargetAssignment> result;
    SA_EXPECT_FAILURE(rule.assign(input, result), StatusCode::InvalidDescription, "head 1",
                      "layout's 1");
}

/** @brief Tests that a grid without a cell is refused. */
TEST(Small_DetectAnchorTargetAssignment, AGridWithoutACellIsRefused)
{
    const AnchorLayout layout = make_layout();
    const GroundTruthBatch truth = make_two_images();
    AssignmentInput input = make_input(layout, truth);
    input.shape.height = 0;
    const ShapeRatioAssignment concrete{ShapeRatioOptions()};
    const AssignmentRule& rule = concrete;

    std::vector<TargetAssignment> result;
    SA_EXPECT_FAILURE(rule.assign(input, result), StatusCode::InvalidDescription,
                      "holds no cell");
}

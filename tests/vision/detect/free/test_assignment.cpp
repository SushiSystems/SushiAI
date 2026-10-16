/**************************************************************************/
/* test_assignment.cpp                                                    */
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
 * @file test_assignment.cpp
 * @brief Tests that a default assignment has no extents and holds no targets.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/detect/free/assignment.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::Assignment;
} // namespace

/** @brief Tests that a default assignment has zero extents and five empty target arrays. */
TEST(Small_DetectAssignment, ADefaultAssignmentHasNoExtentsAndNoTargets)
{
    const Assignment assignment;

    EXPECT_EQ(assignment.batch, 0);
    EXPECT_EQ(assignment.anchors, 0);
    EXPECT_EQ(assignment.num_classes, 0);
    EXPECT_TRUE(assignment.foreground_mask.empty());
    EXPECT_TRUE(assignment.target_boxes.empty());
    EXPECT_TRUE(assignment.target_scores.empty());
    EXPECT_TRUE(assignment.target_labels.empty());
    EXPECT_TRUE(assignment.target_indices.empty());
}

/**************************************************************************/
/* test_detection_row.cpp                                                 */
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
 * @file test_detection_row.cpp
 * @brief Tests the columns of a detection row and the reference's limits on an image's rows.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/detect/detection_row.hpp>

namespace
{
    using SushiAI::Vision::Detect::K_DETECTION_CLASS;
    using SushiAI::Vision::Detect::K_DETECTION_SCORE;
    using SushiAI::Vision::Detect::K_DETECTION_VALUES;
    using SushiAI::Vision::Detect::K_MAX_DETECTIONS;
    using SushiAI::Vision::Detect::K_VALIDATION_CONFIDENCE;

    /** @brief Holds the tolerance of a constant compared with the literal it is defined by. */
    constexpr float K_EXACT = 0.0f;
} // namespace

/** @brief Tests that a row holds four corners, then the score, then the class. */
TEST(Small_DetectDetectionRow, ARowHoldsFourCornersThenTheScoreThenTheClass)
{
    EXPECT_EQ(K_DETECTION_VALUES, 6u);
    EXPECT_EQ(K_DETECTION_SCORE, 4u);
    EXPECT_EQ(K_DETECTION_CLASS, 5u);
    EXPECT_LT(K_DETECTION_CLASS, K_DETECTION_VALUES);
}

/** @brief Tests that the limits are the reference's 300 rows and its 0.001 score floor. */
TEST(Small_DetectDetectionRow, TheLimitsAreTheReferenceValues)
{
    EXPECT_EQ(K_MAX_DETECTIONS, 300);
    EXPECT_NEAR(K_VALIDATION_CONFIDENCE, 0.001f, K_EXACT);
}

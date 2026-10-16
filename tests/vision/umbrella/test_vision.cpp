/**************************************************************************/
/* test_vision.cpp                                                        */
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
 * @file test_vision.cpp
 * @brief Tests that the vision umbrella header alone reaches the units of several modules.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision.hpp>

namespace
{
    /** @brief Holds the absolute tolerance of an IoU whose terms float32 rounds once each. */
    constexpr float K_IOU_TOLERANCE = 1e-6f;
} // namespace

/** @brief Tests that detect, segment and imaging names resolve through the umbrella alone. */
TEST(Small_VisionUmbrella, OneIncludeReachesDetectSegmentAndImaging)
{
    const SushiAI::Vision::Detect::Anchor::AnchorSize size{0.25f, 0.5f};

    EXPECT_NEAR(SushiAI::Vision::Detect::Anchor::shape_iou(size, size), 1.0f, K_IOU_TOLERANCE);
    EXPECT_EQ(SushiAI::Vision::Detect::K_DETECTION_VALUES, 6u);
    EXPECT_EQ(SushiAI::Vision::Segment::K_DEFAULT_IGNORE_INDEX, 255);
    EXPECT_EQ(SushiAI::Vision::Imaging::K_MOSAIC_PARTNERS, 3u);
}

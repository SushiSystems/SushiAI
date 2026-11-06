/**************************************************************************/
/* test_umbrella_reach.cpp                                                */
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
 * @file test_umbrella_reach.cpp
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

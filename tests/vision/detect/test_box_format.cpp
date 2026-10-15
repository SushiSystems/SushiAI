/**************************************************************************/
/* test_box_format.cpp                                                    */
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
 * @file test_box_format.cpp
 * @brief Tests the conversions between box formats.
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

    /** @brief Holds the tolerance of hand cases whose values float32 represents exactly. */
    constexpr float K_EXACT = 0.0f;

} // namespace

/** @brief Tests that corners convert to centre and size and back to the same corners. */
TEST(Medium_DetectBoxes, ConvertsBetweenCornersAndCentreSize)
{
    const std::vector<float> corners{1.0f, 2.0f, 5.0f, 8.0f};
    std::vector<float> centred(4);
    std::vector<float> back(4);
    SushiAI::Vision::Detect::xyxy_to_xywh(corners, centred);
    SushiAI::Vision::Detect::xywh_to_xyxy(centred, back);

    const std::vector<float> expected{(1.0f + 5.0f) / 2.0f, (2.0f + 8.0f) / 2.0f, 5.0f - 1.0f,
                                      8.0f - 2.0f};
    EXPECT_TRUE(near_reference(centred, expected, K_EXACT));
    EXPECT_TRUE(near_reference(back, corners, K_EXACT));
}

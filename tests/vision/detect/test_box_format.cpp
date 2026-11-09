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
 * @file test_box_format.cpp
 * @brief Tests the conversions between box formats.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace
{
    using SushiAI::Test::equal_reference;

} // namespace

/**
 * @brief Tests that corners convert to centre and size and back to the same corners.
 *
 * The corners (1, 2, 5, 8) have the centre (3, 5) and the size 4 by 6; every value is exact in
 * float32, so the results are compared for equality.
 */
TEST(Small_DetectBoxes, ConvertsBetweenCornersAndCentreSize)
{
    const std::vector<float> corners{1.0f, 2.0f, 5.0f, 8.0f};
    std::vector<float> centred(4);
    std::vector<float> back(4);
    SushiAI::Vision::Detect::xyxy_to_xywh(corners, centred);
    SushiAI::Vision::Detect::xywh_to_xyxy(centred, back);

    EXPECT_TRUE(equal_reference(centred, std::vector<float>{3.0f, 5.0f, 4.0f, 6.0f}));
    EXPECT_TRUE(equal_reference(back, corners));
}

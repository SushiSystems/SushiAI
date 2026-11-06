/**************************************************************************/
/* test_ignore_index.cpp                                                  */
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
 * @file test_ignore_index.cpp
 * @brief Tests that the default ignore index is 255, the largest value a mask byte holds.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/vision/segment/ignore_index.hpp>

/** @brief Tests that the default ignore index is 255, which an 8-bit mask pixel can hold. */
TEST(Small_SegmentIgnoreIndex, TheDefaultIsTheLargestValueOfAMaskByte)
{
    EXPECT_EQ(SushiAI::Vision::Segment::K_DEFAULT_IGNORE_INDEX, 255);
    EXPECT_EQ(SushiAI::Vision::Segment::K_DEFAULT_IGNORE_INDEX,
              std::numeric_limits<std::uint8_t>::max());
}

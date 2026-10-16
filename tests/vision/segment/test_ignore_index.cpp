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

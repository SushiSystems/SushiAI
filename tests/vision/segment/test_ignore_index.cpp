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
 * @brief Tests that the default ignore index is the largest value a mask byte holds and that a
 *        confusion matrix leaves that class uncounted.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/ignore_index.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

/**
 * @brief Tests that a pixel whose mask byte is the largest a byte holds is not counted when add
 *        is given the default ignore index, so the index fits a mask byte and ignores that class.
 */
TEST(Small_SegmentIgnoreIndex, TheDefaultIsTheLargestValueOfAMaskByte)
{
    SA_ASSERT_OK_AND_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(2));
    const std::vector<std::int32_t> predicted{0, 1};
    const std::vector<std::int32_t> truth{
        0, static_cast<std::int32_t>(std::numeric_limits<std::uint8_t>::max())};

    SA_EXPECT_OK(matrix.add(
        SushiRuntime::span<const std::int32_t>(predicted.data(), predicted.size()),
        SushiRuntime::span<const std::int32_t>(truth.data(), truth.size()),
        Vision::Segment::K_DEFAULT_IGNORE_INDEX));

    EXPECT_EQ(matrix.pixel_count(), 1);
    EXPECT_EQ(matrix.count(0, 0), 1);
    EXPECT_EQ(matrix.predicted_pixels(1), 0);
}

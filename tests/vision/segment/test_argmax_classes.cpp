/**************************************************************************/
/* test_argmax_classes.cpp                                                */
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
 * @file test_argmax_classes.cpp
 * @brief Tests the class pick of each pixel.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

} // namespace

/** @brief Tests that the pick is the class of the largest score of each pixel. */
TEST(Small_SegmentMetrics, ArgmaxPicksTheLargestScore)
{
    const std::vector<float> scores{0.1f, 0.9f, 0.0f, -3.0f, -2.0f, -1.0f};
    Classes picked(2, -1);

    Vision::Segment::argmax_classes(
        SushiRuntime::span<const float>(scores.data(), scores.size()), 3,
        SushiRuntime::span<std::int32_t>(picked.data(), picked.size()));

    EXPECT_EQ(picked, (Classes{1, 2}));
}

/** @brief Tests that equal scores give the lowest class. */
TEST(Small_SegmentMetrics, ArgmaxGivesTheLowestClassOnATie)
{
    const std::vector<float> scores{2.0f, -1.0f, 2.0f};
    Classes picked(1, -1);

    Vision::Segment::argmax_classes(
        SushiRuntime::span<const float>(scores.data(), scores.size()), 3,
        SushiRuntime::span<std::int32_t>(picked.data(), picked.size()));

    EXPECT_EQ(picked, (Classes{0}));
}

/**************************************************************************/
/* test_pixel_accuracy.cpp                                                */
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
 * @file test_pixel_accuracy.cpp
 * @brief Tests pixel accuracy.
 * @author Mustafa Garip
 */

#include <cstddef>
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
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Holds the index the tests mark a pixel of no class with. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Holds the rounded operations of one ratio of two whole counts: one division. */
    constexpr std::size_t K_RATIO_OPERATIONS = 1;

    /** @brief Returns a matrix of @p classes classes that counted the two lists. */
    Result<ConfusionMatrix> counted(std::int32_t classes, const Classes& predicted,
                                    const Classes& truth)
    {
        SA_TRY_ASSIGN(ConfusionMatrix matrix, ConfusionMatrix::make(classes));
        SA_TRY(matrix.add(
            SushiRuntime::span<const std::int32_t>(predicted.data(), predicted.size()),
            SushiRuntime::span<const std::int32_t>(truth.data(), truth.size()), K_IGNORE));

        return Result<ConfusionMatrix>(std::move(matrix));
    }

} // namespace

/** @brief Tests that pixel accuracy is the diagonal over the counted pixels. */
TEST(Small_SegmentMetrics, PixelAccuracyIsTheDiagonalOverTheCountedPixels)
{
    SA_ASSERT_OK_AND_ASSIGN(const ConfusionMatrix matrix,
                            counted(3, {0, 1, 1, 2, 2, 2, 0}, {0, 0, 1, 2, 1, 2, K_IGNORE}));

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        Vision::Segment::pixel_accuracy(matrix), 4.0 / 6.0, K_RATIO_OPERATIONS)));
}

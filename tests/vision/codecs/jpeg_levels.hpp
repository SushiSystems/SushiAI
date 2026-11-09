/**************************************************************************/
/* jpeg_levels.hpp                                                        */
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
 * @file jpeg_levels.hpp
 * @brief Declares the tolerance a level decoded from a JPEG may have and the checks that use it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <support/tolerance.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Test
{
    /** @brief Holds one level, the step of an 8-bit sample. */
    constexpr double K_LEVEL = 1.0;

    /** @brief Counts the stages of a JPEG round trip that each round to a whole level. */
    constexpr double K_JPEG_ROUNDINGS = 3.0;

    /**
     * @brief Holds the absolute difference, in levels, a decoded level may have from the encoded one.
     *
     * The tolerance is absolute and counts codec levels, not floating-point rounding. At quality 100
     * every quantiser step is one level, and three stages each round to a whole level: the forward
     * transform, the quantiser and the inverse transform with its colour conversion.
     */
    constexpr double K_JPEG_TOLERANCE = K_JPEG_ROUNDINGS * K_LEVEL;

    /** @brief Returns whether a decoded level is within K_JPEG_TOLERANCE of the encoded one. */
    inline ::testing::AssertionResult level_within_jpeg_tolerance(int decoded, int expected)
    {
        return TestSupport::is_within(
            TestSupport::measure_error(decoded, expected, K_JPEG_TOLERANCE, 1.0));
    }

    /** @brief Returns whether every byte of an image is within K_JPEG_TOLERANCE of one level. */
    inline ::testing::AssertionResult every_level_within_jpeg_tolerance(
        const Vision::Imaging::Image& image, int expected)
    {
        for (std::size_t i = 0; i < image.byte_count(); ++i)
        {
            const ::testing::AssertionResult level =
                level_within_jpeg_tolerance(image.data()[i], expected);
            if (!level)
                return ::testing::AssertionFailure()
                       << "byte " << i << " is " << static_cast<int>(image.data()[i])
                       << ", expected " << expected << ": " << level.message();
        }

        return ::testing::AssertionSuccess();
    }
} // namespace SushiAI::Test

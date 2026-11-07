/**************************************************************************/
/* imaging_images.hpp                                                     */
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
 * @file imaging_images.hpp
 * @brief Declares the image builders and image expectations the imaging tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Test
{
    /** @brief Returns a three-channel image with every pixel the same colour. */
    inline Vision::Imaging::Image make_constant(Vision::Imaging::ImageSize size, std::uint8_t red,
                                                std::uint8_t green, std::uint8_t blue)
    {
        Vision::Imaging::Image image(size, 3);
        for (std::int32_t y = 0; y < size.height; ++y)
        {
            std::uint8_t* row = image.row(y);
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                row[3 * x] = red;
                row[3 * x + 1] = green;
                row[3 * x + 2] = blue;
            }
        }

        return image;
    }

    /** @brief Returns an image of the given channel count with every byte at one level. */
    inline Vision::Imaging::Image make_flat_image(Vision::Imaging::ImageSize size,
                                                  std::int32_t channels, std::uint8_t level)
    {
        Vision::Imaging::Image image(size, channels);
        image.fill(level);
        return image;
    }

    /** @brief Checks that two images have the same size, channels and bytes. */
    inline void expect_same_image(const Vision::Imaging::Image& actual,
                                  const Vision::Imaging::Image& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_EQ(actual.channels(), expected.channels());
        for (std::size_t i = 0; i < expected.byte_count(); ++i)
            ASSERT_EQ(actual.data()[i], expected.data()[i]) << "byte " << i;
    }

    /** @brief Checks that an image holds exactly the listed bytes, in order. */
    inline void expect_bytes(const Vision::Imaging::Image& image,
                             std::initializer_list<std::uint8_t> expected)
    {
        ASSERT_EQ(image.byte_count(), expected.size());
        std::size_t offset = 0;
        for (const std::uint8_t value : expected)
        {
            EXPECT_EQ(image.data()[offset], value) << "byte " << offset;
            ++offset;
        }
    }
} // namespace SushiAI::Test

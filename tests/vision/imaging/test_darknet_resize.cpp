/**************************************************************************/
/* test_darknet_resize.cpp                                                */
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
 * @file test_darknet_resize.cpp
 * @brief Tests the bilinear resize of Darknet's resize_image against hand-derived images.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/darknet_resize.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/resize.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;

namespace
{
    using SushiAI::Test::expect_bytes;

    /** @brief Returns a one-channel image holding the listed levels, row by row. */
    Image make_gray(ImageSize size, std::initializer_list<std::uint8_t> levels)
    {
        Image image(size, 1);
        std::size_t offset = 0;
        for (const std::uint8_t level : levels)
            image.data()[offset++] = level;

        return image;
    }
} // namespace

/** @brief Tests that the first and last columns of the result are those of the source. */
TEST(Small_ImagingDarknetResize, CornersMapToCorners)
{
    // Five columns read the positions 0, 0.25 ... 1 of the two source columns, 0 and 100.
    const Image source = make_gray(ImageSize{1, 2}, {0, 100});

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::resize_darknet(source.view(), ImageSize{1, 5}, destination));

    expect_bytes(destination, {0, 25, 50, 75, 100});
}

/** @brief Tests that the kernel differs from the pixel-centre kernel of resize_bilinear. */
TEST(Small_ImagingDarknetResize, ItIsNotTheKernelOfResizeBilinear)
{
    const Image source = make_gray(ImageSize{1, 2}, {0, 100});

    Image centred;
    SA_EXPECT_OK(Vision::Imaging::resize_bilinear(source.view(), ImageSize{1, 5}, centred));

    // Centres read (x + 0.5) * 0.4 - 0.5 = -0.3, 0.1, 0.5, 0.9 and 1.3, clamped to [0, 1].
    expect_bytes(centred, {0, 10, 50, 90, 100});
}

/** @brief Tests a 2 by 2 image enlarged to 3 by 3: columns first, then rows, each at the middle. */
TEST(Small_ImagingDarknetResize, AnEnlargementBlendsColumnsThenRows)
{
    // Columns read 0, 0.5, 1: 0, 50, 100 and 200, 120, 40; the middle row is their mean.
    const Image source = make_gray(ImageSize{2, 2}, {0, 100, 200, 40});

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::resize_darknet(source.view(), ImageSize{3, 3}, destination));

    expect_bytes(destination, {0, 50, 100, 100, 85, 70, 200, 120, 40});
}

/** @brief Tests that each channel is blended on its own. */
TEST(Small_ImagingDarknetResize, EachChannelIsBlendedOnItsOwn)
{
    Image source(ImageSize{1, 2}, 3);
    const std::uint8_t levels[] = {0, 10, 20, 100, 110, 120};
    for (std::size_t i = 0; i < 6; ++i)
        source.data()[i] = levels[i];

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::resize_darknet(source.view(), ImageSize{1, 3}, destination));

    EXPECT_EQ(destination.channels(), 3);
    expect_bytes(destination, {0, 10, 20, 50, 60, 70, 100, 110, 120});
}

/** @brief Tests that a size equal to the source's gives the source back. */
TEST(Small_ImagingDarknetResize, TheSameSizeIsACopy)
{
    const Image source = make_gray(ImageSize{2, 3}, {7, 8, 9, 200, 201, 255});

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::resize_darknet(source.view(), source.size(), destination));

    expect_bytes(destination, {7, 8, 9, 200, 201, 255});
}

/** @brief Tests that a destination side of one reads the first source position. */
TEST(Small_ImagingDarknetResize, ASideOfOneReadsTheFirstPosition)
{
    const Image source = make_gray(ImageSize{2, 3}, {10, 20, 30, 40, 50, 60});

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::resize_darknet(source.view(), ImageSize{1, 1}, destination));

    expect_bytes(destination, {10});
}

/** @brief Tests that a size outside the bounds of an image is refused. */
TEST(Small_ImagingDarknetResize, ASizeOutsideTheBoundsIsRefused)
{
    const Image source = make_gray(ImageSize{1, 1}, {1});

    Image destination;
    SA_EXPECT_FAILURE(Vision::Imaging::resize_darknet(source.view(), ImageSize{0, 3}, destination),
                      StatusCode::InvalidDescription, "resize_darknet", "0x3");
}

/**************************************************************************/
/* test_gray.cpp                                                          */
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
 * @file test_gray.cpp
 * @brief Tests the expansion of a one-channel image to three equal channels.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::expect_bytes;
    /** @brief Returns an image whose byte at offset i is 17 * i + 3 modulo 256. */
    Image make_counted(ImageSize size, std::int32_t channels)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(17 * i + 3);

        return image;
    }
} // namespace

/** @brief Tests that every pixel's three channels take its grey level, at the same size. */
TEST(Small_ImagingGray, ExpandCopiesEachLevelIntoThreeChannels)
{
    const Image gray = make_counted(ImageSize{2, 3}, 1);
    Image color;

    SA_EXPECT_OK(expand_gray(gray.view(), color));

    ASSERT_EQ(color.size(), (ImageSize{2, 3}));
    ASSERT_EQ(color.channels(), K_RGB_CHANNELS);
    expect_bytes(color, {3, 3, 3, 20, 20, 20, 37, 37, 37, 54, 54, 54, 71, 71, 71, 88, 88, 88});
}

/** @brief Tests that a reused destination is reset and wholly rewritten. */
TEST(Small_ImagingGray, ExpandRewritesAReusedDestination)
{
    const Image gray = make_counted(ImageSize{2, 2}, 1);
    Image color(ImageSize{9, 9}, 4);
    color.fill(255);

    SA_EXPECT_OK(expand_gray(gray.view(), color));

    ASSERT_EQ(color.size(), (ImageSize{2, 2}));
    ASSERT_EQ(color.channels(), 3);
    expect_bytes(color, {3, 3, 3, 20, 20, 20, 37, 37, 37, 54, 54, 54});
}

/** @brief Tests that a source with a channel count other than one is refused. */
TEST(Small_ImagingGray, ExpandRejectsASourceThatIsNotGrey)
{
    const Image color = make_counted(ImageSize{2, 2}, 3);
    const Image four = make_counted(ImageSize{2, 2}, 4);
    Image destination;

    SA_EXPECT_FAILURE(expand_gray(color.view(), destination), StatusCode::InvalidDescription,
                      "expand_gray", "3 channels");
    SA_EXPECT_FAILURE(expand_gray(four.view(), destination), StatusCode::InvalidDescription,
                      "expand_gray", "4 channels");
}

/** @brief Tests that a source without pixels stops the process. */
TEST(Medium_ImagingGray, ExpandStopsOnASourceWithoutPixels)
{
    Image image = make_counted(ImageSize{4, 2}, 1);

    SA_EXPECT_FATAL(static_cast<void>(expand_gray(ImageView(), image)), "expand_gray", "no pixels");
}

/** @brief Tests that a source held by the destination, whole or as rows, stops the process. */
TEST(Medium_ImagingGray, ExpandStopsOnASourceInTheDestination)
{
    Image image = make_counted(ImageSize{4, 2}, 1);
    const ImageView lower_rows(image.row(1), ImageSize{2, 2}, 1);

    SA_EXPECT_FATAL(static_cast<void>(expand_gray(image.view(), image)), "expand_gray",
                    "destination holds the source's pixels");
    SA_EXPECT_FATAL(static_cast<void>(expand_gray(lower_rows, image)), "expand_gray",
                    "destination holds the source's pixels");
}

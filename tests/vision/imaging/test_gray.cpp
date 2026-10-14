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
 * @file test_gray.cpp
 * @brief Tests the expansion of a one-channel image to three equal channels.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/gray.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
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
TEST(Unit_ImagingGray, ExpandCopiesEachLevelIntoThreeChannels)
{
    const Image gray = make_counted(ImageSize{3, 5}, 1);
    Image color;

    expand_gray(gray.view(), color);

    ASSERT_EQ(color.size(), (ImageSize{3, 5}));
    ASSERT_EQ(color.channels(), K_RGB_CHANNELS);
    for (std::int32_t y = 0; y < 3; ++y)
    {
        for (std::int32_t x = 0; x < 5; ++x)
        {
            for (std::int32_t c = 0; c < 3; ++c)
                EXPECT_EQ(color.row(y)[3 * x + c], gray.row(y)[x])
                    << "row " << y << ", column " << x << ", channel " << c;
        }
    }
}

/** @brief Tests that a reused destination is reset and wholly rewritten. */
TEST(Unit_ImagingGray, ExpandRewritesAReusedDestination)
{
    const Image gray = make_counted(ImageSize{2, 2}, 1);
    Image color(ImageSize{9, 9}, 4);
    color.fill(255);

    expand_gray(gray.view(), color);

    ASSERT_EQ(color.size(), (ImageSize{2, 2}));
    ASSERT_EQ(color.channels(), 3);
    for (std::size_t i = 0; i < color.byte_count(); ++i)
        EXPECT_EQ(color.data()[i], gray.data()[i / 3]) << "byte " << i;
}

/** @brief Tests that a source with a channel count other than one is refused. */
TEST(Unit_ImagingGray, ExpandRejectsASourceThatIsNotGrey)
{
    const Image color = make_counted(ImageSize{2, 2}, 3);
    const Image four = make_counted(ImageSize{2, 2}, 4);
    Image destination;

    SA_EXPECT_ERROR(expand_gray(color.view(), destination), "expand_gray", "3 channels");
    SA_EXPECT_ERROR(expand_gray(four.view(), destination), "expand_gray", "4 channels");
    EXPECT_EQ(K_GRAY_CHANNELS, 1);
}

/** @brief Tests that an empty source and a source inside the destination are refused. */
TEST(Unit_ImagingGray, ExpandRejectsASourceWithoutPixelsAndOneInTheDestination)
{
    Image image = make_counted(ImageSize{4, 2}, 1);
    const Image untouched = make_counted(ImageSize{4, 2}, 1);
    const ImageView lower_rows(image.row(1), ImageSize{2, 2}, 1);

    SA_EXPECT_ERROR(expand_gray(ImageView(), image), "expand_gray", "no pixels");
    SA_EXPECT_ERROR(expand_gray(image.view(), image), "expand_gray",
                    "destination holds the source's pixels");
    SA_EXPECT_ERROR(expand_gray(lower_rows, image), "expand_gray",
                    "destination holds the source's pixels");

    ASSERT_EQ(image.size(), (ImageSize{4, 2}));
    ASSERT_EQ(image.channels(), 1);
    for (std::size_t i = 0; i < image.byte_count(); ++i)
        EXPECT_EQ(image.data()[i], untouched.data()[i]) << "byte " << i;
}

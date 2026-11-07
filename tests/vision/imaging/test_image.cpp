/**************************************************************************/
/* test_image.cpp                                                         */
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
 * @file test_image.cpp
 * @brief Tests the 8-bit image's construction, bounds, storage reuse, rows and views.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;
using SushiAI::Vision::Imaging::ImageView;
using SushiAI::Vision::Imaging::K_MAX_CHANNELS;
using SushiAI::Vision::Imaging::K_MAX_IMAGE_SIDE;
using SushiAI::Vision::Imaging::check_source;
using SushiAI::Vision::Imaging::is_valid_channel_count;
using SushiAI::Vision::Imaging::is_valid_side;
using SushiAI::Vision::Imaging::is_valid_size;
using SushiAI::Vision::Imaging::overlaps;

namespace
{
    using SushiAI::Test::require_value;
    /**
     * @brief Writes the byte's own offset, modulo 251, into every byte of an image.
     * @param image Any image; an empty one is left alone.
     */
    void write_offsets(Image& image)
    {
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(i % 251);
    }

    /** @brief Expects every byte of an image to hold @p expected. */
    void expect_every_byte(const Image& image, std::uint8_t expected)
    {
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            ASSERT_EQ(image.data()[i], expected) << "byte " << i;
    }

    /** @brief Expects byte i of an image to hold i modulo 251, as write_offsets leaves it. */
    void expect_offsets(const Image& image)
    {
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            ASSERT_EQ(static_cast<std::size_t>(image.data()[i]), i % 251) << "byte " << i;
    }

    /** @brief Expects an image to hold exactly the bytes of @p expected, in order. */
    void expect_bytes(const Image& image, const std::vector<std::uint8_t>& expected)
    {
        ASSERT_EQ(image.byte_count(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            ASSERT_EQ(image.data()[i], expected[i]) << "byte " << i;
    }

    /** @brief Expects row y of an image, its constant form and its view to start at @p offset. */
    void expect_row_starts_at(Image& image, std::int32_t y, std::size_t offset)
    {
        const Image& constant = image;

        EXPECT_EQ(image.row(y), image.data() + offset) << "row " << y;
        EXPECT_EQ(constant.row(y), constant.data() + offset) << "row " << y;
        EXPECT_EQ(image.view().row(y), image.data() + offset) << "row " << y;
    }
} // namespace

static_assert(ImageSize{3, 4} == ImageSize{3, 4}, "equal sizes compare equal");
static_assert(ImageSize{3, 4} != ImageSize{4, 3}, "height and width are not interchangeable");
static_assert(ImageView().byte_count() == 0, "a default view has no bytes");
static_assert(is_valid_side(1) && is_valid_side(K_MAX_IMAGE_SIDE), "both ends are sides");
static_assert(!is_valid_side(0) && !is_valid_side(K_MAX_IMAGE_SIDE + 1), "one past each end");
static_assert(is_valid_size(ImageSize{1, K_MAX_IMAGE_SIDE}), "a size of two valid sides");
static_assert(!is_valid_size(ImageSize{0, 4}) && !is_valid_size(ImageSize{4, -1}), "one bad side");
static_assert(is_valid_channel_count(1) && is_valid_channel_count(K_MAX_CHANNELS), "both ends");
static_assert(!is_valid_channel_count(0) && !is_valid_channel_count(K_MAX_CHANNELS + 1), "past");
static_assert(ImageView(nullptr, ImageSize{2, 5}, 3).row_bytes() == 15, "width by channels");
static_assert(ImageView(nullptr, ImageSize{2, 5}, 3).byte_count() == 30, "rows by row bytes");

TEST(Small_ImagingImage, ImageSizeComparesHeightAndWidth)
{
    EXPECT_TRUE((ImageSize{3, 4} == ImageSize{3, 4}));
    EXPECT_TRUE((ImageSize{3, 4} != ImageSize{3, 5}));
    EXPECT_TRUE((ImageSize{3, 4} != ImageSize{2, 4}));
    EXPECT_FALSE((ImageSize{3, 4} == ImageSize{4, 3}));
}

TEST(Small_ImagingImage, AConstructedImageReportsItsExtents)
{
    const Image image(ImageSize{4, 6}, 3);

    EXPECT_EQ(image.size(), (ImageSize{4, 6}));
    EXPECT_EQ(image.height(), 4);
    EXPECT_EQ(image.width(), 6);
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(image.row_bytes(), 18u);
    EXPECT_EQ(image.byte_count(), 72u);
    ASSERT_NE(image.data(), nullptr);
}

/** @brief Tests that a constructed image holds zero in every byte. */
TEST(Small_ImagingImage, AConstructedImageIsZero)
{
    const Image image(ImageSize{5, 7}, 4);

    expect_every_byte(image, 0);
}

TEST(Small_ImagingImage, TheLargestSideAndChannelCountAreAccepted)
{
    const Image tall(ImageSize{K_MAX_IMAGE_SIDE, 1}, 1);
    const Image wide(ImageSize{1, K_MAX_IMAGE_SIDE}, K_MAX_CHANNELS);

    EXPECT_EQ(tall.byte_count(), static_cast<std::size_t>(K_MAX_IMAGE_SIDE));
    EXPECT_EQ(wide.byte_count(), static_cast<std::size_t>(K_MAX_IMAGE_SIDE) * K_MAX_CHANNELS);
}

/** @brief Tests that construction stops on a height or a width outside the side bounds. */
TEST(Medium_ImagingImage, ConstructionRejectsASideOutsideItsBounds)
{
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{0, 4}, 3)), "Image:", "size of 0x4", "16384");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{-2, 4}, 3)), "Image:", "size of -2x4");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{K_MAX_IMAGE_SIDE + 1, 4}, 3)), "Image:",
                    "size of 16385x4");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, 0}, 3)), "Image:", "size of 4x0", "16384");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, -7}, 3)), "Image:", "size of 4x-7");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, K_MAX_IMAGE_SIDE + 1}, 3)), "Image:",
                    "size of 4x16385");
}

/** @brief Tests that construction stops on a channel count outside one to four. */
TEST(Medium_ImagingImage, ConstructionRejectsAChannelCountOutsideItsBounds)
{
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, 4}, 0)), "Image:", "channel count of 0",
                    "[1, 4]");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, 4}, -1)), "Image:", "channel count of -1");
    SA_EXPECT_FATAL(static_cast<void>(Image(ImageSize{4, 4}, K_MAX_CHANNELS + 1)), "Image:",
                    "channel count of 5");
}

/** @brief Tests that row y starts y times the row bytes after the first byte. */
TEST(Small_ImagingImage, RowsStartOneRowOfBytesApart)
{
    Image image(ImageSize{3, 5}, 3);

    expect_row_starts_at(image, 0, 0);
    expect_row_starts_at(image, 1, 15);
    expect_row_starts_at(image, 2, 30);
}

/** @brief Tests that a byte written through a row lands at its height, width, channel offset. */
TEST(Small_ImagingImage, AByteWrittenThroughARowIsReadAtItsHwcOffset)
{
    Image image(ImageSize{3, 4}, 3);
    std::vector<std::uint8_t> expected(36, 0);

    // Row 2, column 1, channel 2 is byte (2 * 4 + 1) * 3 + 2 = 29.
    image.row(2)[1 * 3 + 2] = 201;
    expected[29] = 201;

    expect_bytes(image, expected);
}

/** @brief Tests that fill sets every byte of the image. */
TEST(Small_ImagingImage, FillSetsEveryByte)
{
    Image image(ImageSize{4, 3}, 2);
    image.fill(114);

    expect_every_byte(image, 114);
}

TEST(Small_ImagingImage, FillOnAnEmptyImageDoesNothing)
{
    Image image;
    image.fill(9);

    EXPECT_EQ(image.byte_count(), 0u);
}

/** @brief Tests that reset gives the image the new size, channels, row bytes and byte count. */
TEST(Small_ImagingImage, ResetGivesTheImageNewExtents)
{
    Image image(ImageSize{2, 2}, 1);
    image.reset(ImageSize{5, 3}, 4);

    EXPECT_EQ(image.size(), (ImageSize{5, 3}));
    EXPECT_EQ(image.channels(), 4);
    EXPECT_EQ(image.row_bytes(), 12u);
    EXPECT_EQ(image.byte_count(), 60u);
}

/** @brief Tests that every byte of the extent a reset gives is writable and readable. */
TEST(Small_ImagingImage, EveryByteOfTheResetExtentIsWritable)
{
    Image image(ImageSize{2, 2}, 1);
    image.reset(ImageSize{5, 3}, 4);

    write_offsets(image);

    expect_offsets(image);
}

TEST(Small_ImagingImage, ResetOnADefaultImageAllocates)
{
    Image image;
    image.reset(ImageSize{3, 2}, 3);

    ASSERT_NE(image.data(), nullptr);
    EXPECT_EQ(image.byte_count(), 18u);
}

TEST(Small_ImagingImage, ResetKeepsTheAllocationWhenItIsLargeEnough)
{
    Image image(ImageSize{8, 8}, 3);
    const std::uint8_t* allocation = image.data();

    image.reset(ImageSize{2, 3}, 1);
    EXPECT_EQ(image.data(), allocation);
    EXPECT_EQ(image.byte_count(), 6u);

    // 4 * 16 * 3 is 192, the byte count the image was constructed with.
    image.reset(ImageSize{4, 16}, 3);
    EXPECT_EQ(image.data(), allocation);
    EXPECT_EQ(image.byte_count(), 192u);
}

/** @brief Tests that reset stops on the sizes and channel counts the constructor refuses. */
TEST(Medium_ImagingImage, ResetRejectsTheBoundsTheConstructorRejects)
{
    Image image(ImageSize{2, 2}, 3);

    SA_EXPECT_FATAL(static_cast<void>(image.reset(ImageSize{0, 2}, 3)), "Image::reset",
                    "size of 0x2");
    SA_EXPECT_FATAL(static_cast<void>(image.reset(ImageSize{2, K_MAX_IMAGE_SIDE + 1}, 3)),
                    "Image::reset", "size of 2x16385");
    SA_EXPECT_FATAL(static_cast<void>(image.reset(ImageSize{2, 2}, 5)), "Image::reset",
                    "channel count of 5");
}

/** @brief Tests that a reset refused for its channel count leaves the image as it was. */
TEST(Medium_ImagingImage, ARejectedResetLeavesTheImageAsItWas)
{
    Image image(ImageSize{2, 3}, 3);
    write_offsets(image);
    const std::uint8_t* allocation = image.data();

    SA_EXPECT_FATAL(static_cast<void>(image.reset(ImageSize{4, 4}, 0)), "channel count");

    EXPECT_EQ(image.size(), (ImageSize{2, 3}));
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(image.data(), allocation);
    expect_offsets(image);
}

TEST(Small_ImagingImage, AViewOfAnImageDescribesTheSamePixels)
{
    Image image(ImageSize{3, 4}, 3);
    write_offsets(image);

    const ImageView view = image.view();

    EXPECT_EQ(view.data(), image.data());
    EXPECT_EQ(view.size(), image.size());
    EXPECT_EQ(view.height(), 3);
    EXPECT_EQ(view.width(), 4);
    EXPECT_EQ(view.channels(), 3);
    EXPECT_EQ(view.row_bytes(), image.row_bytes());
    EXPECT_EQ(view.byte_count(), image.byte_count());
    EXPECT_EQ(view.row(2)[5], 29);
}

TEST(Small_ImagingImage, AViewOverForeignBytesAddressesThemInHwcOrder)
{
    const std::array<std::uint8_t, 12> bytes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const ImageView view(bytes.data(), ImageSize{2, 3}, 2);

    EXPECT_EQ(view.data(), bytes.data());
    EXPECT_EQ(view.row_bytes(), 6u);
    EXPECT_EQ(view.byte_count(), 12u);
    EXPECT_EQ(view.row(0), bytes.data());
    EXPECT_EQ(view.row(1), bytes.data() + 6);

    // Row 1, column 2, channel 1 is byte (1 * 3 + 2) * 2 + 1 = 11.
    EXPECT_EQ(view.row(1)[2 * 2 + 1], 11);
}

TEST(Small_ImagingImage, ACopyOwnsItsOwnPixels)
{
    Image original(ImageSize{2, 2}, 3);
    write_offsets(original);

    Image copy = original;
    copy.fill(255);

    ASSERT_NE(copy.data(), original.data());
    EXPECT_EQ(copy.size(), original.size());

    for (std::size_t i = 0; i < original.byte_count(); ++i)
        ASSERT_EQ(static_cast<std::size_t>(original.data()[i]), i) << "byte " << i;
}

/** @brief Tests that the allocation never shrinks and is what a smaller reset keeps. */
TEST(Small_ImagingImage, AllocatedBytesKeepsTheLargestSizeTheImageHad)
{
    Image image(ImageSize{4, 4}, 3);

    image.reset(ImageSize{2, 2}, 1);

    EXPECT_EQ(image.byte_count(), 4u);
    EXPECT_EQ(image.allocated_bytes(), 48u);
}

/** @brief Tests that a view of the whole image and a view of some of its rows both overlap it. */
TEST(Small_ImagingImage, OverlapsFindsTheWholeImageAndAViewInsideIt)
{
    const Image image(ImageSize{4, 3}, 3);
    const ImageView inner(image.row(1), ImageSize{2, 3}, 3);
    const ImageView last_byte(image.data() + image.byte_count() - 1, ImageSize{1, 1}, 1);

    EXPECT_TRUE(overlaps(image.view(), image));
    EXPECT_TRUE(overlaps(inner, image));
    EXPECT_TRUE(overlaps(last_byte, image));
}

/** @brief Tests that a view over bytes another container owns does not overlap an image. */
TEST(Small_ImagingImage, OverlapsIsFalseForBytesOfAnotherContainer)
{
    const std::array<std::uint8_t, 24> bytes{};
    const Image image(ImageSize{2, 2}, 3);
    const ImageView foreign(bytes.data(), ImageSize{2, 4}, 3);

    EXPECT_FALSE(overlaps(foreign, image));
}

/** @brief Tests that bytes a smaller reset left behind still count as the image's own. */
TEST(Small_ImagingImage, OverlapsCountsTheAllocationBeyondTheCurrentSize)
{
    Image image(ImageSize{4, 4}, 3);
    const ImageView tail(image.row(3), ImageSize{1, 4}, 3);

    image.reset(ImageSize{1, 1}, 1);

    EXPECT_TRUE(overlaps(tail, image));
}

/** @brief Tests that separate images, empty views and unallocated images overlap nothing. */
TEST(Small_ImagingImage, OverlapsIsFalseForSeparateOrEmptyOperands)
{
    const Image first(ImageSize{2, 2}, 3);
    const Image second(ImageSize{2, 2}, 3);
    const Image unallocated;
    const ImageView no_extent(first.data(), ImageSize{0, 2}, 3);
    const ImageView no_channels(first.data(), ImageSize{2, 2}, 0);

    EXPECT_FALSE(overlaps(first.view(), second));
    EXPECT_FALSE(overlaps(ImageView(), first));
    EXPECT_FALSE(overlaps(first.view(), unallocated));
    EXPECT_FALSE(overlaps(no_extent, first));
    EXPECT_FALSE(overlaps(no_channels, first));
}

/** @brief Tests that check_source stops on a view without pixels and names the caller. */
TEST(Medium_ImagingImage, CheckSourceStopsOnAViewWithoutPixels)
{
    const Image image(ImageSize{4, 3}, 3);

    SA_EXPECT_FATAL(static_cast<void>(check_source(ImageView(), image, "caller")), "caller:",
                    "no pixels");
}

/** @brief Tests that check_source stops on a source height or width outside the side bounds. */
TEST(Medium_ImagingImage, CheckSourceStopsOnASourceSizeOutsideItsBounds)
{
    const Image image(ImageSize{4, 3}, 3);
    const Image other(ImageSize{2, 2}, 3);
    const ImageView flat(other.data(), ImageSize{0, 2}, 3);
    const ImageView oversized(other.data(), ImageSize{2, K_MAX_IMAGE_SIDE + 1}, 3);

    SA_EXPECT_FATAL(static_cast<void>(check_source(flat, image, "caller")), "caller:",
                    "source size of 0x2");
    SA_EXPECT_FATAL(static_cast<void>(check_source(oversized, image, "caller")), "caller:",
                    "source size of 2x16385", "16384");
}

/** @brief Tests that check_source stops on a source channel count outside one to four. */
TEST(Medium_ImagingImage, CheckSourceStopsOnASourceChannelCountOutsideItsBounds)
{
    const Image image(ImageSize{4, 3}, 3);
    const Image other(ImageSize{2, 2}, 3);
    const ImageView many_channels(other.data(), ImageSize{2, 2}, 5);

    SA_EXPECT_FATAL(static_cast<void>(check_source(many_channels, image, "caller")), "caller:",
                    "source channel count of 5", "[1, 4]");
}

/** @brief Tests that check_source stops on a source the destination holds, whole or in rows. */
TEST(Medium_ImagingImage, CheckSourceStopsOnASourceTheDestinationHolds)
{
    Image image(ImageSize{4, 3}, 3);
    const ImageView inner(image.row(2), ImageSize{2, 3}, 3);

    SA_EXPECT_FATAL(static_cast<void>(check_source(image.view(), image, "caller")), "caller:",
                    "destination holds the source's pixels");
    SA_EXPECT_FATAL(static_cast<void>(check_source(inner, image, "caller")), "caller:",
                    "destination holds the source's pixels");
}

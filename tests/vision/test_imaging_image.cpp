/**************************************************************************/
/* test_imaging_image.cpp                                                 */
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
 * @file test_imaging_image.cpp
 * @brief Tests the 8-bit image's construction, bounds, storage reuse, rows and views.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/image.hpp>
#include "../common/test_throws.hpp"

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
    /**
     * @brief Writes the byte's own offset, modulo 251, into every byte of an image.
     * @param image Any image; an empty one is left alone.
     */
    void write_offsets(Image& image)
    {
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(i % 251);
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

TEST(Unit_ImagingImage, ImageSizeComparesHeightAndWidth)
{
    EXPECT_TRUE((ImageSize{3, 4} == ImageSize{3, 4}));
    EXPECT_TRUE((ImageSize{3, 4} != ImageSize{3, 5}));
    EXPECT_TRUE((ImageSize{3, 4} != ImageSize{2, 4}));
    EXPECT_FALSE((ImageSize{3, 4} == ImageSize{4, 3}));
}

TEST(Unit_ImagingImage, ADefaultImageHasNoPixels)
{
    const Image image;

    EXPECT_EQ(image.height(), 0);
    EXPECT_EQ(image.width(), 0);
    EXPECT_EQ(image.channels(), 0);
    EXPECT_EQ(image.row_bytes(), 0u);
    EXPECT_EQ(image.byte_count(), 0u);
    EXPECT_EQ(image.view().byte_count(), 0u);
    EXPECT_EQ(image.allocated_bytes(), 0u);
}

TEST(Unit_ImagingImage, AConstructedImageReportsItsExtents)
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

TEST(Unit_ImagingImage, AConstructedImageIsZero)
{
    const Image image(ImageSize{5, 7}, 4);

    for (std::size_t i = 0; i < image.byte_count(); ++i)
        ASSERT_EQ(image.data()[i], 0) << "byte " << i;
}

TEST(Unit_ImagingImage, TheLargestSideAndChannelCountAreAccepted)
{
    const Image tall(ImageSize{K_MAX_IMAGE_SIDE, 1}, 1);
    const Image wide(ImageSize{1, K_MAX_IMAGE_SIDE}, K_MAX_CHANNELS);

    EXPECT_EQ(tall.byte_count(), static_cast<std::size_t>(K_MAX_IMAGE_SIDE));
    EXPECT_EQ(wide.byte_count(), static_cast<std::size_t>(K_MAX_IMAGE_SIDE) * K_MAX_CHANNELS);
}

TEST(Unit_ImagingImage, ConstructionRejectsAHeightOutsideItsBounds)
{
    SA_EXPECT_ERROR(Image(ImageSize{0, 4}, 3), "Image:", "size of 0x4", "16384");
    SA_EXPECT_ERROR(Image(ImageSize{-2, 4}, 3), "Image:", "size of -2x4");
    SA_EXPECT_ERROR(Image(ImageSize{K_MAX_IMAGE_SIDE + 1, 4}, 3), "Image:", "size of 16385x4");
}

TEST(Unit_ImagingImage, ConstructionRejectsAWidthOutsideItsBounds)
{
    SA_EXPECT_ERROR(Image(ImageSize{4, 0}, 3), "Image:", "size of 4x0", "16384");
    SA_EXPECT_ERROR(Image(ImageSize{4, -7}, 3), "Image:", "size of 4x-7");
    SA_EXPECT_ERROR(Image(ImageSize{4, K_MAX_IMAGE_SIDE + 1}, 3), "Image:", "size of 4x16385");
}

TEST(Unit_ImagingImage, ConstructionRejectsAChannelCountOutsideItsBounds)
{
    SA_EXPECT_ERROR(Image(ImageSize{4, 4}, 0), "Image:", "channel count of 0", "[1, 4]");
    SA_EXPECT_ERROR(Image(ImageSize{4, 4}, -1), "Image:", "channel count of -1");
    SA_EXPECT_ERROR(Image(ImageSize{4, 4}, K_MAX_CHANNELS + 1), "Image:", "channel count of 5");
}

TEST(Unit_ImagingImage, RowsStartOneRowOfBytesApart)
{
    Image image(ImageSize{3, 5}, 3);
    const Image& constant = image;

    for (std::int32_t y = 0; y < image.height(); ++y)
    {
        const std::size_t offset = static_cast<std::size_t>(y) * 15u;

        EXPECT_EQ(image.row(y), image.data() + offset) << "row " << y;
        EXPECT_EQ(constant.row(y), constant.data() + offset) << "row " << y;
        EXPECT_EQ(image.view().row(y), image.data() + offset) << "row " << y;
    }
}

TEST(Unit_ImagingImage, AByteWrittenThroughARowIsReadAtItsHwcOffset)
{
    Image image(ImageSize{3, 4}, 3);

    // Row 2, column 1, channel 2 is byte (2 * 4 + 1) * 3 + 2 = 29.
    image.row(2)[1 * 3 + 2] = 201;

    for (std::size_t i = 0; i < image.byte_count(); ++i)
        ASSERT_EQ(image.data()[i], i == 29 ? 201 : 0) << "byte " << i;
}

TEST(Unit_ImagingImage, FillSetsEveryByte)
{
    Image image(ImageSize{4, 3}, 2);
    image.fill(114);

    for (std::size_t i = 0; i < image.byte_count(); ++i)
        ASSERT_EQ(image.data()[i], 114) << "byte " << i;
}

TEST(Unit_ImagingImage, FillOnAnEmptyImageDoesNothing)
{
    Image image;
    image.fill(9);

    EXPECT_EQ(image.byte_count(), 0u);
}

TEST(Unit_ImagingImage, ResetGivesTheImageNewExtents)
{
    Image image(ImageSize{2, 2}, 1);
    image.reset(ImageSize{5, 3}, 4);

    EXPECT_EQ(image.size(), (ImageSize{5, 3}));
    EXPECT_EQ(image.channels(), 4);
    EXPECT_EQ(image.row_bytes(), 12u);
    EXPECT_EQ(image.byte_count(), 60u);

    // Every byte of the new extent is writable and readable.
    write_offsets(image);

    for (std::size_t i = 0; i < image.byte_count(); ++i)
        ASSERT_EQ(static_cast<std::size_t>(image.data()[i]), i % 251) << "byte " << i;
}

TEST(Unit_ImagingImage, ResetOnADefaultImageAllocates)
{
    Image image;
    image.reset(ImageSize{3, 2}, 3);

    ASSERT_NE(image.data(), nullptr);
    EXPECT_EQ(image.byte_count(), 18u);
}

TEST(Unit_ImagingImage, ResetKeepsTheAllocationWhenItIsLargeEnough)
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

TEST(Unit_ImagingImage, ResetRejectsTheBoundsTheConstructorRejects)
{
    Image image(ImageSize{2, 2}, 3);

    SA_EXPECT_ERROR(image.reset(ImageSize{0, 2}, 3), "Image::reset", "size of 0x2");
    SA_EXPECT_ERROR(image.reset(ImageSize{2, K_MAX_IMAGE_SIDE + 1}, 3), "Image::reset",
                    "size of 2x16385");
    SA_EXPECT_ERROR(image.reset(ImageSize{2, 2}, 5), "Image::reset", "channel count of 5");
}

TEST(Unit_ImagingImage, ARejectedResetLeavesTheImageAsItWas)
{
    Image image(ImageSize{2, 3}, 3);
    write_offsets(image);
    const std::uint8_t* allocation = image.data();

    SA_EXPECT_ERROR(image.reset(ImageSize{4, 4}, 0), "channel count");

    EXPECT_EQ(image.size(), (ImageSize{2, 3}));
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(image.data(), allocation);

    for (std::size_t i = 0; i < image.byte_count(); ++i)
        ASSERT_EQ(static_cast<std::size_t>(image.data()[i]), i % 251) << "byte " << i;
}

TEST(Unit_ImagingImage, AViewOfAnImageDescribesTheSamePixels)
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

TEST(Unit_ImagingImage, ADefaultViewIsEmpty)
{
    const ImageView view;

    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), (ImageSize{0, 0}));
    EXPECT_EQ(view.channels(), 0);
    EXPECT_EQ(view.row_bytes(), 0u);
    EXPECT_EQ(view.byte_count(), 0u);
}

TEST(Unit_ImagingImage, AViewOverForeignBytesAddressesThemInHwcOrder)
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

TEST(Unit_ImagingImage, ACopyOwnsItsOwnPixels)
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
TEST(Unit_ImagingImage, AllocatedBytesKeepsTheLargestSizeTheImageHad)
{
    Image image(ImageSize{4, 4}, 3);

    image.reset(ImageSize{2, 2}, 1);

    EXPECT_EQ(image.byte_count(), 4u);
    EXPECT_EQ(image.allocated_bytes(), 48u);
}

/** @brief Tests that a view of the whole image and a view of some of its rows both overlap it. */
TEST(Unit_ImagingImage, OverlapsFindsTheWholeImageAndAViewInsideIt)
{
    const Image image(ImageSize{4, 3}, 3);
    const ImageView inner(image.row(1), ImageSize{2, 3}, 3);
    const ImageView last_byte(image.data() + image.byte_count() - 1, ImageSize{1, 1}, 1);

    EXPECT_TRUE(overlaps(image.view(), image));
    EXPECT_TRUE(overlaps(inner, image));
    EXPECT_TRUE(overlaps(last_byte, image));
}

/** @brief Tests that a view over bytes another container owns does not overlap an image. */
TEST(Unit_ImagingImage, OverlapsIsFalseForBytesOfAnotherContainer)
{
    const std::array<std::uint8_t, 24> bytes{};
    const Image image(ImageSize{2, 2}, 3);
    const ImageView foreign(bytes.data(), ImageSize{2, 4}, 3);

    EXPECT_FALSE(overlaps(foreign, image));
}

/** @brief Tests that bytes a smaller reset left behind still count as the image's own. */
TEST(Unit_ImagingImage, OverlapsCountsTheAllocationBeyondTheCurrentSize)
{
    Image image(ImageSize{4, 4}, 3);
    const ImageView tail(image.row(3), ImageSize{1, 4}, 3);

    image.reset(ImageSize{1, 1}, 1);

    EXPECT_TRUE(overlaps(tail, image));
}

/** @brief Tests that separate images, empty views and unallocated images overlap nothing. */
TEST(Unit_ImagingImage, OverlapsIsFalseForSeparateOrEmptyOperands)
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

/** @brief Tests that check_source accepts a valid view of another image. */
TEST(Unit_ImagingImage, CheckSourceAcceptsAViewOfAnotherImage)
{
    const Image source(ImageSize{2, 2}, 3);
    const Image destination(ImageSize{5, 5}, 1);
    const Image unallocated;

    EXPECT_NO_THROW(check_source(source.view(), destination, "caller"));
    EXPECT_NO_THROW(check_source(source.view(), unallocated, "caller"));
}

/** @brief Tests that check_source names the caller and each of the four faults it finds. */
TEST(Unit_ImagingImage, CheckSourceRejectsEachFaultUnderTheCallersName)
{
    Image image(ImageSize{4, 3}, 3);
    const Image other(ImageSize{2, 2}, 3);
    const ImageView flat(other.data(), ImageSize{0, 2}, 3);
    const ImageView oversized(other.data(), ImageSize{2, K_MAX_IMAGE_SIDE + 1}, 3);
    const ImageView many_channels(other.data(), ImageSize{2, 2}, 5);
    const ImageView inner(image.row(2), ImageSize{2, 3}, 3);

    SA_EXPECT_ERROR(check_source(ImageView(), image, "caller"), "caller:", "no pixels");
    SA_EXPECT_ERROR(check_source(flat, image, "caller"), "caller:", "source size of 0x2");
    SA_EXPECT_ERROR(check_source(oversized, image, "caller"), "caller:",
                    "source size of 2x16385", "16384");
    SA_EXPECT_ERROR(check_source(many_channels, image, "caller"), "caller:",
                    "source channel count of 5", "[1, 4]");
    SA_EXPECT_ERROR(check_source(image.view(), image, "caller"), "caller:",
                    "destination holds the source's pixels");
    SA_EXPECT_ERROR(check_source(inner, image, "caller"), "caller:",
                    "destination holds the source's pixels");
}

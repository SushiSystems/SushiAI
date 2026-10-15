/**************************************************************************/
/* test_resize.cpp                                                        */
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
 * @file test_resize.cpp
 * @brief Tests the resizes on hand-computed images and the training-load size rule.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

using SushiAI::Vision::Imaging::fit_long_side;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;
using SushiAI::Vision::Imaging::ImageView;
using SushiAI::Vision::Imaging::K_MAX_IMAGE_SIDE;
using SushiAI::Vision::Imaging::resize_bilinear;
using SushiAI::Vision::Imaging::resize_long_side;
using SushiAI::Vision::Imaging::resize_nearest;

namespace
{
    /** @brief Names the bytes of an image in HWC order. */
    using Bytes = std::vector<std::uint8_t>;

    /**
     * @brief Returns an image holding the given bytes in HWC order.
     * @param bytes Exactly size.height * size.width * channels values.
     */
    [[nodiscard]] Image make_image(ImageSize size, std::int32_t channels, const Bytes& bytes)
    {
        Image image(size, channels);
        EXPECT_EQ(image.byte_count(), bytes.size());

        for (std::size_t i = 0; i < image.byte_count() && i < bytes.size(); ++i)
            image.data()[i] = bytes[i];

        return image;
    }

    /** @brief Returns the bytes of an image in HWC order. */
    [[nodiscard]] Bytes bytes_of(const Image& image)
    {
        return Bytes(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns the bytes of a single-channel resize of the given source. */
    [[nodiscard]] Bytes resized_bytes(ImageSize source_size, const Bytes& source, ImageSize size)
    {
        const Image image = make_image(source_size, 1, source);
        Image destination;
        resize_bilinear(image.view(), size, destination);

        EXPECT_EQ(destination.size(), size);
        EXPECT_EQ(destination.channels(), 1);

        return bytes_of(destination);
    }
} // namespace

TEST(Small_ImagingResize, WeightBitsGiveAWeightSumOf2048)
{
    EXPECT_EQ(1 << SushiAI::Vision::Imaging::K_BILINEAR_WEIGHT_BITS, 2048);
}

TEST(Small_ImagingResize, TwoByTwoToFourByFourMatchesTheHandComputedImage)
{
    // Along an axis the outputs of a pair (a, b) are a, (3a + b) / 4, (a + 3b) / 4 and b.
    const Bytes source{  0, 100,
                       200,  40};

    // Row 1 is (3 * row 0 + row 3) / 4 rounded half up: 58.75 to 59 and 76.25 to 76.
    const Bytes expected{  0,  25,  75, 100,
                          50,  59,  76,  85,
                         150, 126,  79,  55,
                         200, 160,  80,  40};

    EXPECT_EQ(resized_bytes(ImageSize{2, 2}, source, ImageSize{4, 4}), expected);
}

TEST(Small_ImagingResize, FourByFourToTwoByTwoAveragesEachTwoByTwoBlock)
{
    // Each output reads position 0.5 or 2.5 on both axes, so all four weights are 1024.
    const Bytes source{ 12,  20,  30,  40,
                        50,  60,  70,  80,
                        90, 100, 110, 120,
                       130, 140, 150, 161};

    // The block sums are 142, 220, 460 and 541; 35.5 rounds up to 36 and 135.25 down to 135.
    const Bytes expected{ 36,  55,
                         115, 135};

    EXPECT_EQ(resized_bytes(ImageSize{4, 4}, source, ImageSize{2, 2}), expected);
}

TEST(Small_ImagingResize, ResizeToTheSameSizeCopiesEveryPixel)
{
    const Bytes source{  0, 255,  17,
                       114,   1, 254,
                        63, 128, 200};

    EXPECT_EQ(resized_bytes(ImageSize{3, 3}, source, ImageSize{3, 3}), source);
}

TEST(Small_ImagingResize, AnEnlargedRowClampsAtBothEdges)
{
    // Columns 0 and 3 read positions -0.25 and 1.25, outside the centres 0 and 1.
    const Bytes row = resized_bytes(ImageSize{1, 2}, Bytes{10, 250}, ImageSize{1, 4});

    ASSERT_EQ(row.size(), 4u);
    EXPECT_EQ(row.front(), 10);
    EXPECT_EQ(row.back(), 250);
}

TEST(Small_ImagingResize, AReducedRowUsesWeightsRoundedToElevenBits)
{
    const Bytes source{0, 3, 10, 40, 41, 90, 200, 255};

    // The columns read 0.8333, 3.5 and 6.1667, whose upper weights are 1707, 1024 and 341.
    const Bytes expected{(341 * 0 + 1707 * 3 + 1024) / 2048,
                         (1024 * 40 + 1024 * 41 + 1024) / 2048,
                         (1707 * 200 + 341 * 255 + 1024) / 2048};

    EXPECT_EQ(expected, (Bytes{3, 41, 209}));
    EXPECT_EQ(resized_bytes(ImageSize{1, 8}, source, ImageSize{1, 3}), expected);
}

TEST(Small_ImagingResize, EachChannelIsInterpolatedOnItsOwn)
{
    const Image source = make_image(ImageSize{1, 2}, 3, Bytes{0, 40, 255, 100, 80, 3});
    Image destination;

    resize_bilinear(source.view(), ImageSize{1, 4}, destination);

    // Channel 2 falls from 255 to 3: (765 + 3) / 4 is 192 and (255 + 9) / 4 is 66.
    const Bytes expected{0, 40, 255, 25, 50, 192, 75, 70, 66, 100, 80, 3};

    EXPECT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), expected);
}

TEST(Small_ImagingResize, AConstantImageStaysConstantWhenEnlarged)
{
    Image source(ImageSize{5, 7}, 3);

    for (std::size_t i = 0; i < source.byte_count(); i += 3)
    {
        source.data()[i] = 17;
        source.data()[i + 1] = 114;
        source.data()[i + 2] = 255;
    }

    Image destination;
    resize_bilinear(source.view(), ImageSize{13, 23}, destination);

    ASSERT_EQ(destination.byte_count(), 13u * 23u * 3u);

    for (std::size_t i = 0; i < destination.byte_count(); i += 3)
    {
        ASSERT_EQ(destination.data()[i], 17) << "byte " << i;
        ASSERT_EQ(destination.data()[i + 1], 114) << "byte " << i + 1;
        ASSERT_EQ(destination.data()[i + 2], 255) << "byte " << i + 2;
    }
}

TEST(Small_ImagingResize, AConstantImageStaysConstantWhenReduced)
{
    Image source(ImageSize{31, 17}, 1);
    source.fill(201);

    Image destination;
    resize_bilinear(source.view(), ImageSize{4, 5}, destination);

    ASSERT_EQ(destination.byte_count(), 20u);

    for (std::size_t i = 0; i < destination.byte_count(); ++i)
        ASSERT_EQ(destination.data()[i], 201) << "byte " << i;
}

TEST(Small_ImagingResize, AHorizontalGradientStaysMonotoneAtEveryWidth)
{
    const Bytes source{0, 3, 10, 40, 41, 90, 200, 255};

    for (std::int32_t width = 1; width <= 40; ++width)
    {
        const Bytes row = resized_bytes(ImageSize{1, 8}, source, ImageSize{1, width});

        ASSERT_EQ(row.size(), static_cast<std::size_t>(width));

        for (std::size_t x = 1; x < row.size(); ++x)
            ASSERT_LE(row[x - 1], row[x]) << "width " << width << ", column " << x;
    }
}

TEST(Small_ImagingResize, AnEnlargedGradientKeepsItsEndValues)
{
    const Bytes source{0, 3, 10, 40, 41, 90, 200, 255};
    const Bytes row = resized_bytes(ImageSize{1, 8}, source, ImageSize{1, 20});

    ASSERT_EQ(row.size(), 20u);
    EXPECT_EQ(row.front(), 0);
    EXPECT_EQ(row.back(), 255);
}

TEST(Small_ImagingResize, AVerticalGradientStaysMonotoneAndRowsStayUniform)
{
    // Every row of the source is one level, so every row of the result must be one level.
    const Bytes source{  5,   5,   5,
                        60,  60,  60,
                        61,  61,  61,
                       250, 250, 250};

    for (std::int32_t height = 1; height <= 17; ++height)
    {
        const Bytes image = resized_bytes(ImageSize{4, 3}, source, ImageSize{height, 5});

        ASSERT_EQ(image.size(), static_cast<std::size_t>(height) * 5u);

        for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y)
        {
            for (std::size_t x = 1; x < 5; ++x)
                ASSERT_EQ(image[y * 5], image[y * 5 + x]) << "height " << height << ", row " << y;

            if (y > 0)
                ASSERT_LE(image[(y - 1) * 5], image[y * 5]) << "height " << height << ", row " << y;
        }
    }
}

TEST(Small_ImagingResize, TheDestinationIsResetToTheRequestedSizeAndChannels)
{
    const Image source = make_image(ImageSize{1, 2}, 3, Bytes{1, 2, 3, 4, 5, 6});
    Image destination(ImageSize{9, 9}, 1);
    destination.fill(77);

    resize_bilinear(source.view(), ImageSize{2, 2}, destination);

    EXPECT_EQ(destination.size(), (ImageSize{2, 2}));
    EXPECT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), (Bytes{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}));
}

TEST(Small_ImagingResize, ResizeRejectsASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_ERROR(resize_bilinear(ImageView(), ImageSize{2, 2}, destination),
                    "resize_bilinear", "source", "no pixels");
}

TEST(Small_ImagingResize, ResizeRejectsASizeOutsideTheImageBounds)
{
    const Image source(ImageSize{2, 2}, 1);
    Image destination;

    SA_EXPECT_ERROR(resize_bilinear(source.view(), ImageSize{0, 2}, destination),
                    "resize_bilinear", "destination size", "0x2");
    SA_EXPECT_ERROR(resize_bilinear(source.view(), ImageSize{2, -1}, destination),
                    "resize_bilinear", "destination size", "2x-1");
    SA_EXPECT_ERROR(resize_bilinear(source.view(), ImageSize{2, K_MAX_IMAGE_SIDE + 1}, destination),
                    "resize_bilinear", "destination size", "2x16385");
}

TEST(Small_ImagingResize, ResizeRejectsASourceWithExtentsOutsideTheImageBounds)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_ERROR(resize_bilinear(ImageView(bytes.data(), ImageSize{0, 2}, 1), ImageSize{2, 2},
                                    destination),
                    "resize_bilinear", "source size", "0x2");
    SA_EXPECT_ERROR(resize_bilinear(ImageView(bytes.data(), ImageSize{2, K_MAX_IMAGE_SIDE + 1}, 1),
                                    ImageSize{2, 2}, destination),
                    "resize_bilinear", "source size", "2x16385");
    SA_EXPECT_ERROR(resize_bilinear(ImageView(bytes.data(), ImageSize{2, 2}, 5), ImageSize{2, 2},
                                    destination),
                    "resize_bilinear", "source channel count", "5");
}

TEST(Small_ImagingResize, ResizeRejectsADestinationThatHoldsTheSource)
{
    Image image(ImageSize{2, 2}, 1);
    image.fill(9);

    SA_EXPECT_ERROR(resize_bilinear(image.view(), ImageSize{4, 4}, image),
                    "resize_bilinear", "destination holds the source's pixels");

    // The refusal comes before the reset, so the image is still the 2 by 2 source.
    EXPECT_EQ(image.size(), (ImageSize{2, 2}));
    EXPECT_EQ(bytes_of(image), (Bytes{9, 9, 9, 9}));
}

TEST(Small_ImagingResize, FitLongSideLeavesAnImageWhoseLongSideAlreadyFits)
{
    EXPECT_EQ(fit_long_side(ImageSize{480, 640}, 640), (ImageSize{480, 640}));
    EXPECT_EQ(fit_long_side(ImageSize{640, 427}, 640), (ImageSize{640, 427}));
    EXPECT_EQ(fit_long_side(ImageSize{640, 640}, 640), (ImageSize{640, 640}));
}

TEST(Small_ImagingResize, FitLongSideReducesALargeImage)
{
    // 640 / 1920 is one third, and 1080 / 3 is 360.
    EXPECT_EQ(fit_long_side(ImageSize{1080, 1920}, 640), (ImageSize{360, 640}));
    EXPECT_EQ(fit_long_side(ImageSize{1920, 1080}, 640), (ImageSize{640, 360}));
}

TEST(Small_ImagingResize, FitLongSideEnlargesASmallImage)
{
    // 640 / 100 is 6.4, and 50 * 6.4 is 320.
    EXPECT_EQ(fit_long_side(ImageSize{100, 50}, 640), (ImageSize{640, 320}));

    // 640 / 500 is 1.28, and 375 * 1.28 is 480.
    EXPECT_EQ(fit_long_side(ImageSize{500, 375}, 640), (ImageSize{640, 480}));
}

TEST(Small_ImagingResize, FitLongSideRoundsTheShortSideUp)
{
    // 333 * 1.28 is 426.24, which the ceiling takes to 427.
    EXPECT_EQ(fit_long_side(ImageSize{333, 500}, 640), (ImageSize{427, 640}));

    // 3 * 640 / 7 is 274.29, which the ceiling takes to 275.
    EXPECT_EQ(fit_long_side(ImageSize{7, 3}, 640), (ImageSize{640, 275}));

    // 3 * 0.64 is 1.92, which the ceiling takes to 2.
    EXPECT_EQ(fit_long_side(ImageSize{3, 1000}, 640), (ImageSize{2, 640}));
}

TEST(Small_ImagingResize, FitLongSideCapsBothSidesAtTheImageSize)
{
    // 1280 * 640 / 1281 is 639.5, whose ceiling is the image size itself.
    EXPECT_EQ(fit_long_side(ImageSize{1281, 1280}, 640), (ImageSize{640, 640}));
}

TEST(Small_ImagingResize, FitLongSideAlwaysPutsTheLongSideAtTheImageSize)
{
    for (std::int32_t height = 1; height < 1500; height += 7)
    {
        for (std::int32_t width = 1; width < 1500; width += 11)
        {
            const ImageSize fitted = fit_long_side(ImageSize{height, width}, 640);
            const std::int32_t long_side = height >= width ? fitted.height : fitted.width;
            const std::int32_t short_side = height >= width ? fitted.width : fitted.height;

            ASSERT_EQ(long_side, 640) << height << " by " << width;
            ASSERT_GE(short_side, 1) << height << " by " << width;
            ASSERT_LE(short_side, 640) << height << " by " << width;
        }
    }
}

TEST(Small_ImagingResize, ResizeLongSideCopiesAnImageThatAlreadyFits)
{
    const Bytes bytes{0, 255, 17, 114, 1, 254, 63, 128};
    const Image source = make_image(ImageSize{2, 4}, 1, bytes);
    Image destination(ImageSize{3, 3}, 3);

    resize_long_side(source.view(), 4, destination);

    EXPECT_EQ(destination.size(), (ImageSize{2, 4}));
    EXPECT_EQ(destination.channels(), 1);
    EXPECT_NE(destination.data(), source.data());
    EXPECT_EQ(bytes_of(destination), bytes);
}

TEST(Small_ImagingResize, ResizeLongSideEnlargesToTheFittedSize)
{
    const Image source = make_image(ImageSize{1, 2}, 1, Bytes{0, 100});
    Image destination;

    resize_long_side(source.view(), 4, destination);

    // fit_long_side gives 2 by 4, and both rows are the enlarged row 0, 25, 75, 100.
    EXPECT_EQ(destination.size(), (ImageSize{2, 4}));
    EXPECT_EQ(bytes_of(destination), (Bytes{0, 25, 75, 100, 0, 25, 75, 100}));
}

TEST(Small_ImagingResize, ResizeLongSideReducesToTheFittedSize)
{
    const Bytes bytes{ 12,  20,  30,  40,
                       50,  60,  70,  80,
                       90, 100, 110, 120,
                      130, 140, 150, 161};
    const Image source = make_image(ImageSize{4, 4}, 1, bytes);
    Image destination;

    resize_long_side(source.view(), 2, destination);

    EXPECT_EQ(destination.size(), (ImageSize{2, 2}));
    EXPECT_EQ(bytes_of(destination), (Bytes{36, 55, 115, 135}));
}

TEST(Small_ImagingResize, ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds)
{
    const Image source(ImageSize{2, 2}, 1);
    Image destination;

    SA_EXPECT_ERROR(resize_long_side(source.view(), 0, destination), "resize_long_side", "0");
    SA_EXPECT_ERROR(resize_long_side(source.view(), K_MAX_IMAGE_SIDE + 1, destination),
                    "resize_long_side", "16385");
}

TEST(Small_ImagingResize, ResizeLongSideRejectsASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_ERROR(resize_long_side(ImageView(), 640, destination),
                    "resize_long_side", "no pixels");
}

TEST(Small_ImagingResize, ResizeLongSideRejectsADestinationThatHoldsTheSource)
{
    Image image(ImageSize{2, 4}, 1);

    SA_EXPECT_ERROR(resize_long_side(image.view(), 4, image),
                    "resize_long_side", "destination holds the source's pixels");
}

/** @brief Tests that both resizes refuse a view of some rows of the destination, unwritten. */
TEST(Small_ImagingResize, ResizesRejectAViewInsideTheDestination)
{
    Image image(ImageSize{3, 2}, 1);
    image.fill(7);
    const ImageView lower_rows(image.row(1), ImageSize{2, 2}, 1);

    SA_EXPECT_ERROR(resize_bilinear(lower_rows, ImageSize{4, 4}, image),
                    "resize_bilinear", "destination holds the source's pixels");
    SA_EXPECT_ERROR(resize_long_side(lower_rows, 8, image),
                    "resize_long_side", "destination holds the source's pixels");

    EXPECT_EQ(image.size(), (ImageSize{3, 2}));
    EXPECT_EQ(bytes_of(image), (Bytes{7, 7, 7, 7, 7, 7}));
}

/** @brief Tests that resize_long_side refuses a source size outside the image bounds. */
TEST(Small_ImagingResize, ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_ERROR(resize_long_side(ImageView(bytes.data(), ImageSize{0, 2}, 1), 4, destination),
                    "resize_long_side", "source size", "0x2");
    SA_EXPECT_ERROR(resize_long_side(ImageView(bytes.data(), ImageSize{2, 2}, 5), 4, destination),
                    "resize_long_side", "source channel count", "5");
}

/** @brief Tests that an enlargement repeats each source pixel over its block. */
TEST(Small_ImagingResize, NearestRepeatsEachPixelWhenEnlarged)
{
    const Image source = make_image(ImageSize{2, 2}, 1, Bytes{1, 2, 3, 4});
    Image destination;

    resize_nearest(source.view(), ImageSize{4, 4}, destination);

    EXPECT_EQ(destination.size(), (ImageSize{4, 4}));
    EXPECT_EQ(destination.channels(), 1);
    EXPECT_EQ(bytes_of(destination),
              (Bytes{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
}

/** @brief Tests that a reduction reads the pixel under each destination centre. */
TEST(Small_ImagingResize, NearestReadsThePixelUnderEachCentreWhenReduced)
{
    // Centres 0.5 and 1.5 of two columns fall at 1.25 and 3.75 of five: columns 1 and 3.
    const Image source = make_image(ImageSize{1, 5}, 1, Bytes{10, 11, 12, 13, 14});
    Image destination;

    resize_nearest(source.view(), ImageSize{1, 2}, destination);

    EXPECT_EQ(bytes_of(destination), (Bytes{11, 13}));
}

/** @brief Tests that seven columns read from two hold the two source values alone. */
TEST(Small_ImagingResize, NearestKeepsClassIndicesUnblended)
{
    const Image source = make_image(ImageSize{1, 2}, 1, Bytes{0, 255});
    Image destination;

    resize_nearest(source.view(), ImageSize{1, 7}, destination);

    EXPECT_EQ(bytes_of(destination), (Bytes{0, 0, 0, 255, 255, 255, 255}));
}

/** @brief Tests that the channels of a pixel are copied together. */
TEST(Small_ImagingResize, NearestCopiesEveryChannelOfAPixel)
{
    const Image source = make_image(ImageSize{1, 2}, 3, Bytes{1, 2, 3, 4, 5, 6});
    Image destination(ImageSize{9, 9}, 1);

    resize_nearest(source.view(), ImageSize{2, 2}, destination);

    EXPECT_EQ(destination.size(), (ImageSize{2, 2}));
    EXPECT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), (Bytes{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}));
}

/** @brief Tests that resize_nearest refuses what check_source and Image::reset refuse. */
TEST(Small_ImagingResize, NearestRejectsABadSourceABadSizeAndAnAliasedDestination)
{
    Image image(ImageSize{2, 2}, 1);
    image.fill(9);
    Image destination;

    SA_EXPECT_ERROR(resize_nearest(ImageView(), ImageSize{2, 2}, destination),
                    "resize_nearest", "source", "no pixels");
    SA_EXPECT_ERROR(resize_nearest(image.view(), ImageSize{0, 2}, destination),
                    "Image::reset", "0x2");
    SA_EXPECT_ERROR(resize_nearest(image.view(), ImageSize{4, 4}, image),
                    "resize_nearest", "destination holds the source's pixels");
    EXPECT_EQ(bytes_of(image), (Bytes{9, 9, 9, 9}));
}

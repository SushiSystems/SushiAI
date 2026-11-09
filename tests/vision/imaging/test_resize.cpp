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
 * @file test_resize.cpp
 * @brief Tests the resizes on hand-computed images and the training-load size rule.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/resize.hpp>
#include <support/test_names.hpp>

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
    using SushiAI::StatusCode;
    using SushiAI::Test::require_value;

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
        SA_EXPECT_OK(resize_bilinear(image.view(), size, destination));

        EXPECT_EQ(destination.size(), size);
        EXPECT_EQ(destination.channels(), 1);

        return bytes_of(destination);
    }

    /** @brief Returns an image whose pixels all hold the given level of each channel. */
    [[nodiscard]] Image make_constant_image(ImageSize size, const Bytes& levels)
    {
        Image image(size, static_cast<std::int32_t>(levels.size()));
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = levels[i % levels.size()];

        return image;
    }

    /** @brief Checks that every pixel of an image holds the given level of each channel. */
    void expect_constant_image(const Image& image, const Bytes& levels)
    {
        ASSERT_EQ(image.channels(), static_cast<std::int32_t>(levels.size()));
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            ASSERT_EQ(image.data()[i], levels[i % levels.size()]) << "byte " << i;
    }

    /** @brief Checks that a row never falls, resized to every width from 1 to @p max_width. */
    void expect_row_monotone_at_every_width(const Bytes& source, std::int32_t max_width)
    {
        for (std::int32_t width = 1; width <= max_width; ++width)
        {
            const Bytes row = resized_bytes(ImageSize{1, 8}, source, ImageSize{1, width});

            ASSERT_EQ(row.size(), static_cast<std::size_t>(width));

            for (std::size_t x = 1; x < row.size(); ++x)
                ASSERT_LE(row[x - 1], row[x]) << "width " << width << ", column " << x;
        }
    }

    /** @brief Returns a 4 by 3 source whose every row is one level, rising down the image. */
    [[nodiscard]] Bytes make_row_levels_source()
    {
        return Bytes{  5,   5,   5,
                      60,  60,  60,
                      61,  61,  61,
                     250, 250, 250};
    }

    /** @brief Checks that the first column never falls, resized to every height up to @p max. */
    void expect_column_monotone_at_every_height(std::int32_t max_height)
    {
        for (std::int32_t height = 1; height <= max_height; ++height)
        {
            const Bytes image =
                resized_bytes(ImageSize{4, 3}, make_row_levels_source(), ImageSize{height, 5});

            ASSERT_EQ(image.size(), static_cast<std::size_t>(height) * 5u);

            for (std::size_t y = 1; y < static_cast<std::size_t>(height); ++y)
                ASSERT_LE(image[(y - 1) * 5], image[y * 5]) << "height " << height << ", row "
                                                            << y;
        }
    }

    /** @brief Checks that every row holds one level, resized to every height up to @p max. */
    void expect_rows_uniform_at_every_height(std::int32_t max_height)
    {
        for (std::int32_t height = 1; height <= max_height; ++height)
        {
            const Bytes image =
                resized_bytes(ImageSize{4, 3}, make_row_levels_source(), ImageSize{height, 5});

            ASSERT_EQ(image.size(), static_cast<std::size_t>(height) * 5u);

            for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y)
                for (std::size_t x = 1; x < 5; ++x)
                    ASSERT_EQ(image[y * 5], image[y * 5 + x]) << "height " << height << ", row "
                                                              << y;
        }
    }

    /** @brief Checks that the long side fits the image size and the short side stays within it. */
    void expect_long_side_at_image_size(std::int32_t image_size)
    {
        for (std::int32_t height = 1; height < 1500; height += 7)
        {
            for (std::int32_t width = 1; width < 1500; width += 11)
            {
                const ImageSize fitted = fit_long_side(ImageSize{height, width}, image_size);
                const std::int32_t long_side = height >= width ? fitted.height : fitted.width;
                const std::int32_t short_side = height >= width ? fitted.width : fitted.height;

                ASSERT_EQ(long_side, image_size) << height << " by " << width;
                ASSERT_GE(short_side, 1) << height << " by " << width;
                ASSERT_LE(short_side, image_size) << height << " by " << width;
            }
        }
    }

    /** @brief Describes a constant source and the destination size it is resized to. */
    struct ConstantRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the level of each channel of the source. */
        Bytes levels;

        /** @brief Holds the destination size. */
        ImageSize destination;
    };

    /** @brief Runs each constant-image case. */
    class Small_ImagingResizeConstant : public ::testing::TestWithParam<ConstantRow>
    {
    };

    /** @brief Describes a source, a long side and the fitted size and bytes of its resize. */
    struct LongSideRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the source bytes. */
        Bytes bytes;

        /** @brief Holds the long side asked for. */
        std::int32_t long_side;

        /** @brief Holds the size fit_long_side gives. */
        ImageSize fitted;

        /** @brief Holds the bytes of the resized image. */
        Bytes expected;
    };

    /** @brief Runs each resize_long_side case. */
    class Small_ImagingResizeLongSide : public ::testing::TestWithParam<LongSideRow>
    {
    };

    /** @brief Describes a resize of a view of a destination image into that image. */
    struct AliasRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the size of the destination image. */
        ImageSize image_size;

        /** @brief Holds the row of the image the view starts at. */
        std::int32_t view_row;

        /** @brief Holds the size of the view. */
        ImageSize view_size;

        /** @brief Resizes the view into the image. */
        void (*resize)(const ImageView&, Image&);

        /** @brief Names the function in its message. */
        const char* function;
    };

    /** @brief Runs each aliased-destination refusal; each starts a child process. */
    class Medium_ImagingResizeAliasing : public ::testing::TestWithParam<AliasRow>
    {
    };
} // namespace

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

    SA_EXPECT_OK(resize_bilinear(source.view(), ImageSize{1, 4}, destination));

    // Channel 2 falls from 255 to 3: (765 + 3) / 4 is 192 and (255 + 9) / 4 is 66.
    const Bytes expected{0, 40, 255, 25, 50, 192, 75, 70, 66, 100, 80, 3};

    EXPECT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), expected);
}

/** @brief Tests that a constant image stays constant, in every channel, when resized. */
TEST_P(Small_ImagingResizeConstant, AConstantImageStaysConstant)
{
    const ConstantRow& row = GetParam();
    const Image source = make_constant_image(row.source, row.levels);
    Image destination;

    SA_EXPECT_OK(resize_bilinear(source.view(), row.destination, destination));

    ASSERT_EQ(destination.size(), row.destination);
    expect_constant_image(destination, row.levels);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingResizeConstant,
    ::testing::Values(
        ConstantRow{"WhenEnlarged", ImageSize{5, 7}, Bytes{17, 114, 255}, ImageSize{13, 23}},
        ConstantRow{"WhenReduced", ImageSize{31, 17}, Bytes{201}, ImageSize{4, 5}}),
    [](const ::testing::TestParamInfo<ConstantRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that a horizontal gradient never falls, at every width from 1 to 40. */
TEST(Small_ImagingResize, AHorizontalGradientStaysMonotoneAtEveryWidth)
{
    expect_row_monotone_at_every_width(Bytes{0, 3, 10, 40, 41, 90, 200, 255}, 40);
}

TEST(Small_ImagingResize, AnEnlargedGradientKeepsItsEndValues)
{
    const Bytes source{0, 3, 10, 40, 41, 90, 200, 255};
    const Bytes row = resized_bytes(ImageSize{1, 8}, source, ImageSize{1, 20});

    ASSERT_EQ(row.size(), 20u);
    EXPECT_EQ(row.front(), 0);
    EXPECT_EQ(row.back(), 255);
}

/** @brief Tests that a vertical gradient never falls, at every height from 1 to 17. */
TEST(Small_ImagingResize, AVerticalGradientStaysMonotone)
{
    expect_column_monotone_at_every_height(17);
}

/** @brief Tests that every row of a vertical gradient stays one level, at every height. */
TEST(Small_ImagingResize, EachRowOfAVerticalGradientStaysUniform)
{
    // Every row of the source is one level, so every row of the result must be one level.
    expect_rows_uniform_at_every_height(17);
}

TEST(Small_ImagingResize, TheDestinationIsResetToTheRequestedSizeAndChannels)
{
    const Image source = make_image(ImageSize{1, 2}, 3, Bytes{1, 2, 3, 4, 5, 6});
    Image destination(ImageSize{9, 9}, 1);
    destination.fill(77);

    SA_EXPECT_OK(resize_bilinear(source.view(), ImageSize{2, 2}, destination));

    EXPECT_EQ(destination.size(), (ImageSize{2, 2}));
    EXPECT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), (Bytes{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}));
}

/** @brief Tests that resize_bilinear refuses a source without pixels. */
TEST(Medium_ImagingResize, ResizeRejectsASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_bilinear(ImageView(), ImageSize{2, 2}, destination)),
                    "resize_bilinear",
                    "source", "no pixels");
}

TEST(Small_ImagingResize, ResizeRejectsASizeOutsideTheImageBounds)
{
    const Image source(ImageSize{2, 2}, 1);
    Image destination;

    SA_EXPECT_FAILURE(resize_bilinear(source.view(), ImageSize{0, 2}, destination),
                      StatusCode::InvalidDescription, "resize_bilinear", "destination size", "0x2");
    SA_EXPECT_FAILURE(resize_bilinear(source.view(), ImageSize{2, -1}, destination),
                      StatusCode::InvalidDescription, "resize_bilinear", "destination size",
                      "2x-1");
    SA_EXPECT_FAILURE(resize_bilinear(source.view(), ImageSize{2, K_MAX_IMAGE_SIDE + 1},
                                      destination),
                      StatusCode::InvalidDescription, "resize_bilinear", "destination size",
                      "2x16385");
}

/** @brief Tests that resize_bilinear refuses a source whose extents are outside the bounds. */
TEST(Medium_ImagingResize, ResizeRejectsASourceWithExtentsOutsideTheImageBounds)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_bilinear(ImageView(bytes.data(), ImageSize{0, 2}, 1),
                                                      ImageSize{2, 2},
                                    destination)),
                    "resize_bilinear", "source size", "0x2");
    SA_EXPECT_FATAL(static_cast<void>(resize_bilinear(ImageView(bytes.data(),
                                                                ImageSize{2, K_MAX_IMAGE_SIDE + 1},
                                                                1),
                                    ImageSize{2, 2}, destination)),
                    "resize_bilinear", "source size", "2x16385");
}

/** @brief Tests that resize_bilinear refuses a source with a channel count it cannot read. */
TEST(Medium_ImagingResize, ResizeRejectsASourceWithAnUnsupportedChannelCount)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_bilinear(ImageView(bytes.data(), ImageSize{2, 2}, 5),
                                                      ImageSize{2, 2},
                                    destination)),
                    "resize_bilinear", "source channel count", "5");
}

/** @brief Tests that resize_bilinear refuses a destination that holds the source. */
TEST(Medium_ImagingResize, ResizeRejectsADestinationThatHoldsTheSource)
{
    Image image(ImageSize{2, 2}, 1);
    image.fill(9);

    SA_EXPECT_FATAL(static_cast<void>(resize_bilinear(image.view(), ImageSize{4, 4}, image)),
                    "resize_bilinear",
                    "destination holds the source's pixels");
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

/** @brief Tests that, over a grid of sizes, the long side is the image size and the short fits. */
TEST(Small_ImagingResize, FitLongSideAlwaysPutsTheLongSideAtTheImageSize)
{
    expect_long_side_at_image_size(640);
}

TEST(Small_ImagingResize, ResizeLongSideCopiesAnImageThatAlreadyFits)
{
    const Bytes bytes{0, 255, 17, 114, 1, 254, 63, 128};
    const Image source = make_image(ImageSize{2, 4}, 1, bytes);
    Image destination(ImageSize{3, 3}, 3);

    SA_EXPECT_OK(resize_long_side(source.view(), 4, destination));

    EXPECT_EQ(destination.size(), (ImageSize{2, 4}));
    EXPECT_EQ(destination.channels(), 1);
    EXPECT_NE(destination.data(), source.data());
    EXPECT_EQ(bytes_of(destination), bytes);
}

/** @brief Tests that resize_long_side gives the bilinear resize to the fitted size. */
TEST_P(Small_ImagingResizeLongSide, ResizeLongSideGivesTheResizeToTheFittedSize)
{
    const LongSideRow& row = GetParam();
    const Image source = make_image(row.source, 1, row.bytes);
    Image destination;

    SA_EXPECT_OK(resize_long_side(source.view(), row.long_side, destination));

    EXPECT_EQ(destination.size(), row.fitted);
    EXPECT_EQ(bytes_of(destination), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingResizeLongSide,
    ::testing::Values(
        // fit_long_side gives 2 by 4, and both rows are the enlarged row 0, 25, 75, 100.
        LongSideRow{"Enlarges", ImageSize{1, 2}, Bytes{0, 100}, 4, ImageSize{2, 4},
                    Bytes{0, 25, 75, 100, 0, 25, 75, 100}},
        LongSideRow{"Reduces", ImageSize{4, 4},
                    Bytes{12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 161},
                    2, ImageSize{2, 2}, Bytes{36, 55, 115, 135}}),
    [](const ::testing::TestParamInfo<LongSideRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

TEST(Small_ImagingResize, ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds)
{
    const Image source(ImageSize{2, 2}, 1);
    Image destination;

    SA_EXPECT_FAILURE(resize_long_side(source.view(), 0, destination),
                      StatusCode::InvalidDescription, "resize_long_side", "0");
    SA_EXPECT_FAILURE(resize_long_side(source.view(), K_MAX_IMAGE_SIDE + 1, destination),
                      StatusCode::InvalidDescription, "resize_long_side", "16385");
}

/** @brief Tests that resize_long_side refuses a source without pixels. */
TEST(Medium_ImagingResize, ResizeLongSideRejectsASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_long_side(ImageView(), 640, destination)),
                    "resize_long_side",
                    "no pixels");
}

/** @brief Tests that both resizes refuse a view of the destination as their source. */
TEST_P(Medium_ImagingResizeAliasing, ResizeRejectsAViewOfTheDestination)
{
    const AliasRow& row = GetParam();
    Image image(row.image_size, 1);
    image.fill(7);
    const ImageView view(image.row(row.view_row), row.view_size, 1);

    SA_EXPECT_FATAL(row.resize(view, image), row.function,
                    "destination holds the source's pixels");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_ImagingResizeAliasing,
    ::testing::Values(
        AliasRow{"LongSideOfTheWholeImage", ImageSize{2, 4}, 0, ImageSize{2, 4},
                 [](const ImageView& view, Image& image)
                 {
                     static_cast<void>(resize_long_side(view, 4, image));
                 },
                 "resize_long_side"},
        AliasRow{"BilinearOfTheLowerRows", ImageSize{3, 2}, 1, ImageSize{2, 2},
                 [](const ImageView& view, Image& image)
                 {
                     static_cast<void>(resize_bilinear(view, ImageSize{4, 4}, image));
                 },
                 "resize_bilinear"},
        AliasRow{"LongSideOfTheLowerRows", ImageSize{3, 2}, 1, ImageSize{2, 2},
                 [](const ImageView& view, Image& image)
                 {
                     static_cast<void>(resize_long_side(view, 8, image));
                 },
                 "resize_long_side"}),
    [](const ::testing::TestParamInfo<AliasRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that resize_long_side refuses a source size outside the image bounds. */
TEST(Medium_ImagingResize, ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_long_side(ImageView(bytes.data(), ImageSize{0, 2}, 1),
                                                       4, destination)),
                    "resize_long_side", "source size", "0x2");
}

/** @brief Tests that resize_long_side refuses a source with a channel count it cannot read. */
TEST(Medium_ImagingResize, ResizeLongSideRejectsASourceWithAnUnsupportedChannelCount)
{
    const std::array<std::uint8_t, 20> bytes{};
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_long_side(ImageView(bytes.data(), ImageSize{2, 2}, 5),
                                                       4, destination)),
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

/** @brief Tests that resize_nearest refuses a source without pixels. */
TEST(Medium_ImagingResize, NearestRejectsASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_nearest(ImageView(), ImageSize{2, 2}, destination)),
                    "resize_nearest",
                    "source", "no pixels");
}

/** @brief Tests that resize_nearest refuses a destination size with a zero side. */
TEST(Medium_ImagingResize, NearestRejectsADestinationSizeWithAZeroSide)
{
    Image image(ImageSize{2, 2}, 1);
    image.fill(9);
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(resize_nearest(image.view(), ImageSize{0, 2}, destination)),
                    "Image::reset", "0x2");
}

/** @brief Tests that resize_nearest refuses a destination that holds the source. */
TEST(Medium_ImagingResize, NearestRejectsADestinationThatHoldsTheSource)
{
    Image image(ImageSize{2, 2}, 1);
    image.fill(9);

    SA_EXPECT_FATAL(static_cast<void>(resize_nearest(image.view(), ImageSize{4, 4}, image)),
                    "resize_nearest",
                    "destination holds the source's pixels");
}

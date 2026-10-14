/**************************************************************************/
/* test_letterbox.cpp                                                     */
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
 * @file test_letterbox.cpp
 * @brief Tests the letterbox plan, its padding of an image and its mapping of boxes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

using namespace SushiAI;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::LetterboxPlan;
using Vision::Imaging::LetterboxSettings;
using Vision::Imaging::PixelBox;

namespace
{
    /** @brief Holds the absolute tolerance, in pixels, of every box coordinate compared here. */
    constexpr float K_BOX_TOLERANCE = 1e-4f;

    /** @brief Returns an image whose byte at offset i is first + i modulo 256. */
    Image make_counted_image(ImageSize size, std::int32_t channels, std::uint8_t first)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns an image with every byte at one level. */
    Image make_flat_image(ImageSize size, std::int32_t channels, std::uint8_t level)
    {
        Image image(size, channels);
        image.fill(level);
        return image;
    }

    /** @brief Reports whether a pixel lies inside the resized image of a plan. */
    bool is_content(const LetterboxPlan& plan, std::int32_t y, std::int32_t x)
    {
        return y >= plan.top && y < plan.top + plan.resized.height && x >= plan.left &&
               x < plan.left + plan.resized.width;
    }

    /** @brief Checks that every content byte is one level and every padding byte another. */
    void expect_two_levels(const Image& image, const LetterboxPlan& plan, std::uint8_t content,
                           std::uint8_t padding)
    {
        for (std::int32_t y = 0; y < image.height(); ++y)
            for (std::int32_t x = 0; x < image.width(); ++x)
                for (std::int32_t c = 0; c < image.channels(); ++c)
                {
                    const std::uint8_t expected = is_content(plan, y, x) ? content : padding;
                    ASSERT_EQ(image.row(y)[x * image.channels() + c], expected)
                        << "row " << y << ", column " << x << ", channel " << c;
                }
    }
} // namespace

TEST(Unit_ImagingLetterbox, WideImageIsPaddedAboveAndBelowWithTheOddRowBelow)
{
    const LetterboxPlan plan = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{50, 200}, ImageSize{100, 100});

    EXPECT_DOUBLE_EQ(plan.ratio, 0.5);
    EXPECT_EQ(plan.resized, (ImageSize{25, 100}));
    EXPECT_EQ(plan.top, 37);
    EXPECT_EQ(plan.bottom, 38);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);
    EXPECT_EQ(plan.source, (ImageSize{50, 200}));
    EXPECT_EQ(plan.target, (ImageSize{100, 100}));
}

TEST(Unit_ImagingLetterbox, TallImageIsPaddedLeftAndRightWithTheOddColumnRight)
{
    const LetterboxPlan plan = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100});

    EXPECT_DOUBLE_EQ(plan.ratio, 0.5);
    EXPECT_EQ(plan.resized, (ImageSize{100, 25}));
    EXPECT_EQ(plan.top, 0);
    EXPECT_EQ(plan.bottom, 0);
    EXPECT_EQ(plan.left, 37);
    EXPECT_EQ(plan.right, 38);
}

TEST(Unit_ImagingLetterbox, ExactFitHasRatioOneAndNoPadding)
{
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{64, 64}, ImageSize{64, 64});

    EXPECT_DOUBLE_EQ(plan.ratio, 1.0);
    EXPECT_EQ(plan.resized, (ImageSize{64, 64}));
    EXPECT_EQ(plan.top + plan.bottom + plan.left + plan.right, 0);
}

TEST(Unit_ImagingLetterbox, SmallImageIsOnlyPaddedUnlessScaleUpIsSet)
{
    LetterboxSettings settings;
    const LetterboxPlan padded =
        Vision::Imaging::plan_letterbox(settings, ImageSize{20, 40}, ImageSize{80, 80});

    EXPECT_DOUBLE_EQ(padded.ratio, 1.0);
    EXPECT_EQ(padded.resized, (ImageSize{20, 40}));
    EXPECT_EQ(padded.top, 30);
    EXPECT_EQ(padded.bottom, 30);
    EXPECT_EQ(padded.left, 20);
    EXPECT_EQ(padded.right, 20);

    settings.scale_up = true;
    const LetterboxPlan enlarged =
        Vision::Imaging::plan_letterbox(settings, ImageSize{20, 40}, ImageSize{80, 80});

    EXPECT_DOUBLE_EQ(enlarged.ratio, 2.0);
    EXPECT_EQ(enlarged.resized, (ImageSize{40, 80}));
    EXPECT_EQ(enlarged.top, 20);
    EXPECT_EQ(enlarged.bottom, 20);
    EXPECT_EQ(enlarged.left, 0);
    EXPECT_EQ(enlarged.right, 0);
}

TEST(Unit_ImagingLetterbox, ResizedSideRoundsAHalfToTheEvenInteger)
{
    // 5 * 0.5 is 2.5 and 7 * 0.5 is 3.5; half to even gives 2 and 4.
    const LetterboxPlan down =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{5, 8}, ImageSize{4, 4});
    const LetterboxPlan up =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{7, 8}, ImageSize{4, 4});

    EXPECT_EQ(down.resized, (ImageSize{2, 4}));
    EXPECT_EQ(down.top, 1);
    EXPECT_EQ(down.bottom, 1);
    EXPECT_EQ(up.resized, (ImageSize{4, 4}));
    EXPECT_EQ(up.top, 0);
    EXPECT_EQ(up.bottom, 0);
}

TEST(Unit_ImagingLetterbox, RectangularStrideTargetPadsOnlyTheShortAxis)
{
    // 512 is the stride-32 batch height of a 480x640 image: ceil(480 / 32 + 0.5) * 32.
    const LetterboxPlan plan = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{480, 640}, ImageSize{512, 640});

    EXPECT_DOUBLE_EQ(plan.ratio, 1.0);
    EXPECT_EQ(plan.resized, (ImageSize{480, 640}));
    EXPECT_EQ(plan.top, 16);
    EXPECT_EQ(plan.bottom, 16);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);
}

TEST(Unit_ImagingLetterbox, PlanCopiesThePaddingValueOfTheSettings)
{
    LetterboxSettings settings;
    EXPECT_EQ(
        Vision::Imaging::plan_letterbox(settings, ImageSize{8, 8}, ImageSize{8, 8}).padding_value,
        settings.padding_value);

    settings.padding_value = 7;
    EXPECT_EQ(
        Vision::Imaging::plan_letterbox(settings, ImageSize{8, 8}, ImageSize{8, 8}).padding_value,
        7);
}

TEST(Unit_ImagingLetterbox, PlanRejectsASideThatIsNotPositive)
{
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{0, 8},
                                                  ImageSize{8, 8}),
                    "plan_letterbox", "source size of 0x8");
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{8, 8},
                                                  ImageSize{8, -1}),
                    "plan_letterbox", "target size of 8x-1");
}

/** @brief Tests that the plan refuses a side above the largest an image may have. */
TEST(Unit_ImagingLetterbox, PlanRejectsASideAboveTheImageBound)
{
    const std::int32_t above = Vision::Imaging::K_MAX_IMAGE_SIDE + 1;

    SA_EXPECT_ERROR((void)Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{above, 8},
                                                  ImageSize{8, 8}),
                    "plan_letterbox", "source size of 16385x8", "16384");
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{8, 8},
                                                  ImageSize{8, above}),
                    "plan_letterbox", "target size of 8x16385", "16384");
}

/** @brief Tests that validate_letterbox accepts the extremes of both fields. */
TEST(Unit_ImagingLetterbox, ValidateAcceptsEverySetting)
{
    LetterboxSettings settings;
    settings.scale_up = true;
    settings.padding_value = 255;

    EXPECT_NO_THROW(Vision::Imaging::validate_letterbox(settings));
    EXPECT_NO_THROW(Vision::Imaging::validate_letterbox(LetterboxSettings{}));
}

TEST(Unit_ImagingLetterbox, PlanRejectsAResizedSideOfZeroPixels)
{
    // 1 * (8 / 1000) is 0.008, which rounds to a height of zero.
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{1, 1000},
                                                  ImageSize{8, 8}),
                    "plan_letterbox", "zero pixels");
}

TEST(Unit_ImagingLetterbox, ImageWithoutResizeKeepsItsPixelsInsideThePadding)
{
    const Image source = make_counted_image(ImageSize{2, 4}, 3, 1);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{4, 6});
    ASSERT_EQ(plan.top, 1);
    ASSERT_EQ(plan.left, 1);

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{4, 6}));
    ASSERT_EQ(destination.channels(), 3);
    for (std::int32_t y = 0; y < 4; ++y)
        for (std::int32_t x = 0; x < 6; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
            {
                const std::uint8_t expected = is_content(plan, y, x)
                                                  ? source.row(y - 1)[(x - 1) * 3 + c]
                                                  : plan.padding_value;
                EXPECT_EQ(destination.row(y)[x * 3 + c], expected)
                    << "row " << y << ", column " << x << ", channel " << c;
            }
}

TEST(Unit_ImagingLetterbox, ExactFitCopiesEveryPixel)
{
    const Image source = make_counted_image(ImageSize{6, 6}, 3, 9);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), source.size());

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), source.size());
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        ASSERT_EQ(destination.data()[i], source.data()[i]) << "byte " << i;
}

TEST(Unit_ImagingLetterbox, WideImageIsResizedThenPaddedWithTheFillValue)
{
    // A flat image stays flat under bilinear weights that sum to one.
    const Image source = make_flat_image(ImageSize{50, 200}, 3, 200);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{100, 100});

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{100, 100}));
    expect_two_levels(destination, plan, 200, 114);
}

TEST(Unit_ImagingLetterbox, TallImageIsResizedThenPaddedWithACustomFillValue)
{
    LetterboxSettings settings;
    settings.padding_value = 7;
    const Image source = make_flat_image(ImageSize{200, 50}, 1, 33);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(settings, source.size(), ImageSize{100, 100});

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{100, 100}));
    ASSERT_EQ(destination.channels(), 1);
    expect_two_levels(destination, plan, 33, 7);
}

TEST(Unit_ImagingLetterbox, ResizedContentEqualsTheBilinearResizeOfTheSource)
{
    const Image source = make_counted_image(ImageSize{50, 200}, 3, 0);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{100, 100});
    ASSERT_EQ(plan.resized, (ImageSize{25, 100}));
    ASSERT_EQ(plan.top, 37);
    ASSERT_EQ(plan.left, 0);

    Image expected;
    Vision::Imaging::resize_bilinear(source.view(), plan.resized, expected);

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{100, 100}));
    ASSERT_EQ(destination.channels(), 3);

    bool is_flat = true;
    for (std::int32_t y = 0; y < 100; ++y)
        for (std::int32_t x = 0; x < 100; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
            {
                std::uint8_t wanted = 114;
                if (is_content(plan, y, x))
                {
                    wanted = expected.row(y - plan.top)[(x - plan.left) * 3 + c];
                    is_flat = is_flat && wanted == expected.data()[0];
                }

                ASSERT_EQ(destination.row(y)[x * 3 + c], wanted)
                    << "row " << y << ", column " << x << ", channel " << c;
            }

    EXPECT_FALSE(is_flat) << "a flat content rectangle cannot tell a resize from a fill";
}

TEST(Unit_ImagingLetterbox, SourceWithoutPixelsIsRejected)
{
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{2, 4}, ImageSize{4, 6});

    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_letterbox(Vision::Imaging::ImageView{}, plan,
                                                     destination),
                    "apply_letterbox", "no pixels");
}

TEST(Unit_ImagingLetterbox, DestinationThatIsTheSourceIsRejected)
{
    Image image = make_flat_image(ImageSize{2, 4}, 3, 50);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, image.size(), ImageSize{4, 6});

    SA_EXPECT_ERROR(Vision::Imaging::apply_letterbox(image.view(), plan, image), "apply_letterbox",
                    "destination holds the source's pixels");
    EXPECT_EQ(image.size(), (ImageSize{2, 4}));
}

/** @brief Tests that a view of some rows of the destination is refused before the reset. */
TEST(Unit_ImagingLetterbox, ViewInsideTheDestinationIsRejected)
{
    Image image = make_flat_image(ImageSize{3, 4}, 3, 50);
    const Vision::Imaging::ImageView lower_rows(image.row(1), ImageSize{2, 4}, 3);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{2, 4}, ImageSize{4, 6});

    SA_EXPECT_ERROR(Vision::Imaging::apply_letterbox(lower_rows, plan, image), "apply_letterbox",
                    "destination holds the source's pixels");
    EXPECT_EQ(image.size(), (ImageSize{3, 4}));
    EXPECT_EQ(image.row(2)[0], 50);
}

/** @brief Tests that a resize beside left and right padding writes the kernel's bytes only. */
TEST(Unit_ImagingLetterbox, ResizedContentBesideSidePaddingEqualsTheBilinearResize)
{
    const Image source = make_counted_image(ImageSize{200, 50}, 3, 0);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{100, 101});
    ASSERT_EQ(plan.resized, (ImageSize{100, 25}));
    ASSERT_EQ(plan.top, 0);
    ASSERT_EQ(plan.left, 38);
    ASSERT_EQ(plan.right, 38);

    Image expected;
    Vision::Imaging::resize_bilinear(source.view(), plan.resized, expected);

    Image destination = make_flat_image(ImageSize{120, 120}, 3, 255);
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{100, 101}));
    std::size_t wrong = 0;
    for (std::int32_t y = 0; y < 100; ++y)
        for (std::int32_t x = 0; x < 101; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
            {
                const std::uint8_t wanted =
                    is_content(plan, y, x) ? expected.row(y)[(x - plan.left) * 3 + c] : 114;
                if (destination.row(y)[x * 3 + c] != wanted)
                    ++wrong;
            }

    EXPECT_EQ(wrong, 0u) << "bytes differ from the resize placed between two bands of 114";
}

TEST(Unit_ImagingLetterbox, DestinationIsFullyRewrittenWhenItIsReused)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{4, 6});

    Image destination = make_flat_image(ImageSize{9, 9}, 3, 255);
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{4, 6}));
    expect_two_levels(destination, plan, 50, 114);
}

TEST(Unit_ImagingLetterbox, ImageOfAnotherSizeThanThePlanIsRejected)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{4, 4}, ImageSize{8, 8});

    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_letterbox(source.view(), plan, destination),
                    "apply_letterbox", "plan was made for");
}

TEST(Unit_ImagingLetterbox, PlanWhosePaddingsDoNotTileTheTargetIsRejected)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(), ImageSize{4, 6});
    plan.right += 1;

    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_letterbox(source.view(), plan, destination),
                    "apply_letterbox", "tile");
}

TEST(Unit_ImagingLetterbox, BoxesMoveByTheRatioThenTheLeftAndTopPadding)
{
    const LetterboxPlan tall = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100});
    const LetterboxPlan wide = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{50, 200}, ImageSize{100, 100});

    const Labels labels({3, 7}, {PixelBox{10.0f, 20.0f, 30.0f, 40.0f},
                                 PixelBox{0.0f, 0.0f, 50.0f, 200.0f}});

    // x * 0.5 + 37 and y * 0.5 + 0.
    const Labels in_tall = Vision::Imaging::apply_letterbox(labels, tall);
    ASSERT_EQ(in_tall.size(), 2u);
    EXPECT_EQ(in_tall.classes(), labels.classes());
    EXPECT_NEAR(in_tall.boxes()[0].x1, 42.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[0].y1, 10.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[0].x2, 52.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[0].y2, 20.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[1].x1, 37.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[1].y1, 0.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[1].x2, 62.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_tall.boxes()[1].y2, 100.0f, K_BOX_TOLERANCE);

    // x * 0.5 + 0 and y * 0.5 + 37.
    const Labels in_wide = Vision::Imaging::apply_letterbox(labels, wide);
    ASSERT_EQ(in_wide.size(), 2u);
    EXPECT_NEAR(in_wide.boxes()[0].x1, 5.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_wide.boxes()[0].y1, 47.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_wide.boxes()[0].x2, 15.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(in_wide.boxes()[0].y2, 57.0f, K_BOX_TOLERANCE);
}

TEST(Unit_ImagingLetterbox, BoxesOutsideTheImageAreNeitherClippedNorRemoved)
{
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{20, 40}, ImageSize{80, 80});

    const Labels labels({1, 2}, {PixelBox{-8.0f, -4.0f, 100.0f, 90.0f},
                                 PixelBox{5.0f, 5.0f, 5.0f, 5.0f}});

    // Ratio 1, left 20, top 30.
    const Labels moved = Vision::Imaging::apply_letterbox(labels, plan);
    ASSERT_EQ(moved.size(), 2u);
    EXPECT_EQ(moved.classes(), labels.classes());
    EXPECT_NEAR(moved.boxes()[0].x1, 12.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].y1, 26.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].x2, 120.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].y2, 120.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[1].x1, 25.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[1].y2, 35.0f, K_BOX_TOLERANCE);
}

TEST(Unit_ImagingLetterbox, ExactFitLeavesBoxesWhereTheyWere)
{
    const LetterboxPlan plan =
        Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{64, 64}, ImageSize{64, 64});

    const Labels labels({0}, {PixelBox{1.5f, 2.5f, 60.25f, 63.0f}});

    const Labels moved = Vision::Imaging::apply_letterbox(labels, plan);
    ASSERT_EQ(moved.size(), 1u);
    EXPECT_NEAR(moved.boxes()[0].x1, 1.5f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].y1, 2.5f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].x2, 60.25f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.boxes()[0].y2, 63.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that a polygon moves by the ratio, then by the left and top padding. */
TEST(Unit_ImagingLetterbox, PolygonMovesByTheRatioThenTheLeftAndTopPadding)
{
    const LetterboxPlan tall = Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100});
    ASSERT_EQ(tall.left, 37);
    ASSERT_EQ(tall.top, 0);
    Labels labels;
    labels.add(3, PixelBox{10.0f, 20.0f, 30.0f, 40.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{10.0f, 20.0f},
                                        Vision::Imaging::Point{30.0f, 40.0f},
                                        Vision::Imaging::Point{-4.0f, 210.0f}});
    labels.add(7, PixelBox{0.0f, 0.0f, 50.0f, 200.0f});

    const Labels moved = Vision::Imaging::apply_letterbox(labels, tall);

    ASSERT_EQ(moved.size(), 2u);
    ASSERT_EQ(moved.polygons()[0].size(), 3u);
    EXPECT_NEAR(moved.polygons()[0][0].x, 42.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.polygons()[0][0].y, 10.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.polygons()[0][1].x, 52.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.polygons()[0][1].y, 20.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.polygons()[0][2].x, 35.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(moved.polygons()[0][2].y, 105.0f, K_BOX_TOLERANCE);
    EXPECT_TRUE(moved.polygons()[1].empty());
}

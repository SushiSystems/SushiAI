/**************************************************************************/
/* test_flip.cpp                                                          */
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
 * @file test_flip.cpp
 * @brief Tests the draw of the horizontal flip and its mirroring of pixels and labels.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/flip.hpp>

using namespace SushiAI;
using Vision::Imaging::FlipPlan;
using Vision::Imaging::FlipSettings;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::PixelBox;
using Vision::Imaging::Random;

namespace
{
    /** @brief Holds the absolute tolerance, in pixels, of every box coordinate compared here. */
    constexpr float K_BOX_TOLERANCE = 1e-4f;

    /** @brief Counts the seeds, 0 to this minus one, that a statistical test draws from. */
    constexpr std::uint64_t K_SEED_COUNT = 10000;

    /** @brief Holds the allowed distance of a flip frequency from 0.5, six standard deviations. */
    constexpr double K_FREQUENCY_TOLERANCE = 0.03;

    /** @brief Returns an image whose byte at offset i is i modulo 256. */
    Image make_counted_image(ImageSize size, std::int32_t channels)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(i);

        return image;
    }

    /** @brief Checks that two images have the same size, channels and bytes. */
    void expect_same_image(const Image& actual, const Image& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_EQ(actual.channels(), expected.channels());
        for (std::size_t i = 0; i < expected.byte_count(); ++i)
            ASSERT_EQ(actual.data()[i], expected.data()[i]) << "byte " << i;
    }

    /** @brief Checks the four coordinates of a box against expected values. */
    void expect_box(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_NEAR(actual.x1, expected.x1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y1, expected.y1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.x2, expected.x2, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y2, expected.y2, K_BOX_TOLERANCE);
    }
} // namespace

TEST(Small_ImagingFlip, PlanRecordsTheSizeAndMakesExactlyOneDraw)
{
    Random random(42);
    Random replay(42);

    const FlipPlan plan = Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{480, 640}, random);

    EXPECT_EQ(plan.source, (ImageSize{480, 640}));
    EXPECT_EQ(plan.is_flipped, replay.bernoulli(FlipSettings{}.probability));
    EXPECT_EQ(random, replay);
}

TEST(Small_ImagingFlip, ProbabilityZeroNeverFlipsAndOneAlwaysFlips)
{
    FlipSettings never;
    never.probability = 0.0f;
    FlipSettings always;
    always.probability = 1.0f;

    for (std::uint64_t seed = 0; seed < K_SEED_COUNT; ++seed)
    {
        Random first(seed);
        Random second(seed);
        ASSERT_FALSE(Vision::Imaging::plan_flip(never, ImageSize{8, 8}, first).is_flipped) << seed;
        ASSERT_TRUE(Vision::Imaging::plan_flip(always, ImageSize{8, 8}, second).is_flipped) << seed;
        ASSERT_EQ(first, second) << "both made one draw from seed " << seed;
    }
}

TEST(Small_ImagingFlip, DefaultProbabilityFlipsAboutHalfOfTheSeeds)
{
    std::uint64_t flipped = 0;
    for (std::uint64_t seed = 0; seed < K_SEED_COUNT; ++seed)
    {
        Random random(seed);
        if (Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{8, 8}, random).is_flipped)
            ++flipped;
    }

    const double frequency = static_cast<double>(flipped) / static_cast<double>(K_SEED_COUNT);
    EXPECT_NEAR(frequency, 0.5, K_FREQUENCY_TOLERANCE);
}

TEST(Small_ImagingFlip, PlanRejectsAProbabilityOutsideZeroToOne)
{
    Random random(1);
    FlipSettings settings;

    settings.probability = 1.5f;
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                    "validate_flip", "probability", "1.5");
    SA_EXPECT_ERROR(Vision::Imaging::validate_flip(settings), "validate_flip", "probability");

    settings.probability = -0.1f;
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                    "validate_flip", "probability");
    EXPECT_EQ(random, Random(1)) << "a rejected plan makes no draw";
}

TEST(Small_ImagingFlip, PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw)
{
    Random random(3);
    FlipSettings settings;
    settings.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_ERROR((void)Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                    "validate_flip", "probability");
    EXPECT_EQ(random, Random(3));
}

TEST(Small_ImagingFlip, PlanRejectsASideThatIsNotPositiveBeforeAnyDraw)
{
    Random random(5);

    SA_EXPECT_ERROR((void)Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{0, 8}, random),
                    "plan_flip", "source size", "0x8");
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{8, -2}, random),
                    "plan_flip", "source size", "8x-2");
    EXPECT_EQ(random, Random(5));
}

TEST(Small_ImagingFlip, SourceWithoutPixelsIsRejected)
{
    const FlipPlan plan{ImageSize{2, 3}, true};

    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_flip(Vision::Imaging::ImageView{}, plan, destination),
                    "apply_flip", "no pixels");
}

TEST(Small_ImagingFlip, DestinationThatIsTheSourceIsRejected)
{
    Image image = make_counted_image(ImageSize{2, 3}, 3);
    const Image untouched = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan plan{ImageSize{2, 3}, true};

    SA_EXPECT_ERROR(Vision::Imaging::apply_flip(image.view(), plan, image), "apply_flip",
                    "destination holds the source's pixels");
    expect_same_image(image, untouched);
}

/** @brief Tests that a view of some rows of the destination is refused before any write. */
TEST(Small_ImagingFlip, ViewInsideTheDestinationIsRejected)
{
    Image image = make_counted_image(ImageSize{4, 3}, 3);
    const Image untouched = make_counted_image(ImageSize{4, 3}, 3);
    const Vision::Imaging::ImageView lower_rows(image.row(1), ImageSize{3, 3}, 3);
    const FlipPlan plan{ImageSize{3, 3}, true};

    SA_EXPECT_ERROR(Vision::Imaging::apply_flip(lower_rows, plan, image), "apply_flip",
                    "destination holds the source's pixels");
    expect_same_image(image, untouched);
}

TEST(Small_ImagingFlip, FlippedImageHoldsEachRowInReverseColumnOrder)
{
    const Image source = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan plan{ImageSize{2, 3}, true};

    Image destination;
    Vision::Imaging::apply_flip(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), source.size());
    ASSERT_EQ(destination.channels(), 3);
    for (std::int32_t y = 0; y < 2; ++y)
        for (std::int32_t x = 0; x < 3; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
                EXPECT_EQ(destination.row(y)[x * 3 + c], source.row(y)[(2 - x) * 3 + c])
                    << "row " << y << ", column " << x << ", channel " << c;

    // Row 0 is bytes 0..8; mirrored, its first pixel is bytes 6, 7, 8.
    EXPECT_EQ(destination.row(0)[0], 6);
    EXPECT_EQ(destination.row(0)[1], 7);
    EXPECT_EQ(destination.row(0)[2], 8);
}

TEST(Small_ImagingFlip, UnflippedImageIsCopiedUnchanged)
{
    const Image source = make_counted_image(ImageSize{3, 5}, 3);
    const FlipPlan plan{ImageSize{3, 5}, false};

    Image destination;
    Vision::Imaging::apply_flip(source.view(), plan, destination);

    expect_same_image(destination, source);
}

TEST(Small_ImagingFlip, FlippingAnImageTwiceRestoresIt)
{
    const Image wide = make_counted_image(ImageSize{4, 7}, 3);
    const Image gray = make_counted_image(ImageSize{3, 4}, 1);

    for (const Image* source : {&wide, &gray})
    {
        const FlipPlan plan{source->size(), true};
        Image once;
        Image twice;
        Vision::Imaging::apply_flip(source->view(), plan, once);
        Vision::Imaging::apply_flip(once.view(), plan, twice);

        expect_same_image(twice, *source);
    }
}

TEST(Small_ImagingFlip, ImageOfAnotherSizeThanThePlanIsRejected)
{
    const Image source = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan wider{ImageSize{2, 4}, true};
    const FlipPlan taller{ImageSize{5, 3}, true};

    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_flip(source.view(), wider, destination), "apply_flip",
                    "plan was made for a 2x4 source", "2x3");
    SA_EXPECT_ERROR(Vision::Imaging::apply_flip(source.view(), taller, destination), "apply_flip",
                    "plan was made for a 5x3 source", "2x3");
}

TEST(Small_ImagingFlip, FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder)
{
    const FlipPlan plan{ImageSize{640, 640}, true};

    const Labels labels({4, 9}, {PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
                                 PixelBox{0.0f, 10.0f, 640.0f, 20.0f}});

    const Labels mirrored = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(mirrored.size(), 2u);
    EXPECT_EQ(mirrored.classes(), labels.classes());
    expect_box(mirrored.boxes()[0], PixelBox{340.0f, 50.0f, 540.0f, 200.0f});
    expect_box(mirrored.boxes()[1], PixelBox{0.0f, 10.0f, 640.0f, 20.0f});
    EXPECT_LE(mirrored.boxes()[0].x1, mirrored.boxes()[0].x2);
}

TEST(Small_ImagingFlip, UnflippedBoxesAreReturnedUnchanged)
{
    const FlipPlan plan{ImageSize{640, 640}, false};

    const Labels labels({4}, {PixelBox{100.0f, 50.0f, 300.0f, 200.0f}});

    const Labels same = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(same.size(), 1u);
    EXPECT_EQ(same.classes(), labels.classes());
    expect_box(same.boxes()[0], labels.boxes()[0]);
}

TEST(Small_ImagingFlip, FlippingBoxesTwiceRestoresThem)
{
    const FlipPlan plan{ImageSize{640, 640}, true};

    const Labels labels({1, 2}, {PixelBox{12.5f, 3.0f, 99.25f, 40.0f},
                                 PixelBox{600.0f, 0.0f, 640.0f, 640.0f}});

    const Labels twice =
        Vision::Imaging::apply_flip(Vision::Imaging::apply_flip(labels, plan), plan);

    ASSERT_EQ(twice.size(), 2u);
    EXPECT_EQ(twice.classes(), labels.classes());
    expect_box(twice.boxes()[0], labels.boxes()[0]);
    expect_box(twice.boxes()[1], labels.boxes()[1]);
}

/** @brief Tests that a flipped polygon has every x mirrored about the width and every y kept. */
TEST(Small_ImagingFlip, FlippedPolygonMirrorsEveryVertexAboutTheWidth)
{
    const FlipPlan plan{ImageSize{480, 640}, true};
    Labels labels;
    labels.add(2, PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{100.0f, 50.0f},
                                        Vision::Imaging::Point{300.0f, 60.0f},
                                        Vision::Imaging::Point{150.5f, 200.0f}});
    labels.add(3, PixelBox{0.0f, 0.0f, 10.0f, 10.0f});

    const Labels mirrored = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(mirrored.size(), 2u);
    ASSERT_EQ(mirrored.polygons()[0].size(), 3u);
    EXPECT_NEAR(mirrored.polygons()[0][0].x, 540.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(mirrored.polygons()[0][0].y, 50.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(mirrored.polygons()[0][1].x, 340.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(mirrored.polygons()[0][1].y, 60.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(mirrored.polygons()[0][2].x, 489.5f, K_BOX_TOLERANCE);
    EXPECT_NEAR(mirrored.polygons()[0][2].y, 200.0f, K_BOX_TOLERANCE);
    EXPECT_TRUE(mirrored.polygons()[1].empty());
}

/** @brief Tests that an unflipped plan returns the polygons as they were. */
TEST(Small_ImagingFlip, UnflippedPolygonIsReturnedUnchanged)
{
    const FlipPlan plan{ImageSize{480, 640}, false};
    Labels labels;
    labels.add(2, PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{100.0f, 50.0f},
                                        Vision::Imaging::Point{300.0f, 60.0f}});

    const Labels same = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(same.polygons()[0].size(), 2u);
    EXPECT_NEAR(same.polygons()[0][1].x, 300.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(same.polygons()[0][1].y, 60.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that validate_flip accepts both ends of the probability range. */
TEST(Small_ImagingFlip, ValidateAcceptsZeroAndOne)
{
    FlipSettings never;
    never.probability = 0.0f;
    FlipSettings always;
    always.probability = 1.0f;

    EXPECT_NO_THROW(Vision::Imaging::validate_flip(never));
    EXPECT_NO_THROW(Vision::Imaging::validate_flip(always));
    EXPECT_NO_THROW(Vision::Imaging::validate_flip(FlipSettings{}));
}

/**************************************************************************/
/* test_imaging_hsv.cpp                                                   */
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
 * @file test_imaging_hsv.cpp
 * @brief Tests the HSV jitter: its draws, its lookup tables, the 8-bit hexcone and the image pass.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Vision::Imaging::HsvPixel;
using Vision::Imaging::HsvPlan;
using Vision::Imaging::HsvSettings;
using Vision::Imaging::HsvTables;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::K_HUE_STEPS;
using Vision::Imaging::Random;
using Vision::Imaging::RgbPixel;

namespace
{
    /** @brief Counts the seeds, 0 to this minus one, that a statistical test draws from. */
    constexpr std::uint64_t K_SEED_COUNT = 10000;

    /** @brief Holds the largest level error of to_hsv then to_rgb over all 2^24 colours. */
    constexpr std::int32_t K_ROUND_TRIP_LEVELS = 4;

    /** @brief Holds the step between the levels of the colour sweep. */
    constexpr std::int32_t K_SWEEP_STEP = 5;

    /** @brief Returns a one-row RGB image of the given pixels. */
    Image make_row(std::initializer_list<RgbPixel> pixels)
    {
        Image image(ImageSize{1, static_cast<std::int32_t>(pixels.size())}, 3);
        std::uint8_t* out = image.data();
        for (const RgbPixel& pixel : pixels)
        {
            *out++ = pixel.red;
            *out++ = pixel.green;
            *out++ = pixel.blue;
        }

        return image;
    }

    /** @brief Returns the pixel at a column of a one-row RGB image. */
    RgbPixel pixel_at(const Image& image, std::int32_t x)
    {
        const std::uint8_t* in = image.row(0) + x * 3;
        return RgbPixel{in[0], in[1], in[2]};
    }

    /** @brief Returns the result of jittering a single pixel with a plan. */
    RgbPixel jitter(RgbPixel pixel, const HsvPlan& plan)
    {
        const Image source = make_row({pixel});
        Image destination;
        Vision::Imaging::apply_hsv(source.view(), plan, destination);
        return pixel_at(destination, 0);
    }

    /** @brief Checks the three levels of a pixel against expected values. */
    void expect_rgb(RgbPixel actual, RgbPixel expected)
    {
        EXPECT_EQ(actual.red, expected.red);
        EXPECT_EQ(actual.green, expected.green);
        EXPECT_EQ(actual.blue, expected.blue);
    }

    /** @brief Checks the three levels of an HSV pixel against expected values. */
    void expect_hsv(HsvPixel actual, HsvPixel expected)
    {
        EXPECT_EQ(actual.hue, expected.hue);
        EXPECT_EQ(actual.saturation, expected.saturation);
        EXPECT_EQ(actual.value, expected.value);
    }

    /** @brief Returns the distance between two levels. */
    std::int32_t level_distance(std::uint8_t a, std::uint8_t b)
    {
        return a > b ? a - b : b - a;
    }
} // namespace

TEST(Unit_ImagingHsv, PlanDrawsHueThenSaturationThenValueAndNothingMore)
{
    const HsvSettings settings;
    Random random(7);
    Random replay(7);

    const HsvPlan plan = Vision::Imaging::plan_hsv(settings, random);

    const float hue = replay.uniform(-1.0f, 1.0f) * settings.hue_gain;
    const float saturation = replay.uniform(-1.0f, 1.0f) * settings.saturation_gain;
    const float value = replay.uniform(-1.0f, 1.0f) * settings.value_gain;

    // The same float operations on the same draws give the same bits.
    EXPECT_EQ(plan.hue, hue);
    EXPECT_EQ(plan.saturation, saturation);
    EXPECT_EQ(plan.value, value);
    EXPECT_EQ(random, replay);
}

TEST(Unit_ImagingHsv, DrawnGainsStayInsideTheConfiguredRangesOverManySeeds)
{
    const HsvSettings settings;
    HsvPlan lowest{settings.hue_gain, settings.saturation_gain, settings.value_gain};
    HsvPlan highest{-settings.hue_gain, -settings.saturation_gain, -settings.value_gain};

    for (std::uint64_t seed = 0; seed < K_SEED_COUNT; ++seed)
    {
        Random random(seed);
        const HsvPlan plan = Vision::Imaging::plan_hsv(settings, random);

        ASSERT_GE(plan.hue, -settings.hue_gain) << seed;
        ASSERT_LT(plan.hue, settings.hue_gain) << seed;
        ASSERT_GE(plan.saturation, -settings.saturation_gain) << seed;
        ASSERT_LT(plan.saturation, settings.saturation_gain) << seed;
        ASSERT_GE(plan.value, -settings.value_gain) << seed;
        ASSERT_LT(plan.value, settings.value_gain) << seed;

        lowest.hue = std::min(lowest.hue, plan.hue);
        lowest.saturation = std::min(lowest.saturation, plan.saturation);
        lowest.value = std::min(lowest.value, plan.value);
        highest.hue = std::max(highest.hue, plan.hue);
        highest.saturation = std::max(highest.saturation, plan.saturation);
        highest.value = std::max(highest.value, plan.value);
    }

    // 10000 uniform draws miss the outer twentieth of a range with probability 0.95^10000.
    EXPECT_LT(lowest.hue, -0.9f * settings.hue_gain);
    EXPECT_GT(highest.hue, 0.9f * settings.hue_gain);
    EXPECT_LT(lowest.saturation, -0.9f * settings.saturation_gain);
    EXPECT_GT(highest.saturation, 0.9f * settings.saturation_gain);
    EXPECT_LT(lowest.value, -0.9f * settings.value_gain);
    EXPECT_GT(highest.value, 0.9f * settings.value_gain);
}

TEST(Unit_ImagingHsv, ZeroGainsDrawThreeTimesAndGiveAZeroPlan)
{
    HsvSettings settings;
    settings.hue_gain = 0.0f;
    settings.saturation_gain = 0.0f;
    settings.value_gain = 0.0f;

    Random random(11);
    Random replay(11);
    const HsvPlan plan = Vision::Imaging::plan_hsv(settings, random);

    EXPECT_EQ(plan.hue, 0.0f);
    EXPECT_EQ(plan.saturation, 0.0f);
    EXPECT_EQ(plan.value, 0.0f);

    (void)replay.next_u64();
    (void)replay.next_u64();
    (void)replay.next_u64();
    EXPECT_EQ(random, replay);
}

TEST(Unit_ImagingHsv, PlanRejectsANegativeGain)
{
    Random random(1);
    HsvSettings settings;
    settings.saturation_gain = -0.5f;

    SA_EXPECT_ERROR((void)Vision::Imaging::plan_hsv(settings, random), "validate_hsv", "gain");
    SA_EXPECT_ERROR(Vision::Imaging::validate_hsv(settings), "validate_hsv", "gain", "-0.5");
    EXPECT_EQ(random, Random(1)) << "a rejected plan makes no draw";
}

TEST(Unit_ImagingHsv, PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw)
{
    Random random(2);

    HsvSettings infinite;
    infinite.hue_gain = std::numeric_limits<float>::infinity();
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_hsv(infinite, random), "validate_hsv", "gain");

    HsvSettings not_a_number;
    not_a_number.value_gain = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_ERROR((void)Vision::Imaging::plan_hsv(not_a_number, random), "validate_hsv", "gain");

    EXPECT_EQ(random, Random(2));
}

TEST(Unit_ImagingHsv, SourceWithoutPixelsIsRejected)
{
    Image destination;
    SA_EXPECT_ERROR(Vision::Imaging::apply_hsv(Vision::Imaging::ImageView{}, HsvPlan{},
                                               destination),
                    "apply_hsv", "no pixels");
}

TEST(Unit_ImagingHsv, DestinationThatIsTheSourceIsRejected)
{
    Image image = make_row({RgbPixel{12, 200, 99}, RgbPixel{250, 3, 180}});
    const HsvPlan plan{0.25f, 0.5f, 0.5f};

    SA_EXPECT_ERROR(Vision::Imaging::apply_hsv(image.view(), plan, image), "apply_hsv",
                    "destination holds the source's pixels");
    expect_rgb(pixel_at(image, 0), RgbPixel{12, 200, 99});
    expect_rgb(pixel_at(image, 1), RgbPixel{250, 3, 180});
}

/** @brief Tests that a view of the destination's second pixel is refused before any write. */
TEST(Unit_ImagingHsv, ViewInsideTheDestinationIsRejected)
{
    Image image = make_row({RgbPixel{12, 200, 99}, RgbPixel{250, 3, 180}});
    const Vision::Imaging::ImageView second_pixel(image.data() + 3, ImageSize{1, 1}, 3);
    const HsvPlan plan{0.25f, 0.5f, 0.5f};

    SA_EXPECT_ERROR(Vision::Imaging::apply_hsv(second_pixel, plan, image), "apply_hsv",
                    "destination holds the source's pixels");
    EXPECT_EQ(image.size(), (ImageSize{1, 2}));
    expect_rgb(pixel_at(image, 0), RgbPixel{12, 200, 99});
    expect_rgb(pixel_at(image, 1), RgbPixel{250, 3, 180});
}

TEST(Unit_ImagingHsv, ZeroPlanGivesIdentityTables)
{
    const HsvTables tables = Vision::Imaging::make_hsv_tables(HsvPlan{});

    for (std::int32_t x = 0; x < 256; ++x)
    {
        const auto index = static_cast<std::size_t>(x);
        const std::int32_t hue = x < K_HUE_STEPS ? x : x - K_HUE_STEPS;
        ASSERT_EQ(tables.hue[index], hue) << "hue of " << x;
        ASSERT_EQ(tables.saturation[index], x) << "saturation of " << x;
        ASSERT_EQ(tables.value[index], x) << "value of " << x;
    }
}

TEST(Unit_ImagingHsv, HueTableShiftsByAFractionOfATurnAndWrapsBothWays)
{
    // 0.25 of a turn is 45 steps; 0.25f * 180 is exact.
    const HsvTables forward = Vision::Imaging::make_hsv_tables(HsvPlan{0.25f, 0.0f, 0.0f});
    EXPECT_EQ(forward.hue[0], 45);
    EXPECT_EQ(forward.hue[134], 179);
    EXPECT_EQ(forward.hue[135], 0);
    EXPECT_EQ(forward.hue[179], 44);

    const HsvTables backward = Vision::Imaging::make_hsv_tables(HsvPlan{-0.25f, 0.0f, 0.0f});
    EXPECT_EQ(backward.hue[0], 135);
    EXPECT_EQ(backward.hue[44], 179);
    EXPECT_EQ(backward.hue[45], 0);
    EXPECT_EQ(backward.hue[179], 134);

    for (std::size_t x = 0; x < 256; ++x)
    {
        ASSERT_LT(forward.hue[x], K_HUE_STEPS) << x;
        ASSERT_LT(backward.hue[x], K_HUE_STEPS) << x;
    }
}

TEST(Unit_ImagingHsv, SaturationAndValueTablesScaleTruncateAndClamp)
{
    // Saturation times 1.5 and value times 1.25.
    const HsvTables raised = Vision::Imaging::make_hsv_tables(HsvPlan{0.0f, 0.5f, 0.25f});
    EXPECT_EQ(raised.saturation[0], 0);
    EXPECT_EQ(raised.saturation[3], 4);
    EXPECT_EQ(raised.saturation[170], 255);
    EXPECT_EQ(raised.saturation[171], 255);
    EXPECT_EQ(raised.saturation[255], 255);
    EXPECT_EQ(raised.value[3], 3);
    EXPECT_EQ(raised.value[204], 255);
    EXPECT_EQ(raised.value[203], 253);
    EXPECT_EQ(raised.value[255], 255);

    // Saturation times 0.5 and value times -1, which clamps to 0.
    const HsvTables lowered = Vision::Imaging::make_hsv_tables(HsvPlan{0.0f, -0.5f, -2.0f});
    EXPECT_EQ(lowered.saturation[255], 127);
    EXPECT_EQ(lowered.saturation[1], 0);
    EXPECT_EQ(lowered.value[1], 0);
    EXPECT_EQ(lowered.value[255], 0);
}

TEST(Unit_ImagingHsv, PrimaryColoursHaveHuesSixtyStepsApart)
{
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{255, 0, 0}), HsvPixel{0, 255, 255});
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{0, 255, 0}), HsvPixel{60, 255, 255});
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{0, 0, 255}), HsvPixel{120, 255, 255});
}

TEST(Unit_ImagingHsv, GreyHasHueZeroAndBlackHasSaturationZero)
{
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{0, 0, 0}), HsvPixel{0, 0, 0});
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{90, 90, 90}), HsvPixel{0, 0, 90});
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{255, 255, 255}), HsvPixel{0, 0, 255});
}

TEST(Unit_ImagingHsv, MixedColourConvertsToTheHandComputedLevels)
{
    // (100, 50, 50): saturation (255 * 50 + 50) / 100 = 128, hue 0.
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{100, 50, 50}), HsvPixel{0, 128, 100});

    // (200, 100, 50): saturation (255 * 150 + 100) / 200 = 191, hue (3000 + 150) / 300 = 10.
    expect_hsv(Vision::Imaging::to_hsv(RgbPixel{200, 100, 50}), HsvPixel{10, 191, 200});
}

TEST(Unit_ImagingHsv, EachSectorConvertsBackToTheHandComputedColour)
{
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{0, 255, 255}), RgbPixel{255, 0, 0});
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{45, 255, 255}), RgbPixel{128, 255, 0});
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{60, 255, 255}), RgbPixel{0, 255, 0});
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{90, 255, 255}), RgbPixel{0, 255, 255});
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{120, 255, 255}), RgbPixel{0, 0, 255});
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{135, 255, 255}), RgbPixel{128, 0, 255});

    // Hue 178, offset 28 in the last sector: (255 * (7650 - 255 * 28) + 3825) / 7650 = 17.
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{178, 255, 255}), RgbPixel{255, 0, 17});

    // Hue 10, saturation 153, value 200: low 80, rising (200 * 4590 + 3825) / 7650 = 120.
    expect_rgb(Vision::Imaging::to_rgb(HsvPixel{10, 153, 200}), RgbPixel{200, 120, 80});
}

TEST(Unit_ImagingHsv, PureHueShiftOnRedGivesTheHandComputedColours)
{
    // Red is hue 0. Plus 45 steps is hue 45, minus 45 is hue 135, plus 90 is hue 90.
    expect_rgb(jitter(RgbPixel{255, 0, 0}, HsvPlan{0.25f, 0.0f, 0.0f}), RgbPixel{128, 255, 0});
    expect_rgb(jitter(RgbPixel{255, 0, 0}, HsvPlan{-0.25f, 0.0f, 0.0f}), RgbPixel{128, 0, 255});
    expect_rgb(jitter(RgbPixel{255, 0, 0}, HsvPlan{0.5f, 0.0f, 0.0f}), RgbPixel{0, 255, 255});
}

TEST(Unit_ImagingHsv, ValueGainClampsAtFullAndAtZero)
{
    // (100, 50, 50) is hue 0, saturation 128, value 100; value 300 clamps to 255.
    expect_rgb(jitter(RgbPixel{100, 50, 50}, HsvPlan{0.0f, 0.0f, 2.0f}),
               RgbPixel{255, 127, 127});
    expect_rgb(jitter(RgbPixel{100, 50, 50}, HsvPlan{0.0f, 0.0f, -2.0f}), RgbPixel{0, 0, 0});
}

TEST(Unit_ImagingHsv, SaturationGainClampsAtFullAndAtZero)
{
    // Saturation 128 times 3 clamps to 255; times -1 clamps to 0, which is grey.
    expect_rgb(jitter(RgbPixel{100, 50, 50}, HsvPlan{0.0f, 2.0f, 0.0f}), RgbPixel{100, 0, 0});
    expect_rgb(jitter(RgbPixel{100, 50, 50}, HsvPlan{0.0f, -2.0f, 0.0f}),
               RgbPixel{100, 100, 100});
}

TEST(Unit_ImagingHsv, WhiteStaysWhiteUnderASaturationGain)
{
    expect_rgb(jitter(RgbPixel{255, 255, 255}, HsvPlan{0.3f, 0.7f, 0.0f}),
               RgbPixel{255, 255, 255});
}

TEST(Unit_ImagingHsv, ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries)
{
    const Image source = make_row({RgbPixel{0, 0, 0}, RgbPixel{255, 255, 255},
                                   RgbPixel{37, 37, 37}, RgbPixel{255, 0, 0},
                                   RgbPixel{0, 255, 0}, RgbPixel{0, 0, 255},
                                   RgbPixel{255, 255, 0}, RgbPixel{0, 255, 255},
                                   RgbPixel{255, 0, 255}});

    Image destination;
    Vision::Imaging::apply_hsv(source.view(), HsvPlan{}, destination);

    ASSERT_EQ(destination.size(), source.size());
    ASSERT_EQ(destination.channels(), 3);
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        EXPECT_EQ(destination.data()[i], source.data()[i]) << "byte " << i;
}

/** @brief Tests that a zero plan copies colours the hexcone round trip would change. */
TEST(Unit_ImagingHsv, ZeroPlanCopiesColoursTheRoundTripWouldChange)
{
    // to_rgb(to_hsv(.)) gives (12, 200, 100) and (250, 3, 176) for these two colours.
    const Image source = make_row({RgbPixel{12, 200, 99}, RgbPixel{250, 3, 180}});
    ASSERT_EQ(Vision::Imaging::to_rgb(Vision::Imaging::to_hsv(RgbPixel{12, 200, 99})).blue, 100);
    ASSERT_EQ(Vision::Imaging::to_rgb(Vision::Imaging::to_hsv(RgbPixel{250, 3, 180})).blue, 176);

    Image destination;
    Vision::Imaging::apply_hsv(source.view(), HsvPlan{}, destination);

    ASSERT_EQ(destination.size(), source.size());
    ASSERT_EQ(destination.channels(), 3);
    expect_rgb(pixel_at(destination, 0), RgbPixel{12, 200, 99});
    expect_rgb(pixel_at(destination, 1), RgbPixel{250, 3, 180});
}

/** @brief Tests that a plan from three zero gains is an identity plan and copies the image. */
TEST(Unit_ImagingHsv, PlanOfZeroGainsCopiesTheImage)
{
    HsvSettings settings;
    settings.hue_gain = 0.0f;
    settings.saturation_gain = 0.0f;
    settings.value_gain = 0.0f;
    const Image source = make_row({RgbPixel{12, 200, 99}, RgbPixel{250, 3, 180}});

    for (std::uint64_t seed = 0; seed < 64; ++seed)
    {
        Random random(seed);
        const HsvPlan plan = Vision::Imaging::plan_hsv(settings, random);
        Image destination;
        Vision::Imaging::apply_hsv(source.view(), plan, destination);

        ASSERT_TRUE(Vision::Imaging::is_hsv_identity(plan)) << "seed " << seed;
        expect_rgb(pixel_at(destination, 0), RgbPixel{12, 200, 99});
        expect_rgb(pixel_at(destination, 1), RgbPixel{250, 3, 180});
    }
}

/** @brief Tests that one gain away from zero is enough to send the image through the tables. */
TEST(Unit_ImagingHsv, PlanWithOneGainAwayFromZeroIsNotAnIdentity)
{
    const float tiny = 1e-6f;

    EXPECT_TRUE(Vision::Imaging::is_hsv_identity(HsvPlan{}));
    EXPECT_TRUE(Vision::Imaging::is_hsv_identity(HsvPlan{-0.0f, 0.0f, -0.0f}));
    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HsvPlan{tiny, 0.0f, 0.0f}));
    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HsvPlan{0.0f, -tiny, 0.0f}));
    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HsvPlan{0.0f, 0.0f, tiny}));

    // A value gain of 1e-6 leaves every table entry alone, so only the round trip shows.
    expect_rgb(jitter(RgbPixel{12, 200, 99}, HsvPlan{0.0f, 0.0f, tiny}), RgbPixel{12, 200, 100});
}

/** @brief Tests that validate_hsv accepts zero gains and the defaults. */
TEST(Unit_ImagingHsv, ValidateAcceptsZeroGainsAndTheDefaults)
{
    HsvSettings zero;
    zero.hue_gain = 0.0f;
    zero.saturation_gain = 0.0f;
    zero.value_gain = 0.0f;

    EXPECT_NO_THROW(Vision::Imaging::validate_hsv(zero));
    EXPECT_NO_THROW(Vision::Imaging::validate_hsv(HsvSettings{}));
}

TEST(Unit_ImagingHsv, RoundTripMovesNoLevelByMoreThanItsBound)
{
    std::int32_t worst = 0;
    for (std::int32_t red = 0; red < 256; red += K_SWEEP_STEP)
        for (std::int32_t green = 0; green < 256; green += K_SWEEP_STEP)
            for (std::int32_t blue = 0; blue < 256; blue += K_SWEEP_STEP)
            {
                const RgbPixel pixel{static_cast<std::uint8_t>(red),
                                     static_cast<std::uint8_t>(green),
                                     static_cast<std::uint8_t>(blue)};
                const HsvPixel hsv = Vision::Imaging::to_hsv(pixel);
                ASSERT_LT(hsv.hue, K_HUE_STEPS);

                const RgbPixel back = Vision::Imaging::to_rgb(hsv);
                const std::int32_t distances[] = {level_distance(back.red, pixel.red),
                                                  level_distance(back.green, pixel.green),
                                                  level_distance(back.blue, pixel.blue)};
                for (std::int32_t distance : distances)
                    worst = distance > worst ? distance : worst;
            }

    EXPECT_LE(worst, K_ROUND_TRIP_LEVELS);
}

TEST(Unit_ImagingHsv, ImageWithoutThreeChannelsIsCopiedUnchanged)
{
    Image source(ImageSize{2, 3}, 1);
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        source.data()[i] = static_cast<std::uint8_t>(40 * i + 3);

    Image destination;
    Vision::Imaging::apply_hsv(source.view(), HsvPlan{0.25f, 0.5f, 0.5f}, destination);

    ASSERT_EQ(destination.size(), source.size());
    ASSERT_EQ(destination.channels(), 1);
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        EXPECT_EQ(destination.data()[i], source.data()[i]) << "byte " << i;
}

TEST(Unit_ImagingHsv, SamePlanGivesTheSameImageTwice)
{
    const Image source = make_row({RgbPixel{12, 200, 99}, RgbPixel{250, 3, 180},
                                   RgbPixel{77, 77, 80}});
    const HsvPlan plan{0.01f, -0.3f, 0.2f};

    Image first;
    Image second;
    Vision::Imaging::apply_hsv(source.view(), plan, first);
    Vision::Imaging::apply_hsv(source.view(), plan, second);

    for (std::size_t i = 0; i < source.byte_count(); ++i)
        EXPECT_EQ(first.data()[i], second.data()[i]) << "byte " << i;
}

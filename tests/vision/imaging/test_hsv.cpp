/**************************************************************************/
/* test_hsv.cpp                                                           */
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
 * @file test_hsv.cpp
 * @brief Tests the HSV jitter: its draws, its lookup tables, the 8-bit hexcone and the image pass.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using Vision::Imaging::HSVPixel;
using Vision::Imaging::HSVPlan;
using Vision::Imaging::HSVSettings;
using Vision::Imaging::HSVTables;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::K_HUE_STEPS;
using Vision::Imaging::Random;
using Vision::Imaging::RGBPixel;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::expect_same_image;

    /** @brief Counts the plans or colours a property test draws, all from one generator. */
    constexpr std::uint64_t K_SAMPLE_COUNT = 1024;

    /** @brief Holds the seed of the generator a property test draws from. */
    constexpr std::uint64_t K_PROPERTY_SEED = 20261007;

    /** @brief Holds the largest level error of to_hsv then to_rgb over all 2^24 colours. */
    constexpr std::int32_t K_ROUND_TRIP_LEVELS = 4;

    /** @brief Returns a one-row RGB image of the given pixels. */
    Image make_row(std::initializer_list<RGBPixel> pixels)
    {
        Image image(ImageSize{1, static_cast<std::int32_t>(pixels.size())}, 3);
        std::uint8_t* out = image.data();
        for (const RGBPixel& pixel : pixels)
        {
            *out++ = pixel.red;
            *out++ = pixel.green;
            *out++ = pixel.blue;
        }

        return image;
    }

    /** @brief Returns an image of the given size and channels holding the listed bytes in order. */
    Image make_bytes_image(ImageSize size, std::int32_t channels,
                           std::initializer_list<std::uint8_t> bytes)
    {
        Image image(size, channels);
        std::copy(bytes.begin(), bytes.end(), image.data());
        return image;
    }

    /** @brief Returns the pixel at a column of a one-row RGB image. */
    RGBPixel pixel_at(const Image& image, std::int32_t x)
    {
        const std::uint8_t* in = image.row(0) + x * 3;
        return RGBPixel{in[0], in[1], in[2]};
    }

    /** @brief Returns the result of jittering a single pixel with a plan. */
    RGBPixel jitter(RGBPixel pixel, const HSVPlan& plan)
    {
        const Image source = make_row({pixel});
        Image destination;
        Vision::Imaging::apply_hsv(source.view(), plan, destination);
        return pixel_at(destination, 0);
    }

    /** @brief Checks the three levels of a pixel against expected values. */
    void expect_rgb(RGBPixel actual, RGBPixel expected)
    {
        EXPECT_EQ(actual.red, expected.red);
        EXPECT_EQ(actual.green, expected.green);
        EXPECT_EQ(actual.blue, expected.blue);
    }

    /** @brief Checks the three levels of an HSV pixel against expected values. */
    void expect_hsv(HSVPixel actual, HSVPixel expected)
    {
        EXPECT_EQ(actual.hue, expected.hue);
        EXPECT_EQ(actual.saturation, expected.saturation);
        EXPECT_EQ(actual.value, expected.value);
    }

    /** @brief Checks that the tables map every level to itself, and every hue to its hue step. */
    void expect_identity_tables(const HSVTables& tables)
    {
        for (std::int32_t x = 0; x < 256; ++x)
        {
            const auto index = static_cast<std::size_t>(x);
            const std::int32_t hue = x < K_HUE_STEPS ? x : x - K_HUE_STEPS;
            ASSERT_EQ(tables.hue[index], hue) << "hue of " << x;
            ASSERT_EQ(tables.saturation[index], x) << "saturation of " << x;
            ASSERT_EQ(tables.value[index], x) << "value of " << x;
        }
    }

    /** @brief Checks that every entry of a hue table is below K_HUE_STEPS. */
    void expect_hue_entries_below_the_steps(const std::array<std::uint8_t, 256>& hue)
    {
        for (std::size_t x = 0; x < hue.size(); ++x)
            ASSERT_LT(hue[x], K_HUE_STEPS) << x;
    }

    /** @brief Returns the distance between two levels. */
    std::int32_t level_distance(std::uint8_t a, std::uint8_t b)
    {
        return a > b ? a - b : b - a;
    }

    /** @brief Returns the largest level error of to_hsv then to_rgb on one colour. */
    std::int32_t round_trip_error(RGBPixel pixel)
    {
        const HSVPixel hsv = Vision::Imaging::to_hsv(pixel);
        const RGBPixel back = Vision::Imaging::to_rgb(hsv);
        return std::max({level_distance(back.red, pixel.red),
                         level_distance(back.green, pixel.green),
                         level_distance(back.blue, pixel.blue)});
    }

    /** @brief Returns settings whose three gains are zero. */
    HSVSettings zero_gain_settings()
    {
        HSVSettings settings;
        settings.hue_gain = 0.0f;
        settings.saturation_gain = 0.0f;
        settings.value_gain = 0.0f;
        return settings;
    }

    /** @brief Returns the plan that zero gains give with a generator of the given seed. */
    HSVPlan zero_gain_plan(std::uint64_t seed)
    {
        Random random(seed);
        return require_value(Vision::Imaging::plan_hsv(zero_gain_settings(), random));
    }

    /** @brief Checks that an identity plan copies two colours the round trip changes. */
    void expect_identity_plan_copies_colours(const HSVPlan& plan)
    {
        const Image source = make_row({RGBPixel{12, 200, 99}, RGBPixel{250, 3, 180}});
        Image destination;

        Vision::Imaging::apply_hsv(source.view(), plan, destination);

        ASSERT_TRUE(Vision::Imaging::is_hsv_identity(plan));
        ASSERT_EQ(destination.size(), source.size());
        ASSERT_EQ(destination.channels(), 3);
        expect_rgb(pixel_at(destination, 0), RGBPixel{12, 200, 99});
        expect_rgb(pixel_at(destination, 1), RGBPixel{250, 3, 180});
    }
} // namespace

/** @brief Tests that the plan values are the hue, saturation and value draws, in that order. */
TEST(Small_ImagingHsv, PlanValuesAreTheHueThenSaturationThenValueDraws)
{
    const HSVSettings settings;
    Random random(7);
    Random replay(7);

    const HSVPlan plan = require_value(Vision::Imaging::plan_hsv(settings, random));

    const float hue = replay.uniform(-1.0f, 1.0f) * settings.hue_gain;
    const float saturation = replay.uniform(-1.0f, 1.0f) * settings.saturation_gain;
    const float value = replay.uniform(-1.0f, 1.0f) * settings.value_gain;

    // The same float operations on the same draws give the same bits.
    EXPECT_EQ(plan.hue, hue);
    EXPECT_EQ(plan.saturation, saturation);
    EXPECT_EQ(plan.value, value);
}

/** @brief Tests that the plan makes no draw beyond the three of hue, saturation and value. */
TEST(Small_ImagingHsv, PlanMakesNoDrawBeyondTheThird)
{
    const HSVSettings settings;
    Random random(7);
    Random replay(7);

    static_cast<void>(require_value(Vision::Imaging::plan_hsv(settings, random)));

    (void)replay.uniform(-1.0f, 1.0f);
    (void)replay.uniform(-1.0f, 1.0f);
    (void)replay.uniform(-1.0f, 1.0f);
    EXPECT_EQ(random, replay);
}

/** @brief Tests that drawn gains stay inside their ranges and reach the outer twentieth of each. */
TEST(Small_ImagingHsv, DrawnGainsStayInsideTheConfiguredRanges)
{
    const HSVSettings settings;
    Random random(K_PROPERTY_SEED);
    HSVPlan lowest{settings.hue_gain, settings.saturation_gain, settings.value_gain};
    HSVPlan highest{-settings.hue_gain, -settings.saturation_gain, -settings.value_gain};

    for (std::uint64_t sample = 0; sample < K_SAMPLE_COUNT; ++sample)
    {
        const HSVPlan plan = require_value(Vision::Imaging::plan_hsv(settings, random));

        ASSERT_GE(plan.hue, -settings.hue_gain) << sample;
        ASSERT_LT(plan.hue, settings.hue_gain) << sample;
        ASSERT_GE(plan.saturation, -settings.saturation_gain) << sample;
        ASSERT_LT(plan.saturation, settings.saturation_gain) << sample;
        ASSERT_GE(plan.value, -settings.value_gain) << sample;
        ASSERT_LT(plan.value, settings.value_gain) << sample;

        lowest.hue = std::min(lowest.hue, plan.hue);
        lowest.saturation = std::min(lowest.saturation, plan.saturation);
        lowest.value = std::min(lowest.value, plan.value);
        highest.hue = std::max(highest.hue, plan.hue);
        highest.saturation = std::max(highest.saturation, plan.saturation);
        highest.value = std::max(highest.value, plan.value);
    }

    // 1024 uniform draws miss the outer twentieth of a range with probability 0.95^1024, 1.6e-23.
    EXPECT_LT(lowest.hue, -0.9f * settings.hue_gain);
    EXPECT_GT(highest.hue, 0.9f * settings.hue_gain);
    EXPECT_LT(lowest.saturation, -0.9f * settings.saturation_gain);
    EXPECT_GT(highest.saturation, 0.9f * settings.saturation_gain);
    EXPECT_LT(lowest.value, -0.9f * settings.value_gain);
    EXPECT_GT(highest.value, 0.9f * settings.value_gain);
}

/** @brief Tests that three zero gains give a zero plan. */
TEST(Small_ImagingHsv, ZeroGainsGiveAZeroPlan)
{
    Random random(11);

    const HSVPlan plan = require_value(Vision::Imaging::plan_hsv(zero_gain_settings(), random));

    EXPECT_EQ(plan.hue, 0.0f);
    EXPECT_EQ(plan.saturation, 0.0f);
    EXPECT_EQ(plan.value, 0.0f);
}

/** @brief Tests that a plan from zero gains still makes its three draws. */
TEST(Small_ImagingHsv, ZeroGainsStillDrawThreeTimes)
{
    Random random(11);
    Random replay(11);

    static_cast<void>(require_value(Vision::Imaging::plan_hsv(zero_gain_settings(), random)));

    (void)replay.next_u64();
    (void)replay.next_u64();
    (void)replay.next_u64();
    EXPECT_EQ(random, replay);
}

/** @brief Tests that a negative gain is refused by the plan and by validate_hsv. */
TEST(Small_ImagingHsv, PlanRejectsANegativeGain)
{
    Random random(1);
    HSVSettings settings;
    settings.saturation_gain = -0.5f;

    SA_EXPECT_FAILURE(Vision::Imaging::plan_hsv(settings, random), StatusCode::InvalidDescription,
                      "validate_hsv", "gain");
    SA_EXPECT_FAILURE(Vision::Imaging::validate_hsv(settings), StatusCode::InvalidDescription,
                      "validate_hsv", "gain", "-0.5");
}

/** @brief Tests that a plan refused for a negative gain makes no draw. */
TEST(Small_ImagingHsv, PlanRejectedForANegativeGainMakesNoDraw)
{
    Random random(1);
    HSVSettings settings;
    settings.saturation_gain = -0.5f;

    static_cast<void>(Vision::Imaging::plan_hsv(settings, random));

    EXPECT_EQ(random, Random(1));
}

/** @brief Tests that a gain that is infinite or NaN is refused before any draw. */
TEST(Small_ImagingHsv, PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw)
{
    Random random(2);

    HSVSettings infinite;
    infinite.hue_gain = std::numeric_limits<float>::infinity();
    SA_EXPECT_FAILURE(Vision::Imaging::plan_hsv(infinite, random), StatusCode::InvalidDescription,
                      "validate_hsv", "gain");

    HSVSettings not_a_number;
    not_a_number.value_gain = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_FAILURE(Vision::Imaging::plan_hsv(not_a_number, random),
                      StatusCode::InvalidDescription, "validate_hsv", "gain");

    EXPECT_EQ(random, Random(2));
}

/** @brief Tests that applying a jitter to a source without pixels stops the process. */
TEST(Medium_ImagingHsv, SourceWithoutPixelsIsRejected)
{
    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_hsv(Vision::Imaging::ImageView{},
                                                                 HSVPlan{}, destination)),
                    "apply_hsv", "no pixels");
}

/** @brief Tests that a source held by the destination, whole or as a pixel, is refused. */
TEST(Medium_ImagingHsv, SourceHeldByTheDestinationIsRejected)
{
    Image whole = make_row({RGBPixel{12, 200, 99}, RGBPixel{250, 3, 180}});
    Image pixels = make_row({RGBPixel{12, 200, 99}, RGBPixel{250, 3, 180}});
    const Vision::Imaging::ImageView second_pixel(pixels.data() + 3, ImageSize{1, 1}, 3);
    const HSVPlan plan{0.25f, 0.5f, 0.5f};

    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_hsv(whole.view(), plan, whole)),
                    "apply_hsv", "destination holds the source's pixels");
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_hsv(second_pixel, plan, pixels)),
                    "apply_hsv", "destination holds the source's pixels");
}

/** @brief Tests that a zero plan gives tables that map each level to itself. */
TEST(Small_ImagingHsv, ZeroPlanGivesIdentityTables)
{
    expect_identity_tables(Vision::Imaging::make_hsv_tables(HSVPlan{}));
}

/** @brief Tests that a forward hue gain shifts the hue table by a fraction of a turn. */
TEST(Small_ImagingHsv, HueTableShiftsForwardByAQuarterTurn)
{
    // 0.25 of a turn is 45 steps; 0.25f * 180 is exact.
    const HSVTables forward = Vision::Imaging::make_hsv_tables(HSVPlan{0.25f, 0.0f, 0.0f});

    EXPECT_EQ(forward.hue[0], 45);
    EXPECT_EQ(forward.hue[134], 179);
    EXPECT_EQ(forward.hue[135], 0);
    EXPECT_EQ(forward.hue[179], 44);
    expect_hue_entries_below_the_steps(forward.hue);
}

/** @brief Tests that a backward hue gain shifts the hue table back and wraps below zero. */
TEST(Small_ImagingHsv, HueTableShiftsBackwardAndWraps)
{
    const HSVTables backward = Vision::Imaging::make_hsv_tables(HSVPlan{-0.25f, 0.0f, 0.0f});

    EXPECT_EQ(backward.hue[0], 135);
    EXPECT_EQ(backward.hue[44], 179);
    EXPECT_EQ(backward.hue[45], 0);
    EXPECT_EQ(backward.hue[179], 134);
    expect_hue_entries_below_the_steps(backward.hue);
}

/** @brief Tests that raised gains scale the saturation and value tables, truncate and clamp. */
TEST(Small_ImagingHsv, RaisedSaturationAndValueTablesScaleTruncateAndClamp)
{
    // Saturation times 1.5 and value times 1.25.
    const HSVTables raised = Vision::Imaging::make_hsv_tables(HSVPlan{0.0f, 0.5f, 0.25f});

    EXPECT_EQ(raised.saturation[0], 0);
    EXPECT_EQ(raised.saturation[3], 4);
    EXPECT_EQ(raised.saturation[170], 255);
    EXPECT_EQ(raised.saturation[171], 255);
    EXPECT_EQ(raised.saturation[255], 255);
    EXPECT_EQ(raised.value[3], 3);
    EXPECT_EQ(raised.value[204], 255);
    EXPECT_EQ(raised.value[203], 253);
    EXPECT_EQ(raised.value[255], 255);
}

/** @brief Tests that lowered gains scale the saturation table down and clamp the value at zero. */
TEST(Small_ImagingHsv, LoweredSaturationAndValueTablesClampAtZero)
{
    // Saturation times 0.5 and value times -1, which clamps to 0.
    const HSVTables lowered = Vision::Imaging::make_hsv_tables(HSVPlan{0.0f, -0.5f, -2.0f});

    EXPECT_EQ(lowered.saturation[255], 127);
    EXPECT_EQ(lowered.saturation[1], 0);
    EXPECT_EQ(lowered.value[1], 0);
    EXPECT_EQ(lowered.value[255], 0);
}

/** @brief Tests that the primary colours have hues sixty steps apart. */
TEST(Small_ImagingHsv, PrimaryColoursHaveHuesSixtyStepsApart)
{
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{255, 0, 0}), HSVPixel{0, 255, 255});
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{0, 255, 0}), HSVPixel{60, 255, 255});
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{0, 0, 255}), HSVPixel{120, 255, 255});
}

/** @brief Tests that a grey level and white have hue zero and saturation zero. */
TEST(Small_ImagingHsv, GreyHasHueZero)
{
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{90, 90, 90}), HSVPixel{0, 0, 90});
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{255, 255, 255}), HSVPixel{0, 0, 255});
}

/** @brief Tests that black has saturation zero and value zero. */
TEST(Small_ImagingHsv, BlackHasSaturationZero)
{
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{0, 0, 0}), HSVPixel{0, 0, 0});
}

/** @brief Tests that the hexcone round trip changes the blue of two colours. */
TEST(Small_ImagingHsv, HexconeRoundTripChangesTheBlueOfTwoColours)
{
    EXPECT_EQ(Vision::Imaging::to_rgb(Vision::Imaging::to_hsv(RGBPixel{12, 200, 99})).blue, 100);
    EXPECT_EQ(Vision::Imaging::to_rgb(Vision::Imaging::to_hsv(RGBPixel{250, 3, 180})).blue, 176);
}

/** @brief Tests that a mixed colour converts to the hand-computed levels. */
TEST(Small_ImagingHsv, MixedColourConvertsToTheHandComputedLevels)
{
    // (100, 50, 50): saturation (255 * 50 + 50) / 100 = 128, hue 0.
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{100, 50, 50}), HSVPixel{0, 128, 100});

    // (200, 100, 50): saturation (255 * 150 + 100) / 200 = 191, hue (3000 + 150) / 300 = 10.
    expect_hsv(Vision::Imaging::to_hsv(RGBPixel{200, 100, 50}), HSVPixel{10, 191, 200});
}

/** @brief Tests that each sector of the hexcone converts back to its hand-computed colour. */
TEST(Small_ImagingHsv, EachSectorConvertsBackToTheHandComputedColour)
{
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{0, 255, 255}), RGBPixel{255, 0, 0});
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{45, 255, 255}), RGBPixel{128, 255, 0});
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{60, 255, 255}), RGBPixel{0, 255, 0});
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{90, 255, 255}), RGBPixel{0, 255, 255});
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{120, 255, 255}), RGBPixel{0, 0, 255});
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{135, 255, 255}), RGBPixel{128, 0, 255});

    // Hue 178, offset 28 in the last sector: (255 * (7650 - 255 * 28) + 3825) / 7650 = 17.
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{178, 255, 255}), RGBPixel{255, 0, 17});

    // Hue 10, saturation 153, value 200: low 80, rising (200 * 4590 + 3825) / 7650 = 120.
    expect_rgb(Vision::Imaging::to_rgb(HSVPixel{10, 153, 200}), RGBPixel{200, 120, 80});
}

/** @brief Tests that a pure hue shift on red gives the hand-computed colours. */
TEST(Small_ImagingHsv, PureHueShiftOnRedGivesTheHandComputedColours)
{
    // Red is hue 0. Plus 45 steps is hue 45, minus 45 is hue 135, plus 90 is hue 90.
    expect_rgb(jitter(RGBPixel{255, 0, 0}, HSVPlan{0.25f, 0.0f, 0.0f}), RGBPixel{128, 255, 0});
    expect_rgb(jitter(RGBPixel{255, 0, 0}, HSVPlan{-0.25f, 0.0f, 0.0f}), RGBPixel{128, 0, 255});
    expect_rgb(jitter(RGBPixel{255, 0, 0}, HSVPlan{0.5f, 0.0f, 0.0f}), RGBPixel{0, 255, 255});
}

/** @brief Tests that a value gain that raises the value clamps it at full. */
TEST(Small_ImagingHsv, ValueGainClampsAtFull)
{
    // (100, 50, 50) is hue 0, saturation 128, value 100; value 300 clamps to 255.
    expect_rgb(jitter(RGBPixel{100, 50, 50}, HSVPlan{0.0f, 0.0f, 2.0f}), RGBPixel{255, 127, 127});
}

/** @brief Tests that a value gain that lowers the value clamps it at zero. */
TEST(Small_ImagingHsv, ValueGainClampsAtZero)
{
    expect_rgb(jitter(RGBPixel{100, 50, 50}, HSVPlan{0.0f, 0.0f, -2.0f}), RGBPixel{0, 0, 0});
}

/** @brief Tests that a saturation gain that raises the saturation clamps it at full. */
TEST(Small_ImagingHsv, SaturationGainClampsAtFull)
{
    // Saturation 128 times 3 clamps to 255.
    expect_rgb(jitter(RGBPixel{100, 50, 50}, HSVPlan{0.0f, 2.0f, 0.0f}), RGBPixel{100, 0, 0});
}

/** @brief Tests that a saturation gain that lowers the saturation clamps it at zero, a grey. */
TEST(Small_ImagingHsv, SaturationGainClampsAtZero)
{
    // Saturation 128 times -1 clamps to 0, which is grey.
    expect_rgb(jitter(RGBPixel{100, 50, 50}, HSVPlan{0.0f, -2.0f, 0.0f}),
               RGBPixel{100, 100, 100});
}

/** @brief Tests that white stays white under a saturation gain and a hue gain. */
TEST(Small_ImagingHsv, WhiteStaysWhiteUnderASaturationGain)
{
    expect_rgb(jitter(RGBPixel{255, 255, 255}, HSVPlan{0.3f, 0.7f, 0.0f}),
               RGBPixel{255, 255, 255});
}

/** @brief Tests that a zero plan leaves greys, primaries and secondaries unchanged. */
TEST(Small_ImagingHsv, ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries)
{
    const Image source =
        make_row({RGBPixel{0, 0, 0}, RGBPixel{255, 255, 255}, RGBPixel{37, 37, 37},
                  RGBPixel{255, 0, 0}, RGBPixel{0, 255, 0}, RGBPixel{0, 0, 255},
                  RGBPixel{255, 255, 0}, RGBPixel{0, 255, 255}, RGBPixel{255, 0, 255}});

    Image destination;
    Vision::Imaging::apply_hsv(source.view(), HSVPlan{}, destination);

    ASSERT_EQ(destination.channels(), 3);
    expect_same_image(destination, source);
}

/** @brief Tests that a zero plan, made directly or from zero gains, copies the colours. */
TEST(Small_ImagingHsv, ZeroPlanCopiesColoursTheRoundTripWouldChange)
{
    expect_identity_plan_copies_colours(HSVPlan{});
    expect_identity_plan_copies_colours(zero_gain_plan(0));
    expect_identity_plan_copies_colours(zero_gain_plan(1));
    expect_identity_plan_copies_colours(zero_gain_plan(63));
}

/** @brief Tests that the identity predicate holds for zero gains and for signed zero gains. */
TEST(Small_ImagingHsv, IdentityPredicateHoldsForZeroAndSignedZeroGains)
{
    EXPECT_TRUE(Vision::Imaging::is_hsv_identity(HSVPlan{}));
    EXPECT_TRUE(Vision::Imaging::is_hsv_identity(HSVPlan{-0.0f, 0.0f, -0.0f}));
}

/** @brief Tests that one gain away from zero is enough to send the image through the tables. */
TEST(Small_ImagingHsv, PlanWithOneGainAwayFromZeroIsNotAnIdentity)
{
    const float tiny = 1e-6f;

    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HSVPlan{tiny, 0.0f, 0.0f}));
    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HSVPlan{0.0f, -tiny, 0.0f}));
    EXPECT_FALSE(Vision::Imaging::is_hsv_identity(HSVPlan{0.0f, 0.0f, tiny}));

    // A value gain of 1e-6 leaves every table entry alone, so only the round trip shows.
    expect_rgb(jitter(RGBPixel{12, 200, 99}, HSVPlan{0.0f, 0.0f, tiny}), RGBPixel{12, 200, 100});
}

/** @brief Tests that validate_hsv accepts zero gains and the defaults. */
TEST(Small_ImagingHsv, ValidateAcceptsZeroGainsAndTheDefaults)
{
    SA_EXPECT_OK(Vision::Imaging::validate_hsv(zero_gain_settings()));
    SA_EXPECT_OK(Vision::Imaging::validate_hsv(HSVSettings{}));
}

/** @brief Tests that to_hsv then to_rgb moves no level by more than its bound, on drawn colours. */
TEST(Small_ImagingHsv, RoundTripMovesNoLevelByMoreThanItsBound)
{
    Random random(K_PROPERTY_SEED);
    std::int32_t worst = 0;

    for (std::uint64_t sample = 0; sample < K_SAMPLE_COUNT; ++sample)
    {
        const RGBPixel pixel{static_cast<std::uint8_t>(random.index(256)),
                             static_cast<std::uint8_t>(random.index(256)),
                             static_cast<std::uint8_t>(random.index(256))};
        ASSERT_LT(Vision::Imaging::to_hsv(pixel).hue, K_HUE_STEPS) << sample;
        worst = std::max(worst, round_trip_error(pixel));
    }

    EXPECT_LE(worst, K_ROUND_TRIP_LEVELS);
}

/** @brief Tests that an image without three channels is copied unchanged. */
TEST(Small_ImagingHsv, ImageWithoutThreeChannelsIsCopiedUnchanged)
{
    const Image source = make_bytes_image(ImageSize{2, 3}, 1, {3, 43, 83, 123, 163, 203});

    Image destination;
    Vision::Imaging::apply_hsv(source.view(), HSVPlan{0.25f, 0.5f, 0.5f}, destination);

    ASSERT_EQ(destination.channels(), 1);
    expect_same_image(destination, source);
}

/** @brief Tests that the same plan gives the same image twice. */
TEST(Small_ImagingHsv, SamePlanGivesTheSameImageTwice)
{
    const Image source =
        make_row({RGBPixel{12, 200, 99}, RGBPixel{250, 3, 180}, RGBPixel{77, 77, 80}});
    const HSVPlan plan{0.01f, -0.3f, 0.2f};

    Image first;
    Image second;
    Vision::Imaging::apply_hsv(source.view(), plan, first);
    Vision::Imaging::apply_hsv(source.view(), plan, second);

    expect_same_image(first, second);
}

/**************************************************************************/
/* test_darknet_augment.cpp                                               */
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
 * @file test_darknet_augment.cpp
 * @brief Tests the crop jitter and the colour distortion of Darknet's training.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/darknet_augment.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/random.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using Vision::Imaging::DarknetAugmentPlan;
using Vision::Imaging::DarknetAugmentSettings;
using Vision::Imaging::FitTransform;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::PixelBox;
using Vision::Imaging::Random;

namespace
{
    using SushiAI::Test::expect_bytes;

    /** @brief Holds the seed of the draws below. */
    constexpr std::uint64_t K_SEED = 20261115;

    /** @brief Holds the number of seeds the range cases draw. */
    constexpr std::uint64_t K_DRAWS = 400;

    /** @brief Returns a one-channel image holding the listed levels, row by row. */
    Image make_gray(ImageSize size, std::initializer_list<std::uint8_t> levels)
    {
        Image image(size, 1);
        std::size_t offset = 0;
        for (const std::uint8_t level : levels)
            image.data()[offset++] = level;

        return image;
    }

    /** @brief Returns the settings of a .cfg of jitter .3, hue .1, saturation 1.5, exposure 2. */
    DarknetAugmentSettings augmenting()
    {
        DarknetAugmentSettings settings;
        settings.jitter = 0.3f;
        settings.hue = 0.1f;
        settings.saturation = 1.5f;
        settings.exposure = 2.0f;
        return settings;
    }

    /** @brief Returns a plan that crops all of a source of the given size and distorts nothing. */
    DarknetAugmentPlan whole_image_plan(ImageSize size)
    {
        DarknetAugmentPlan plan;
        plan.source = size;
        plan.crop_width = size.width;
        plan.crop_height = size.height;
        return plan;
    }

    /** @brief Draws rand_scale of Darknet from @p random: a scale in [1, bound], maybe inverted. */
    float host_scale(float bound, Random& random)
    {
        const float scale = random.uniform(1.0f, bound);
        return random.bernoulli(0.5f) ? scale : 1.0f / scale;
    }
} // namespace

/** @brief Tests the nine draws of a plan against the draws of a copy of the generator. */
TEST(Small_ImagingDarknetAugment, ThePlanIsTheNineDrawsOfDarknetsCode)
{
    const ImageSize size{20, 30};
    const DarknetAugmentSettings settings = augmenting();
    Random random(K_SEED);
    Random host = random;

    SA_ASSERT_OK_AND_ASSIGN(const DarknetAugmentPlan plan,
                            Vision::Imaging::plan_darknet_augment(settings, size, random));

    // Darknet draws the left, right, top and bottom shifts, then the hue, then the two scales.
    const float reach_x = 30.0f * settings.jitter;
    const float reach_y = 20.0f * settings.jitter;
    const float left = host.uniform(-reach_x, reach_x);
    const float right = host.uniform(-reach_x, reach_x);
    const float top = host.uniform(-reach_y, reach_y);
    const float bottom = host.uniform(-reach_y, reach_y);
    const float hue = host.uniform(-settings.hue, settings.hue);
    const float saturation = host_scale(settings.saturation, host);
    const float exposure = host_scale(settings.exposure, host);

    EXPECT_EQ(plan.source, size);
    EXPECT_EQ(plan.shift_x, left);
    EXPECT_EQ(plan.shift_y, top);
    EXPECT_EQ(plan.left, static_cast<std::int32_t>(left));
    EXPECT_EQ(plan.top, static_cast<std::int32_t>(top));
    EXPECT_EQ(plan.crop_width, static_cast<std::int32_t>(30.0f - left - right));
    EXPECT_EQ(plan.crop_height, static_cast<std::int32_t>(20.0f - top - bottom));
    EXPECT_EQ(plan.color.hue, hue);
    EXPECT_EQ(plan.color.saturation, saturation - 1.0f);
    EXPECT_EQ(plan.color.value, exposure - 1.0f);
    EXPECT_EQ(random, host);
}

/** @brief Tests that every crop edge, the hue and the two scales stay in their ranges. */
TEST(Small_ImagingDarknetAugment, EveryDrawStaysInItsRangeAndReachesBothSides)
{
    const ImageSize size{20, 30};
    const DarknetAugmentSettings settings = augmenting();
    float lowest_shift = 0.0f;
    float highest_shift = 0.0f;
    float lowest_hue = 0.0f;
    float highest_hue = 0.0f;
    bool has_reduction = false;
    bool has_gain = false;

    for (std::uint64_t seed = 0; seed < K_DRAWS; ++seed)
    {
        Random random(seed);
        SA_ASSERT_OK_AND_ASSIGN(const DarknetAugmentPlan plan,
                                Vision::Imaging::plan_darknet_augment(settings, size, random));

        // A shift is in [-jitter * side, jitter * side), so the crop is between 0.4 and 1.6 sides.
        EXPECT_GE(plan.shift_x, -9.0f);
        EXPECT_LT(plan.shift_x, 9.0f);
        EXPECT_GE(plan.shift_y, -6.0f);
        EXPECT_LT(plan.shift_y, 6.0f);
        EXPECT_GE(plan.crop_width, 12);
        EXPECT_LE(plan.crop_width, 48);
        EXPECT_GE(plan.crop_height, 8);
        EXPECT_LE(plan.crop_height, 32);

        EXPECT_GE(plan.color.hue, -settings.hue);
        EXPECT_LT(plan.color.hue, settings.hue);

        // A factor is in [1 / s, s]; the plan holds it less one, rounded in float.
        const float saturation = plan.color.saturation + 1.0f;
        const float exposure = plan.color.value + 1.0f;
        EXPECT_GE(saturation, 1.0f / settings.saturation - 1e-6f);
        EXPECT_LE(saturation, settings.saturation);
        EXPECT_GE(exposure, 1.0f / settings.exposure - 1e-6f);
        EXPECT_LE(exposure, settings.exposure);

        lowest_shift = std::fmin(lowest_shift, plan.shift_x);
        highest_shift = std::fmax(highest_shift, plan.shift_x);
        lowest_hue = std::fmin(lowest_hue, plan.color.hue);
        highest_hue = std::fmax(highest_hue, plan.color.hue);
        has_reduction = has_reduction || exposure < 1.0f;
        has_gain = has_gain || exposure > 1.0f;
    }

    // Four hundred draws of a uniform range leave neither end of it empty.
    EXPECT_LT(lowest_shift, -4.5f);
    EXPECT_GT(highest_shift, 4.5f);
    EXPECT_LT(lowest_hue, -0.05f);
    EXPECT_GT(highest_hue, 0.05f);
    EXPECT_TRUE(has_reduction);
    EXPECT_TRUE(has_gain);
}

/** @brief Tests that settings of no change still make the nine draws and crop all of the image. */
TEST(Small_ImagingDarknetAugment, TheIdentitySettingsCropAllAndKeepTheColour)
{
    const ImageSize size{20, 30};
    Random random(K_SEED);
    Random host = random;

    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetAugmentPlan plan,
        Vision::Imaging::plan_darknet_augment(DarknetAugmentSettings{}, size, random));
    for (int draw = 0; draw < 9; ++draw)
        static_cast<void>(host.next_u64());

    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.top, 0);
    EXPECT_EQ(plan.crop_width, 30);
    EXPECT_EQ(plan.crop_height, 20);
    EXPECT_EQ(plan.color.hue, 0.0f);
    EXPECT_EQ(plan.color.saturation, 0.0f);
    EXPECT_EQ(plan.color.value, 0.0f);
    EXPECT_EQ(random, host);
}

/** @brief Tests the settings the plan refuses, before it draws. */
TEST(Small_ImagingDarknetAugment, SettingsItCannotDrawAreRefusedBeforeAnyDraw)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Random random(K_SEED);
    const Random untouched = random;
    const ImageSize size{8, 8};

    DarknetAugmentSettings settings;
    settings.jitter = 0.5f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the jitter is 0.5", "[0, 0.5)");
    settings.jitter = -0.1f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the jitter is -0.1");
    settings.jitter = nan;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the jitter is nan");

    settings = DarknetAugmentSettings{};
    settings.hue = -0.1f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the hue is -0.1");

    settings = DarknetAugmentSettings{};
    settings.saturation = 0.9f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the saturation is 0.9");

    settings = DarknetAugmentSettings{};
    settings.exposure = inf;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(settings, size, random),
                      StatusCode::InvalidDescription, "the exposure inf");

    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_augment(DarknetAugmentSettings{},
                                                            ImageSize{0, 8}, random),
                      StatusCode::InvalidDescription, "a source size of 0x8");

    EXPECT_EQ(random, untouched);
}

/** @brief Tests that a crop that leaves the image repeats its edge pixels, as Darknet's does. */
TEST(Small_ImagingDarknetAugment, ACropBeyondTheEdgeRepeatsTheEdgePixels)
{
    // A crop of columns -1 to 2 of a 2 by 3 image, as wide as four columns.
    const Image source = make_gray(ImageSize{2, 3}, {10, 20, 30, 40, 50, 60});
    DarknetAugmentPlan plan = whole_image_plan(source.size());
    plan.shift_x = -1.0f;
    plan.left = -1;
    plan.crop_width = 4;

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::apply_darknet_augment(source.view(), plan, ImageSize{2, 4},
                                                        destination));

    expect_bytes(destination, {10, 10, 20, 30, 40, 40, 50, 60});
}

/** @brief Tests that the crop is resized to the input with Darknet's kernel. */
TEST(Small_ImagingDarknetAugment, TheCropIsResizedWithDarknetsKernel)
{
    // Columns 1 and 2 hold 20, 30 and 50, 60; a 3 by 3 result reads them at 0, 0.5 and 1 on both axes.
    const Image source = make_gray(ImageSize{2, 3}, {10, 20, 30, 40, 50, 60});
    DarknetAugmentPlan plan = whole_image_plan(source.size());
    plan.shift_x = 1.0f;
    plan.left = 1;
    plan.crop_width = 2;

    Image destination;
    SA_EXPECT_OK(Vision::Imaging::apply_darknet_augment(source.view(), plan, ImageSize{3, 3},
                                                        destination));

    expect_bytes(destination, {20, 25, 30, 35, 40, 45, 50, 55, 60});
}

/** @brief Tests that the colour of the plan scales the value of a colour image and clamps it. */
TEST(Small_ImagingDarknetAugment, TheColourScalesTheValueAndClamps)
{
    // A grey pixel has no saturation, so its value is the one level: a gain of 1 doubles it.
    const Image source = SushiAI::Test::make_constant(ImageSize{1, 1}, 60, 60, 60);
    DarknetAugmentPlan plan = whole_image_plan(source.size());
    plan.color.value = 1.0f;

    Image doubled;
    SA_EXPECT_OK(Vision::Imaging::apply_darknet_augment(source.view(), plan, ImageSize{1, 1},
                                                        doubled));
    expect_bytes(doubled, {120, 120, 120});

    const Image bright = SushiAI::Test::make_constant(ImageSize{1, 1}, 200, 200, 200);
    Image clamped;
    SA_EXPECT_OK(Vision::Imaging::apply_darknet_augment(bright.view(), plan, ImageSize{1, 1},
                                                        clamped));
    expect_bytes(clamped, {255, 255, 255});
}

/** @brief Tests that a plan made for another size, and a target of no pixel, are refused. */
TEST(Small_ImagingDarknetAugment, APlanOfAnotherSizeAndAnEmptyTargetAreRefused)
{
    const Image source = make_gray(ImageSize{2, 3}, {1, 2, 3, 4, 5, 6});
    Image destination;

    SA_EXPECT_FAILURE(Vision::Imaging::apply_darknet_augment(
                          source.view(), whole_image_plan(ImageSize{3, 3}), ImageSize{2, 2},
                          destination),
                      StatusCode::InvalidDescription, "made for a 3x3 source", "2x3");
    SA_EXPECT_FAILURE(Vision::Imaging::apply_darknet_augment(
                          source.view(), whole_image_plan(source.size()), ImageSize{0, 2},
                          destination),
                      StatusCode::InvalidDescription, "a target size of 0x2");
}

/** @brief Tests that a box moves by the unrounded shift and the scale of its axis. */
TEST(Small_ImagingDarknetAugment, ABoxMovesByTheShiftAndTheScaleOfItsAxis)
{
    // correct_boxes: (x - shift) * target / crop, here (x + 1.5) * 8 / 4 and (y - 0.5) * 6 / 2.
    DarknetAugmentPlan plan = whole_image_plan(ImageSize{2, 3});
    plan.shift_x = -1.5f;
    plan.left = -1;
    plan.shift_y = 0.5f;
    plan.crop_width = 4;
    plan.crop_height = 2;

    Labels labels;
    labels.add(3, PixelBox{1.0f, 1.0f, 2.0f, 1.5f});

    const Labels moved = Vision::Imaging::apply_darknet_augment(labels, plan, ImageSize{6, 8});

    ASSERT_EQ(moved.size(), 1u);
    EXPECT_EQ(moved.classes()[0], 3);
    EXPECT_EQ(moved.boxes()[0].x1, 5.0f);
    EXPECT_EQ(moved.boxes()[0].y1, 1.5f);
    EXPECT_EQ(moved.boxes()[0].x2, 7.0f);
    EXPECT_EQ(moved.boxes()[0].y2, 3.0f);
}

/** @brief Tests that the transform of a plan states the same map as its labels. */
TEST(Small_ImagingDarknetAugment, TheTransformStatesTheMapOfTheLabels)
{
    DarknetAugmentPlan plan = whole_image_plan(ImageSize{2, 3});
    plan.shift_x = -1.5f;
    plan.shift_y = 0.5f;
    plan.crop_width = 4;
    plan.crop_height = 2;

    const FitTransform transform = Vision::Imaging::darknet_augment_transform(plan,
                                                                              ImageSize{6, 8});

    EXPECT_EQ(transform.source, (ImageSize{2, 3}));
    EXPECT_EQ(transform.scale_x, 2.0f);
    EXPECT_EQ(transform.scale_y, 3.0f);
    EXPECT_EQ(transform.offset_x, 3.0f);
    EXPECT_EQ(transform.offset_y, -1.5f);
}

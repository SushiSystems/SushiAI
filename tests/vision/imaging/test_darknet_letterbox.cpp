/**************************************************************************/
/* test_darknet_letterbox.cpp                                             */
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
 * @file test_darknet_letterbox.cpp
 * @brief Tests the plan, the padding and the label mapping of Darknet's letterbox_image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/darknet_letterbox.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <support/imaging_images.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Vision::Imaging::DarknetLetterboxPlan;
using Vision::Imaging::FitTransform;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::PixelBox;

namespace
{
    /** @brief Holds a source, a target and the plan Darknet's integer arithmetic gives them. */
    struct PlanRow
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the target size. */
        ImageSize target;

        /** @brief Holds the size after the resize. */
        ImageSize resized;

        /** @brief Holds the padding rows above. */
        std::int32_t top;

        /** @brief Holds the padding columns left. */
        std::int32_t left;
    };

    /** @brief Holds one plan per row. */
    class Small_ImagingDarknetLetterboxPlan : public ::testing::TestWithParam<PlanRow>
    {
    };
} // namespace

/** @brief Tests that the plan truncates its sides and its offsets as Darknet's integer arithmetic does. */
TEST_P(Small_ImagingDarknetLetterboxPlan, TruncatesTheSidesAndTheOffsets)
{
    const PlanRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const DarknetLetterboxPlan plan,
                            Vision::Imaging::plan_darknet_letterbox(row.source, row.target));

    EXPECT_EQ(plan.source, row.source);
    EXPECT_EQ(plan.target, row.target);
    EXPECT_EQ(plan.resized, row.resized);
    EXPECT_EQ(plan.top, row.top);
    EXPECT_EQ(plan.left, row.left);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingDarknetLetterboxPlan,
    ::testing::Values(
        // 6 / 3 = 2 is below 6 / 2 = 3: the width is 6, the height 2 * 6 / 3 = 4, top (6 - 4) / 2.
        PlanRow{"AWideImageIsScaledUp", ImageSize{2, 3}, ImageSize{6, 6}, ImageSize{4, 6}, 1, 0},
        // 6 / 5 = 1.2 is below 4 / 3: the width is 6, the height 3 * 6 / 5 = 3.6 truncates to 3.
        PlanRow{"ASideIsTruncated", ImageSize{3, 5}, ImageSize{4, 6}, ImageSize{3, 6}, 0, 0},
        // 6 / 2 = 3 is not below 6 / 3 = 2: the height is 6, the width 2 * 6 / 3 = 4, left 1.
        PlanRow{"ATallImageIsPaddedOnTheSides", ImageSize{3, 2}, ImageSize{6, 6}, ImageSize{6, 4},
                0, 1},
        // The two ratios are equal, so the height is the target's and nothing is padded.
        PlanRow{"EqualRatiosFillTheTarget", ImageSize{2, 4}, ImageSize{4, 8}, ImageSize{4, 8}, 0,
                0},
        // 5 / 30 is below 5 / 10: the width is 5, the height 10 * 5 / 30 = 1.67 truncates to 1.
        PlanRow{"AReductionIsTruncated", ImageSize{10, 30}, ImageSize{5, 5}, ImageSize{1, 5}, 2,
                0},
        // The padding of one row is (5 - 4) / 2 = 0 above, which truncates.
        PlanRow{"AnOddPaddingGoesBelow", ImageSize{4, 5}, ImageSize{5, 6}, ImageSize{4, 6}, 0, 0}),
    [](const ::testing::TestParamInfo<PlanRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that a source that leaves a resized side of zero pixels is refused. */
TEST(Small_ImagingDarknetLetterbox, ASourceThatLeavesNoPixelsIsRefused)
{
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_letterbox(ImageSize{1, 4000}, ImageSize{4, 4}),
                      StatusCode::InvalidDescription, "plan_darknet_letterbox", "zero pixels");
}

/** @brief Tests that a size outside the bounds of an image is refused, source or target. */
TEST(Small_ImagingDarknetLetterbox, ASizeOutsideTheBoundsIsRefused)
{
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_letterbox(ImageSize{0, 4}, ImageSize{4, 4}),
                      StatusCode::InvalidDescription, "a source size of 0x4");
    SA_EXPECT_FAILURE(Vision::Imaging::plan_darknet_letterbox(ImageSize{4, 4}, ImageSize{4, 0}),
                      StatusCode::InvalidDescription, "a target size of 4x0");
}

/** @brief Tests that the image sits at the offsets of the plan and every other byte is 128. */
TEST(Small_ImagingDarknetLetterbox, TheImageSitsInGreyPadding)
{
    // 2 by 2 into 2 by 4: the height ratio is the smaller, so 2 by 2 at column (4 - 2) / 2 = 1.
    Image source(ImageSize{2, 2}, 3);
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        source.data()[i] = static_cast<std::uint8_t>(10 + i);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetLetterboxPlan plan,
                            Vision::Imaging::plan_darknet_letterbox(source.size(),
                                                                    ImageSize{2, 4}));
    Image destination;
    Vision::Imaging::apply_darknet_letterbox(source.view(), plan, destination);

    const std::uint8_t g = Vision::Imaging::K_DARKNET_PADDING;
    EXPECT_EQ(g, 128);
    SushiAI::Test::expect_bytes(destination,
                                {g, g, g, 10, 11, 12, 13, 14, 15, g, g, g,
                                 g, g, g, 16, 17, 18, 19, 20, 21, g, g, g});
}

/** @brief Tests that the plan's transform has the two scales of the truncated sides and the offsets. */
TEST(Small_ImagingDarknetLetterbox, TheTransformHasTwoScalesAndThePaddingOffsets)
{
    // The 3 by 5 image into 4 by 6 becomes 3 by 6: the width scales by 1.2, the height by 1.
    SA_ASSERT_OK_AND_ASSIGN(const DarknetLetterboxPlan plan,
                            Vision::Imaging::plan_darknet_letterbox(ImageSize{3, 5},
                                                                    ImageSize{4, 6}));
    const FitTransform transform = Vision::Imaging::darknet_letterbox_transform(plan);

    EXPECT_EQ(transform.source, (ImageSize{3, 5}));
    EXPECT_EQ(transform.scale_x, 6.0f / 5.0f);
    EXPECT_EQ(transform.scale_y, 1.0f);
    EXPECT_EQ(transform.offset_x, 0.0f);
    EXPECT_EQ(transform.offset_y, 0.0f);
}

/** @brief Tests that boxes move by the scale of their axis and the padding, and classes are kept. */
TEST(Small_ImagingDarknetLetterbox, LabelsMoveByTheScalesAndThePadding)
{
    // The 2 by 3 image into 6 by 6 becomes 4 by 6 below one row of padding: both scales are 2.
    SA_ASSERT_OK_AND_ASSIGN(const DarknetLetterboxPlan plan,
                            Vision::Imaging::plan_darknet_letterbox(ImageSize{2, 3},
                                                                    ImageSize{6, 6}));
    Labels labels;
    labels.add(7, PixelBox{0.5f, 0.25f, 2.0f, 1.5f});

    const Labels moved = Vision::Imaging::apply_darknet_letterbox(labels, plan);

    ASSERT_EQ(moved.size(), 1u);
    EXPECT_EQ(moved.classes()[0], 7);
    EXPECT_EQ(moved.boxes()[0].x1, 1.0f);
    EXPECT_EQ(moved.boxes()[0].y1, 1.5f);
    EXPECT_EQ(moved.boxes()[0].x2, 4.0f);
    EXPECT_EQ(moved.boxes()[0].y2, 4.0f);
}

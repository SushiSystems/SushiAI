/**************************************************************************/
/* test_fit_transform.cpp                                                 */
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
 * @file test_fit_transform.cpp
 * @brief Tests the placement of a fitted image and the map of a box back to its source.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/fit_transform.hpp>
#include <SushiAI/vision/imaging/labels.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds a plan that halves a 200 by 200 source, then pads 16 rows above and 8 columns left. */
    LetterboxPlan half_plan()
    {
        LetterboxPlan plan;
        plan.source = ImageSize{200, 200};
        plan.target = ImageSize{416, 416};
        plan.ratio = 0.5;
        plan.left = 8;
        plan.top = 16;
        return plan;
    }
} // namespace

/** @brief Tests that a box of a stretched image returns to the source scaled by each axis alone. */
TEST(Small_ImagingFitTransform, AResizedBoxReturnsToTheSourceByEachAxisScale)
{
    const FitTransform transform = resize_transform(ImageSize{208, 416}, ImageSize{416, 416});

    const PixelBox mapped = box_in_source(transform, PixelBox{96.0f, 64.0f, 256.0f, 192.0f});

    EXPECT_EQ(mapped.x1, 96.0f);
    EXPECT_EQ(mapped.y1, 32.0f);
    EXPECT_EQ(mapped.x2, 256.0f);
    EXPECT_EQ(mapped.y2, 96.0f);
}

/** @brief Tests that a box of a letterboxed image returns to the source past the padding. */
TEST(Small_ImagingFitTransform, ALetterboxedBoxReturnsToTheSourcePastThePadding)
{
    const FitTransform transform = letterbox_transform(half_plan());

    const PixelBox mapped = box_in_source(transform, PixelBox{24.0f, 32.0f, 72.0f, 96.0f});

    EXPECT_EQ(mapped.x1, 32.0f);
    EXPECT_EQ(mapped.y1, 32.0f);
    EXPECT_EQ(mapped.x2, 128.0f);
    EXPECT_EQ(mapped.y2, 160.0f);
}

/** @brief Tests that apply_letterbox of a box followed by box_in_source gives the box back. */
TEST(Small_ImagingFitTransform, TheBoxOfApplyLetterboxReturnsThroughTheMap)
{
    const LetterboxPlan plan = half_plan();
    Labels labels;
    labels.add(3, PixelBox{32.0f, 40.0f, 128.0f, 160.0f}, Polygon{});

    const Labels moved = apply_letterbox(labels, plan);
    const PixelBox mapped = box_in_source(letterbox_transform(plan), moved.boxes()[0]);

    EXPECT_EQ(mapped.x1, 32.0f);
    EXPECT_EQ(mapped.y1, 40.0f);
    EXPECT_EQ(mapped.x2, 128.0f);
    EXPECT_EQ(mapped.y2, 160.0f);
}

/** @brief Tests that a box reaching into the padding is clipped to the edges of the source. */
TEST(Small_ImagingFitTransform, ABoxInThePaddingIsClippedToTheSource)
{
    const FitTransform transform = letterbox_transform(half_plan());

    const PixelBox mapped = box_in_source(transform, PixelBox{0.0f, 0.0f, 416.0f, 416.0f});

    EXPECT_EQ(mapped.x1, 0.0f);
    EXPECT_EQ(mapped.y1, 0.0f);
    EXPECT_EQ(mapped.x2, 200.0f);
    EXPECT_EQ(mapped.y2, 200.0f);
}

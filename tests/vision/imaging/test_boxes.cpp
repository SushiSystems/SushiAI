/**************************************************************************/
/* test_boxes.cpp                                                         */
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
 * @file test_boxes.cpp
 * @brief Tests the box conversions, the clip and the candidate filters on hand-computed cases.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    /**
     * @brief Holds the absolute tolerance of a pixel coordinate after a round trip through the
     *        normalized form, in pixels: 32768 times the float unit roundoff, about 0.002.
     *
     * A round trip makes a handful of float operations on values up to 640; each adds at most
     * 640 * u = 3.8e-5, so 32768 u = 2e-3 covers them with a margin of fifty.
     */
    constexpr float K_PIXEL_TOLERANCE =
        static_cast<float>(32768.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the size the hand-computed cases use: 480 rows by 640 columns. */
    constexpr ImageSize K_SIZE{480, 640};

    /** @brief Checks the four corners of a pixel box against expected values, exactly. */
    void expect_pixel_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_EQ(actual.x1, x1);
        EXPECT_EQ(actual.y1, y1);
        EXPECT_EQ(actual.x2, x2);
        EXPECT_EQ(actual.y2, y2);
    }

    /** @brief Checks the four corners of a pixel box against a box within K_PIXEL_TOLERANCE. */
    void expect_pixel_box_near(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_NEAR(actual.x1, expected.x1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y1, expected.y1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.x2, expected.x2, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y2, expected.y2, K_PIXEL_TOLERANCE);
    }

    /** @brief Checks the four values of a normalized box against expected values, exactly. */
    void expect_normalized_box(const NormalizedBox& actual, float x_center, float y_center,
                               float width, float height)
    {
        EXPECT_EQ(actual.x_center, x_center);
        EXPECT_EQ(actual.y_center, y_center);
        EXPECT_EQ(actual.width, width);
        EXPECT_EQ(actual.height, height);
    }

    /** @brief Returns a box with its left top corner at the origin and the given sides. */
    PixelBox box_of_sides(float width, float height)
    {
        return PixelBox{0.0f, 0.0f, width, height};
    }

    /** @brief Holds the box a candidate test measures and whether the filter must keep it. */
    struct SideRow
    {
        PixelBox after;
        bool expected_candidate;
    };

} // namespace

/** @brief Tests that a normalized box places its corners from its centre and extent. */
TEST(Small_ImagingBoxes, ToPixelBoxPlacesCornersFromCentreAndExtent)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.5f, 0.25f, 0.5f, 0.25f}, K_SIZE);

    expect_pixel_box(box, 160.0f, 60.0f, 480.0f, 180.0f);
}

/** @brief Tests that the conversion scales x by the width and y by the height. */
TEST(Small_ImagingBoxes, ToPixelBoxUsesWidthForXAndHeightForY)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}, ImageSize{100, 200});

    expect_pixel_box(box, 0.0f, 0.0f, 200.0f, 100.0f);
}

/** @brief Tests that the conversion leaves a corner outside the image as it is. */
TEST(Small_ImagingBoxes, ToPixelBoxDoesNotClip)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.0f, 1.0f, 0.5f, 0.5f}, ImageSize{100, 200});

    expect_pixel_box(box, -50.0f, 75.0f, 50.0f, 125.0f);
}

/** @brief Tests that the conversion divides centre and extent by the image size. */
TEST(Small_ImagingBoxes, ToNormalizedBoxDividesCentreAndExtentByTheImageSize)
{
    const NormalizedBox box = to_normalized_box(PixelBox{160.0f, 60.0f, 480.0f, 180.0f}, K_SIZE);

    expect_normalized_box(box, 0.5f, 0.25f, 0.5f, 0.25f);
}

/** @brief Tests that a pixel box comes back from the normalized form and back to pixels. */
TEST(Small_ImagingBoxes, PixelToNormalizedAndBackReturnsTheBox)
{
    // A plain box, then boxes that span the image, touch a corner, mix fractions and have no area.
    const std::vector<PixelBox> boxes = {
        {12.5f, 7.25f, 630.0f, 471.75f},
        {0.0f, 0.0f, 640.0f, 480.0f},
        {0.0f, 384.0f, 128.0f, 480.0f},
        {177.6f, 308.4f, 248.64f, 331.92f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        const PixelBox back = to_pixel_box(to_normalized_box(boxes[i], K_SIZE), K_SIZE);

        SCOPED_TRACE(i);
        expect_pixel_box_near(back, boxes[i]);
    }
}

/** @brief Tests that scale_box multiplies each axis by its own factor. */
TEST(Small_ImagingBoxes, ScaleBoxMultipliesEachAxisByItsOwnFactor)
{
    const PixelBox box = scale_box(PixelBox{10.0f, 20.0f, 30.0f, 40.0f}, 2.0f, 0.5f);

    expect_pixel_box(box, 20.0f, 10.0f, 60.0f, 20.0f);
}

/** @brief Tests that translate_box adds each axis its own offset. */
TEST(Small_ImagingBoxes, TranslateBoxAddsEachAxisItsOwnOffset)
{
    const PixelBox box = translate_box(PixelBox{10.0f, 20.0f, 30.0f, 40.0f}, 5.0f, -25.0f);

    expect_pixel_box(box, 15.0f, -5.0f, 35.0f, 15.0f);
}

/** @brief Tests that clip_box clamps each edge to the image. */
TEST(Small_ImagingBoxes, ClipBoxClampsEachEdgeToTheImage)
{
    const PixelBox box = clip_box(PixelBox{-10.0f, 20.0f, 700.0f, 500.0f}, K_SIZE);

    expect_pixel_box(box, 0.0f, 20.0f, 640.0f, 480.0f);
}

/** @brief Tests that clip_box leaves a box that fills the image alone. */
TEST(Small_ImagingBoxes, ClipBoxLeavesABoxInsideTheImageAlone)
{
    const PixelBox box = clip_box(PixelBox{0.0f, 0.0f, 640.0f, 480.0f}, K_SIZE);

    expect_pixel_box(box, 0.0f, 0.0f, 640.0f, 480.0f);
}

/** @brief Tests that clip_box puts a box outside the image on the nearest edge. */
TEST(Small_ImagingBoxes, ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge)
{
    const PixelBox right = clip_box(PixelBox{700.0f, 10.0f, 800.0f, 20.0f}, K_SIZE);
    const PixelBox above = clip_box(PixelBox{10.0f, -30.0f, 20.0f, -5.0f}, K_SIZE);

    expect_pixel_box(right, 640.0f, 10.0f, 640.0f, 20.0f);
    expect_pixel_box(above, 10.0f, 0.0f, 20.0f, 0.0f);
}

/** @brief Tests that a box clipped from outside the image has zero area. */
TEST(Small_ImagingBoxes, ClipBoxGivesABoxOutsideTheImageZeroArea)
{
    const PixelBox right = clip_box(PixelBox{700.0f, 10.0f, 800.0f, 20.0f}, K_SIZE);
    const PixelBox above = clip_box(PixelBox{10.0f, -30.0f, 20.0f, -5.0f}, K_SIZE);

    EXPECT_EQ(box_area(right), 0.0f);
    EXPECT_EQ(box_area(above), 0.0f);
}

/** @brief Tests that the area of a box is its width times its height. */
TEST(Small_ImagingBoxes, BoxAreaIsWidthTimesHeight)
{
    EXPECT_EQ(box_area(PixelBox{1.0f, 2.0f, 4.0f, 6.0f}), 12.0f);
}

/** @brief Tests that the polygon filter differs from the default one in its area ratio alone. */
TEST(Small_ImagingBoxes, PolygonFilterLowersOnlyTheAreaRatioToOneHundredth)
{
    const CandidateFilter plain;
    const CandidateFilter polygon = make_polygon_filter();

    EXPECT_EQ(polygon.min_area_ratio, 0.01f);
    EXPECT_EQ(polygon.min_side, plain.min_side);
    EXPECT_EQ(polygon.max_aspect_ratio, plain.max_aspect_ratio);
    EXPECT_EQ(polygon.epsilon, plain.epsilon);
}

/** @brief Tests that a box at 5 percent of its area passes the polygon filter alone. */
TEST(Small_ImagingBoxes, PolygonFilterKeepsABoxThePlainFilterRejects)
{
    const PixelBox before{0.0f, 0.0f, 100.0f, 100.0f};
    const PixelBox after{0.0f, 0.0f, 50.0f, 10.0f};

    EXPECT_FALSE(is_box_candidate(before, after, CandidateFilter{}));
    EXPECT_TRUE(is_box_candidate(before, after, make_polygon_filter()));
}

namespace
{
/** @brief Rows of a 10 by 10 box warped to the row's box. */
class Small_ImagingBoxesSide : public ::testing::TestWithParam<SideRow>
{
};
} // namespace

/** @brief Tests that a candidate needs each side strictly above the minimum side. */
TEST_P(Small_ImagingBoxesSide, CandidateNeedsASideAboveTheMinimumSide)
{
    const SideRow& row = GetParam();

    EXPECT_EQ(is_box_candidate(box_of_sides(10.0f, 10.0f), row.after, CandidateFilter{}),
              row.expected_candidate);
}

INSTANTIATE_TEST_SUITE_P(Sides, Small_ImagingBoxesSide,
                         ::testing::Values(SideRow{box_of_sides(2.0f, 10.0f), false},
                                           SideRow{box_of_sides(2.25f, 10.0f), true},
                                           SideRow{box_of_sides(10.0f, 2.0f), false},
                                           SideRow{box_of_sides(10.0f, 2.25f), true}));

/** @brief Tests that the area ratio must be strictly above the minimum. */
TEST(Small_ImagingBoxes, CandidateNeedsMoreThanTheMinimumAreaRatio)
{
    const CandidateFilter filter;
    const PixelBox before = box_of_sides(100.0f, 100.0f);

    // 10 * 100 / (100 * 100) is 0.10 in float, which is not above 0.10.
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(10.0f, 100.0f), filter));
    EXPECT_TRUE(is_box_candidate(before, box_of_sides(10.5f, 100.0f), filter));
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(9.5f, 100.0f), filter));
}

/** @brief Tests that the area ratio used is the filter's own field. */
TEST(Small_ImagingBoxes, CandidateAreaRatioFollowsTheFilterField)
{
    CandidateFilter filter;
    filter.min_area_ratio = 0.01f;
    const PixelBox before = box_of_sides(100.0f, 100.0f);

    EXPECT_TRUE(is_box_candidate(before, box_of_sides(10.0f, 100.0f), filter));
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(10.0f, 10.0f), filter));
}

/** @brief Tests that the aspect ratio must be strictly below the maximum. */
TEST(Small_ImagingBoxes, CandidateNeedsAnAspectRatioBelowTheMaximum)
{
    const CandidateFilter filter;

    // 300 / (3 + 1e-16) is 100 in float, which is not below 100.
    const PixelBox wide = box_of_sides(300.0f, 3.0f);
    const PixelBox tall = box_of_sides(3.0f, 300.0f);
    const PixelBox nearly_wide = box_of_sides(297.0f, 3.0f);
    const PixelBox nearly_tall = box_of_sides(3.0f, 297.0f);

    EXPECT_FALSE(is_box_candidate(wide, wide, filter));
    EXPECT_FALSE(is_box_candidate(tall, tall, filter));
    EXPECT_TRUE(is_box_candidate(nearly_wide, nearly_wide, filter));
    EXPECT_TRUE(is_box_candidate(nearly_tall, nearly_tall, filter));
}

/** @brief Tests that the candidate test reads side lengths and not positions. */
TEST(Small_ImagingBoxes, CandidateMeasuresSidesNotPositions)
{
    const CandidateFilter filter;
    const PixelBox before{100.0f, 200.0f, 150.0f, 240.0f};
    const PixelBox after{300.0f, 50.0f, 350.0f, 90.0f};

    EXPECT_TRUE(is_box_candidate(before, after, filter));
}

/** @brief Tests that the epsilon is added to the area before the ratio. */
TEST(Small_ImagingBoxes, CandidateEpsilonIsAddedToTheAreaBefore)
{
    CandidateFilter filter;
    filter.epsilon = 100.0f;
    filter.min_area_ratio = 0.6f;
    const PixelBox box = box_of_sides(10.0f, 10.0f);

    // 100 / (100 + 100) is 0.5, below 0.6; without epsilon the ratio is 1.
    EXPECT_FALSE(is_box_candidate(box, box, filter));
}

/** @brief Tests that the epsilon is added to the short side of a wide box and of a tall box. */
TEST(Small_ImagingBoxes, CandidateEpsilonIsAddedToTheShortSide)
{
    CandidateFilter filter;
    filter.epsilon = 1.0f;
    const PixelBox boxes[] = {box_of_sides(300.0f, 3.0f), box_of_sides(3.0f, 300.0f)};

    // 300 / (3 + 1) is 75, below 100; without epsilon the ratio is 100.
    for (const PixelBox& box : boxes)
        EXPECT_TRUE(is_box_candidate(box, box, filter)) << box.x2 << " by " << box.y2;
}

/** @brief Tests that an empty box before the augmentation is kept. */
TEST(Small_ImagingBoxes, CandidateWithAnEmptyBeforeBoxIsKept)
{
    const CandidateFilter filter;

    EXPECT_TRUE(is_box_candidate(box_of_sides(0.0f, 0.0f), box_of_sides(10.0f, 10.0f), filter));
}

/** @brief Tests that an empty box after the augmentation is dropped. */
TEST(Small_ImagingBoxes, CandidateWithAnEmptyAfterBoxIsDropped)
{
    const CandidateFilter filter;

    EXPECT_FALSE(is_box_candidate(box_of_sides(10.0f, 10.0f), box_of_sides(0.0f, 0.0f), filter));
}

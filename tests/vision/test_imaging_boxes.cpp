/**************************************************************************/
/* test_imaging_boxes.cpp                                                 */
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
 * @file test_imaging_boxes.cpp
 * @brief Tests the box conversions, the clip and the candidate filters on hand-computed cases.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/boxes.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the absolute tolerance of a pixel coordinate, in pixels. */
    constexpr float K_PIXEL_TOLERANCE = 1e-3f;

    /** @brief Holds the absolute tolerance of a normalized coordinate, a fraction of the image. */
    constexpr float K_FRACTION_TOLERANCE = 1e-6f;

    /** @brief Holds the absolute tolerance of the filter's default epsilon, which is 1e-16. */
    constexpr float K_EPSILON_TOLERANCE = 1e-22f;

    /** @brief Holds the size the hand-computed cases use: 480 rows by 640 columns. */
    constexpr ImageSize K_SIZE{480, 640};

    /** @brief Checks the four corners of a pixel box against expected values. */
    void expect_pixel_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_NEAR(actual.x1, x1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y1, y1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.x2, x2, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y2, y2, K_PIXEL_TOLERANCE);
    }

    /** @brief Checks the four values of a normalized box against expected values. */
    void expect_normalized_box(const NormalizedBox& actual, float x_center, float y_center,
                               float width, float height)
    {
        EXPECT_NEAR(actual.x_center, x_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.y_center, y_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.width, width, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.height, height, K_FRACTION_TOLERANCE);
    }

    /** @brief Returns a box with its left top corner at the origin and the given sides. */
    PixelBox box_of_sides(float width, float height)
    {
        return PixelBox{0.0f, 0.0f, width, height};
    }
} // namespace

TEST(Unit_ImagingBoxes, ToPixelBoxPlacesCornersFromCentreAndExtent)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.5f, 0.25f, 0.5f, 0.25f}, K_SIZE);

    expect_pixel_box(box, 160.0f, 60.0f, 480.0f, 180.0f);
}

TEST(Unit_ImagingBoxes, ToPixelBoxUsesWidthForXAndHeightForY)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}, ImageSize{100, 200});

    expect_pixel_box(box, 0.0f, 0.0f, 200.0f, 100.0f);
}

TEST(Unit_ImagingBoxes, ToPixelBoxDoesNotClip)
{
    const PixelBox box = to_pixel_box(NormalizedBox{0.0f, 1.0f, 0.5f, 0.5f}, ImageSize{100, 200});

    expect_pixel_box(box, -50.0f, 75.0f, 50.0f, 125.0f);
}

TEST(Unit_ImagingBoxes, ToNormalizedBoxDividesCentreAndExtentByTheImageSize)
{
    const NormalizedBox box = to_normalized_box(PixelBox{160.0f, 60.0f, 480.0f, 180.0f}, K_SIZE);

    expect_normalized_box(box, 0.5f, 0.25f, 0.5f, 0.25f);
}

TEST(Unit_ImagingBoxes, NormalizedToPixelAndBackReturnsTheBox)
{
    const std::vector<NormalizedBox> boxes = {
        {0.5f, 0.5f, 1.0f, 1.0f},
        {0.1f, 0.9f, 0.2f, 0.2f},
        {0.333f, 0.667f, 0.111f, 0.049f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    for (const NormalizedBox& box : boxes)
    {
        const NormalizedBox back = to_normalized_box(to_pixel_box(box, K_SIZE), K_SIZE);

        expect_normalized_box(back, box.x_center, box.y_center, box.width, box.height);
    }
}

TEST(Unit_ImagingBoxes, PixelToNormalizedAndBackReturnsTheBox)
{
    const PixelBox box{12.5f, 7.25f, 630.0f, 471.75f};

    const PixelBox back = to_pixel_box(to_normalized_box(box, K_SIZE), K_SIZE);

    expect_pixel_box(back, box.x1, box.y1, box.x2, box.y2);
}

TEST(Unit_ImagingBoxes, ScaleBoxMultipliesEachAxisByItsOwnFactor)
{
    const PixelBox box = scale_box(PixelBox{10.0f, 20.0f, 30.0f, 40.0f}, 2.0f, 0.5f);

    expect_pixel_box(box, 20.0f, 10.0f, 60.0f, 20.0f);
}

TEST(Unit_ImagingBoxes, TranslateBoxAddsEachAxisItsOwnOffset)
{
    const PixelBox box = translate_box(PixelBox{10.0f, 20.0f, 30.0f, 40.0f}, 5.0f, -25.0f);

    expect_pixel_box(box, 15.0f, -5.0f, 35.0f, 15.0f);
}

TEST(Unit_ImagingBoxes, ClipBoxClampsEachEdgeToTheImage)
{
    const PixelBox box = clip_box(PixelBox{-10.0f, 20.0f, 700.0f, 500.0f}, K_SIZE);

    expect_pixel_box(box, 0.0f, 20.0f, 640.0f, 480.0f);
}

TEST(Unit_ImagingBoxes, ClipBoxLeavesABoxInsideTheImageAlone)
{
    const PixelBox box = clip_box(PixelBox{0.0f, 0.0f, 640.0f, 480.0f}, K_SIZE);

    expect_pixel_box(box, 0.0f, 0.0f, 640.0f, 480.0f);
}

TEST(Unit_ImagingBoxes, ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge)
{
    const PixelBox right = clip_box(PixelBox{700.0f, 10.0f, 800.0f, 20.0f}, K_SIZE);
    const PixelBox above = clip_box(PixelBox{10.0f, -30.0f, 20.0f, -5.0f}, K_SIZE);

    expect_pixel_box(right, 640.0f, 10.0f, 640.0f, 20.0f);
    expect_pixel_box(above, 10.0f, 0.0f, 20.0f, 0.0f);
    EXPECT_NEAR(box_area(right), 0.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(box_area(above), 0.0f, K_PIXEL_TOLERANCE);
}

TEST(Unit_ImagingBoxes, BoxAreaIsWidthTimesHeight)
{
    EXPECT_NEAR(box_area(PixelBox{1.0f, 2.0f, 4.0f, 6.0f}), 12.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that the polygon filter differs from the default one in its area ratio alone. */
TEST(Unit_ImagingBoxes, PolygonFilterLowersOnlyTheAreaRatioToOneHundredth)
{
    const CandidateFilter plain;
    const CandidateFilter polygon = make_polygon_filter();

    EXPECT_NEAR(polygon.min_area_ratio, 0.01f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(K_POLYGON_MIN_AREA_RATIO, 0.01f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(polygon.min_side, plain.min_side, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(polygon.max_aspect_ratio, plain.max_aspect_ratio, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(polygon.epsilon, plain.epsilon, K_EPSILON_TOLERANCE);
}

/** @brief Tests that a box at 5 percent of its area passes the polygon filter alone. */
TEST(Unit_ImagingBoxes, PolygonFilterKeepsABoxThePlainFilterRejects)
{
    const PixelBox before{0.0f, 0.0f, 100.0f, 100.0f};
    const PixelBox after{0.0f, 0.0f, 50.0f, 10.0f};

    EXPECT_FALSE(is_box_candidate(before, after, CandidateFilter{}));
    EXPECT_TRUE(is_box_candidate(before, after, make_polygon_filter()));
}

TEST(Unit_ImagingBoxes, CandidateFilterDefaultsAreTheReferenceThresholds)
{
    const CandidateFilter filter;

    EXPECT_NEAR(filter.min_side, 2.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(filter.max_aspect_ratio, 100.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(filter.min_area_ratio, 0.10f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(filter.epsilon, 1e-16f, K_EPSILON_TOLERANCE);
}

TEST(Unit_ImagingBoxes, CandidateNeedsAWidthAboveTheMinimumSide)
{
    const CandidateFilter filter;
    const PixelBox before = box_of_sides(10.0f, 10.0f);

    EXPECT_FALSE(is_box_candidate(before, box_of_sides(2.0f, 10.0f), filter));
    EXPECT_TRUE(is_box_candidate(before, box_of_sides(2.25f, 10.0f), filter));
}

TEST(Unit_ImagingBoxes, CandidateNeedsAHeightAboveTheMinimumSide)
{
    const CandidateFilter filter;
    const PixelBox before = box_of_sides(10.0f, 10.0f);

    EXPECT_FALSE(is_box_candidate(before, box_of_sides(10.0f, 2.0f), filter));
    EXPECT_TRUE(is_box_candidate(before, box_of_sides(10.0f, 2.25f), filter));
}

TEST(Unit_ImagingBoxes, CandidateNeedsMoreThanTheMinimumAreaRatio)
{
    const CandidateFilter filter;
    const PixelBox before = box_of_sides(100.0f, 100.0f);

    // 10 * 100 / (100 * 100) is 0.10 in float, which is not above 0.10.
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(10.0f, 100.0f), filter));
    EXPECT_TRUE(is_box_candidate(before, box_of_sides(10.5f, 100.0f), filter));
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(9.5f, 100.0f), filter));
}

TEST(Unit_ImagingBoxes, CandidateAreaRatioFollowsTheFilterField)
{
    CandidateFilter filter;
    filter.min_area_ratio = 0.01f;
    const PixelBox before = box_of_sides(100.0f, 100.0f);

    EXPECT_TRUE(is_box_candidate(before, box_of_sides(10.0f, 100.0f), filter));
    EXPECT_FALSE(is_box_candidate(before, box_of_sides(10.0f, 10.0f), filter));
}

TEST(Unit_ImagingBoxes, CandidateNeedsAnAspectRatioBelowTheMaximum)
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

TEST(Unit_ImagingBoxes, CandidateMeasuresSidesNotPositions)
{
    const CandidateFilter filter;
    const PixelBox before{100.0f, 200.0f, 150.0f, 240.0f};
    const PixelBox after{300.0f, 50.0f, 350.0f, 90.0f};

    EXPECT_TRUE(is_box_candidate(before, after, filter));
}

TEST(Unit_ImagingBoxes, CandidateEpsilonIsAddedToTheAreaBefore)
{
    CandidateFilter filter;
    filter.epsilon = 100.0f;
    filter.min_area_ratio = 0.6f;
    const PixelBox box = box_of_sides(10.0f, 10.0f);

    // 100 / (100 + 100) is 0.5, below 0.6; without epsilon the ratio is 1.
    EXPECT_FALSE(is_box_candidate(box, box, filter));
}

TEST(Unit_ImagingBoxes, CandidateEpsilonIsAddedToTheHeightOfAWideBox)
{
    CandidateFilter filter;
    filter.epsilon = 1.0f;
    const PixelBox wide = box_of_sides(300.0f, 3.0f);

    // 300 / (3 + 1) is 75, below 100; without epsilon the ratio is 100.
    EXPECT_TRUE(is_box_candidate(wide, wide, filter));
}

TEST(Unit_ImagingBoxes, CandidateEpsilonIsAddedToTheWidthOfATallBox)
{
    CandidateFilter filter;
    filter.epsilon = 1.0f;
    const PixelBox tall = box_of_sides(3.0f, 300.0f);

    // 300 / (3 + 1) is 75, below 100; without epsilon the ratio is 100.
    EXPECT_TRUE(is_box_candidate(tall, tall, filter));
}

TEST(Unit_ImagingBoxes, CandidateWithAnEmptyBeforeBoxIsKeptAndAnEmptyAfterBoxIsNot)
{
    const CandidateFilter filter;

    EXPECT_TRUE(is_box_candidate(box_of_sides(0.0f, 0.0f), box_of_sides(10.0f, 10.0f), filter));
    EXPECT_FALSE(is_box_candidate(box_of_sides(10.0f, 10.0f), box_of_sides(0.0f, 0.0f), filter));
}

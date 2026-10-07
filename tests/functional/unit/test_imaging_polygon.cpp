/**************************************************************************/
/* test_imaging_polygon.cpp                                               */
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
 * @file test_imaging_polygon.cpp
 * @brief Tests the polygon rules against values the pinned reference package printed.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/imaging/polygon.hpp>

#include "../../common/test_throws.hpp"

using namespace SushiAI;
using namespace SushiAI::Imaging;

namespace
{
    /** @brief Holds the absolute tolerance of a resampled coordinate of magnitude up to 4. */
    constexpr float K_POINT_TOLERANCE = 1e-6f;

    /** @brief Holds the absolute tolerance, in pixels, of a box edge of magnitude up to 100. */
    constexpr float K_BOX_TOLERANCE = 1e-4f;

    /** @brief Holds the image the reference's segment2box was run on: 80 rows by 100 columns. */
    constexpr ImageSize K_IMAGE{80, 100};

    /** @brief Checks a polygon against expected vertices, x then y, within K_POINT_TOLERANCE. */
    void expect_polygon(const Polygon& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size() * 2, expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i].x, expected[2 * i], K_POINT_TOLERANCE) << "point " << i;
            EXPECT_NEAR(actual[i].y, expected[2 * i + 1], K_POINT_TOLERANCE) << "point " << i;
        }
    }

    /** @brief Checks the four edges of a box within K_BOX_TOLERANCE. */
    void expect_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_NEAR(actual.x1, x1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y1, y1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.x2, x2, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y2, y2, K_BOX_TOLERANCE);
    }

    /** @brief Returns the square with corners (0, 0) and (4, 4), clockwise from the origin. */
    Polygon make_square()
    {
        return Polygon{Point{0.0f, 0.0f}, Point{4.0f, 0.0f}, Point{4.0f, 4.0f}, Point{0.0f, 4.0f}};
    }

    /** @brief Returns a triangle in fractions of an image. */
    Polygon make_triangle()
    {
        return Polygon{Point{0.1f, 0.2f}, Point{0.7f, 0.3f}, Point{0.4f, 0.9f}};
    }

    /** @brief Returns a hexagon of six vertices with integer coordinates. */
    Polygon make_hexagon()
    {
        return Polygon{Point{1.0f, 0.0f}, Point{3.0f, 0.0f}, Point{4.0f, 2.0f},
                       Point{3.0f, 4.0f}, Point{1.0f, 4.0f}, Point{0.0f, 2.0f}};
    }
} // namespace

/** @brief Tests that the point count is 1000 up to 1000 vertices and the length plus one above. */
TEST(Unit_ImagingPolygon, PointCountIsAThousandOrTheLongestPlusOne)
{
    EXPECT_EQ(K_POLYGON_POINTS, 1000u);
    EXPECT_EQ(resampled_point_count(0), 1000u);
    EXPECT_EQ(resampled_point_count(3), 1000u);
    EXPECT_EQ(resampled_point_count(1000), 1000u);
    EXPECT_EQ(resampled_point_count(1001), 1002u);
}

/** @brief Tests that a short polygon keeps its vertices and gains evenly spread positions. */
TEST(Unit_ImagingPolygon, ResampleMergesTheVerticesWithEvenPositions)
{
    // resample_segments([square], 8) and ([square], 11) of the reference.
    expect_polygon(resample_polygon(make_square(), 8),
                   {0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f, 4.0f, 0.0f,
                    0.0f, 0.0f, 0.0f});
    expect_polygon(resample_polygon(make_square(), 11),
                   {0.0f, 0.0f, 0.0f, 0.0f, 3.2f, 0.0f, 4.0f, 0.0f, 4.0f, 2.4f, 4.0f,
                    4.0f, 2.4f, 4.0f, 0.0f, 4.0f, 0.0f, 3.2f, 0.0f, 0.0f, 0.0f, 0.0f});
}

/** @brief Tests that a count one above the closed length adds one point at the first vertex. */
TEST(Unit_ImagingPolygon, ResampleToOneMoreThanTheClosedLengthRepeatsTheFirstVertex)
{
    // resample_segments([square], 6) of the reference: linspace(0, 4, 1) is [0.0].
    expect_polygon(resample_polygon(make_square(), 6),
                   {0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f, 4.0f, 0.0f, 4.0f, 0.0f, 0.0f});
}

/** @brief Tests that a position between two vertices is interpolated along their edge. */
TEST(Unit_ImagingPolygon, ResampleInterpolatesAlongTheEdges)
{
    // resample_segments([triangle], 9) of the reference.
    expect_polygon(resample_polygon(make_triangle(), 9),
                   {0.1f, 0.2f, 0.1f, 0.2f, 0.55f, 0.275f, 0.7f, 0.3f, 0.55f, 0.6f, 0.4f, 0.9f,
                    0.325f, 0.725f, 0.1f, 0.2f, 0.1f, 0.2f});
}

/** @brief Tests that a closed polygon no shorter than the count is sampled at even positions. */
TEST(Unit_ImagingPolygon, ResampleOfALongPolygonTakesEvenPositionsOnly)
{
    // resample_segments([hexagon], 4) and ([hexagon], 7) of the reference.
    expect_polygon(resample_polygon(make_hexagon(), 4),
                   {1.0f, 0.0f, 4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 0.0f});
    expect_polygon(resample_polygon(make_hexagon(), 7),
                   {1.0f, 0.0f, 3.0f, 0.0f, 4.0f, 2.0f, 3.0f, 4.0f, 1.0f, 4.0f, 0.0f, 2.0f, 1.0f,
                    0.0f});
}

/** @brief Tests that a polygon of exactly the count and an empty one come back unchanged. */
TEST(Unit_ImagingPolygon, ResampleLeavesAPolygonOfTheCountAndAnEmptyOneAlone)
{
    expect_polygon(resample_polygon(make_triangle(), 3), {0.1f, 0.2f, 0.7f, 0.3f, 0.4f, 0.9f});
    EXPECT_TRUE(resample_polygon(Polygon{}, 1000).empty());
}

/** @brief Tests that resampling to 1000 points keeps the count, the vertices and the bounds. */
TEST(Unit_ImagingPolygon, ResampleToAThousandPointsKeepsTheOutline)
{
    const Polygon resampled = resample_polygon(make_triangle(), K_POLYGON_POINTS);

    ASSERT_EQ(resampled.size(), K_POLYGON_POINTS);
    EXPECT_NEAR(resampled.front().x, 0.1f, K_POINT_TOLERANCE);
    EXPECT_NEAR(resampled.back().x, 0.1f, K_POINT_TOLERANCE);

    const PixelBox bounds = visible_polygon_box(resampled, ImageSize{1, 1});
    EXPECT_NEAR(bounds.x1, 0.1f, K_POINT_TOLERANCE);
    EXPECT_NEAR(bounds.y1, 0.2f, K_POINT_TOLERANCE);
    EXPECT_NEAR(bounds.x2, 0.7f, K_POINT_TOLERANCE);
    EXPECT_NEAR(bounds.y2, 0.9f, K_POINT_TOLERANCE);
}

/** @brief Tests that a count of zero is refused. */
TEST(Unit_ImagingPolygon, ResampleRejectsACountOfZero)
{
    SA_EXPECT_ERROR(static_cast<void>(resample_polygon(make_triangle(), 0)), "resample_polygon",
                    "point count of 0");
}

/** @brief Tests that map_polygon multiplies each axis by its scale, then adds its offset. */
TEST(Unit_ImagingPolygon, MapScalesThenTranslatesEachAxis)
{
    Polygon polygon = make_square();

    map_polygon(polygon, 2.0f, 0.5f, 10.0f, -1.0f);

    expect_polygon(polygon, {10.0f, -1.0f, 18.0f, -1.0f, 18.0f, 1.0f, 10.0f, 1.0f});
}

/** @brief Tests that clamp_polygon moves every vertex outside the bounds onto them. */
TEST(Unit_ImagingPolygon, ClampMovesVerticesOntoTheBounds)
{
    Polygon polygon{Point{-3.0f, 2.0f}, Point{9.0f, -1.0f}, Point{2.0f, 7.0f}, Point{1.0f, 1.0f}};

    clamp_polygon(polygon, PixelBox{0.0f, 0.0f, 4.0f, 5.0f});

    expect_polygon(polygon, {0.0f, 2.0f, 4.0f, 0.0f, 2.0f, 5.0f, 1.0f, 1.0f});
}

/** @brief Tests that a point inside, on an edge or on a vertex counts as inside. */
TEST(Unit_ImagingPolygon, PointInsideOrOnTheOutlineIsInside)
{
    const Polygon square = make_square();

    EXPECT_TRUE(is_point_inside(square, Point{2.0f, 2.0f}));
    EXPECT_TRUE(is_point_inside(square, Point{4.0f, 1.5f}));
    EXPECT_TRUE(is_point_inside(square, Point{1.0f, 0.0f}));
    EXPECT_TRUE(is_point_inside(square, Point{0.0f, 0.0f}));
    EXPECT_TRUE(is_point_inside(square, Point{4.0f, 4.0f}));
}

/** @brief Tests that a point outside is outside, also in the notch of a concave polygon. */
TEST(Unit_ImagingPolygon, PointOutsideIsOutside)
{
    const Polygon square = make_square();
    const Polygon notched{Point{0.0f, 0.0f}, Point{6.0f, 0.0f}, Point{6.0f, 6.0f},
                          Point{3.0f, 2.0f}, Point{0.0f, 6.0f}};

    EXPECT_FALSE(is_point_inside(square, Point{4.5f, 2.0f}));
    EXPECT_FALSE(is_point_inside(square, Point{-0.5f, 2.0f}));
    EXPECT_FALSE(is_point_inside(square, Point{2.0f, 4.5f}));
    EXPECT_FALSE(is_point_inside(square, Point{5.0f, 0.0f}));
    EXPECT_FALSE(is_point_inside(notched, Point{3.0f, 5.0f}));
    EXPECT_TRUE(is_point_inside(notched, Point{3.0f, 1.0f}));
    EXPECT_TRUE(is_point_inside(notched, Point{1.0f, 4.0f}));
    EXPECT_FALSE(is_point_inside(Polygon{}, Point{0.0f, 0.0f}));
}

/** @brief Tests that a polygon inside the image gives the bounds of its vertices. */
TEST(Unit_ImagingPolygon, VisibleBoxOfAPolygonInsideTheImageIsItsBounds)
{
    const Polygon inside{Point{10.0f, 20.0f}, Point{50.0f, 25.0f}, Point{30.0f, 60.0f}};

    expect_box(visible_polygon_box(inside, K_IMAGE), 10.0f, 20.0f, 50.0f, 60.0f);
}

/** @brief Tests that a polygon wholly outside the image, and an empty one, give four zeros. */
TEST(Unit_ImagingPolygon, VisibleBoxOfAPolygonOutsideTheImageIsZero)
{
    const Polygon above_left{Point{-30.0f, -20.0f}, Point{-5.0f, -25.0f}, Point{-10.0f, -3.0f}};
    const Polygon right{Point{120.0f, 10.0f}, Point{180.0f, 40.0f}, Point{130.0f, 90.0f}};

    expect_box(visible_polygon_box(above_left, K_IMAGE), 0.0f, 0.0f, 0.0f, 0.0f);
    expect_box(visible_polygon_box(right, K_IMAGE), 0.0f, 0.0f, 0.0f, 0.0f);
    expect_box(visible_polygon_box(Polygon{}, K_IMAGE), 0.0f, 0.0f, 0.0f, 0.0f);
}

/** @brief Tests that edges crossing the image's lines bound the visible part. */
TEST(Unit_ImagingPolygon, VisibleBoxUsesTheCrossingsOfTheImageLines)
{
    // segment2box of the reference gives these for a 100 by 80 image (width by height).
    const Polygon crossing{Point{-20.0f, 30.0f}, Point{40.0f, -10.0f}, Point{60.0f, 50.0f}};
    const Polygon slant{Point{90.0f, 70.0f}, Point{130.0f, 75.0f}, Point{95.0f, 120.0f}};

    expect_box(visible_polygon_box(crossing, K_IMAGE), 0.0f, 0.0f, 60.0f, 50.0f);
    expect_box(visible_polygon_box(slant, K_IMAGE), 90.0f, 70.0f, 100.0f, 80.0f);
}

/** @brief Tests that an image corner inside the polygon belongs to the visible part. */
TEST(Unit_ImagingPolygon, VisibleBoxIncludesTheImageCornersThePolygonContains)
{
    // segment2box of the reference: (0, 0, 36.666664, 33) and the whole image.
    const Polygon corner{Point{-40.0f, -40.0f}, Point{70.0f, -30.0f}, Point{-30.0f, 60.0f}};
    const Polygon cover{Point{-50.0f, -50.0f}, Point{500.0f, -50.0f}, Point{500.0f, 500.0f},
                        Point{-50.0f, 500.0f}};

    expect_box(visible_polygon_box(corner, K_IMAGE), 0.0f, 0.0f, 36.666664f, 33.0f);
    expect_box(visible_polygon_box(cover, K_IMAGE), 0.0f, 0.0f, 100.0f, 80.0f);
}

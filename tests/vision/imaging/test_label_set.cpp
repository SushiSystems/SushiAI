/**************************************************************************/
/* test_label_set.cpp                                                     */
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
 * @file test_label_set.cpp
 * @brief Tests the label set's count invariant and the rules that apply to a whole set.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the absolute tolerance of a pixel coordinate, in pixels. */
    constexpr float K_PIXEL_TOLERANCE = 1e-3f;

    /** @brief Holds the absolute tolerance of a normalized coordinate, a fraction of the image. */
    constexpr float K_FRACTION_TOLERANCE = 1e-6f;

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

    /** @brief Checks one vertex against expected coordinates within a tolerance. */
    void expect_point(const Point& actual, float x, float y, float tolerance)
    {
        EXPECT_NEAR(actual.x, x, tolerance);
        EXPECT_NEAR(actual.y, y, tolerance);
    }

    /** @brief Returns a polygon of a number of vertices spread along a diagonal. */
    Polygon make_diagonal(std::size_t count)
    {
        Polygon polygon;
        for (std::size_t i = 0; i < count; ++i)
        {
            const float value = static_cast<float>(i) / static_cast<float>(count);
            polygon.push_back(Point{value, value});
        }

        return polygon;
    }
} // namespace

/** @brief Tests that a default set holds no object in any of its three lists. */
TEST(Unit_ImagingLabelSet, DefaultSetHoldsNoObject)
{
    const Labels labels;

    EXPECT_TRUE(labels.is_empty());
    EXPECT_EQ(labels.size(), 0u);
    EXPECT_TRUE(labels.classes().empty());
    EXPECT_TRUE(labels.boxes().empty());
    EXPECT_TRUE(labels.polygons().empty());
}

/** @brief Tests that the constructors refuse lists of different lengths. */
TEST(Unit_ImagingLabelSet, ConstructorsRejectListsOfDifferentLengths)
{
    const PixelBox box{0.0f, 0.0f, 1.0f, 1.0f};

    SA_EXPECT_ERROR(Labels({1, 2}, {box}), "LabelSet", "2 classes", "1 boxes");
    SA_EXPECT_ERROR(Labels({1}, {box, box}), "LabelSet", "1 classes", "2 boxes");
    SA_EXPECT_ERROR(Labels({1, 2}, {box, box}, {Polygon{}}), "LabelSet", "1 polygons");
    SA_EXPECT_ERROR(FileLabels({}, {NormalizedBox{}}), "LabelSet", "0 classes", "1 boxes");
}

/** @brief Tests that a set built from classes and boxes gives each object an empty polygon. */
TEST(Unit_ImagingLabelSet, SetWithoutPolygonsHoldsOneEmptyPolygonPerObject)
{
    const Labels labels({4, 9},
                        {PixelBox{0.0f, 0.0f, 1.0f, 1.0f}, PixelBox{2.0f, 2.0f, 3.0f, 3.0f}});

    ASSERT_EQ(labels.size(), 2u);
    ASSERT_EQ(labels.polygons().size(), 2u);
    EXPECT_TRUE(labels.polygons()[0].empty());
    EXPECT_TRUE(labels.polygons()[1].empty());
}

/** @brief Tests that add grows the three lists together, with or without a polygon. */
TEST(Unit_ImagingLabelSet, AddAppendsOneEntryToEachList)
{
    Labels labels;

    labels.add(5, PixelBox{1.0f, 2.0f, 3.0f, 4.0f});
    labels.add(6, PixelBox{5.0f, 6.0f, 7.0f, 8.0f}, Polygon{Point{5.0f, 6.0f}, Point{7.0f, 8.0f}});

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{5, 6}));
    ASSERT_EQ(labels.boxes().size(), 2u);
    ASSERT_EQ(labels.polygons().size(), 2u);
    EXPECT_TRUE(labels.polygons()[0].empty());
    ASSERT_EQ(labels.polygons()[1].size(), 2u);
    expect_point(labels.polygons()[1][1], 7.0f, 8.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that keep_if removes an object's class, box and polygon together. */
TEST(Unit_ImagingLabelSet, KeepIfRemovesAnObjectFromAllThreeLists)
{
    Labels labels;
    labels.add(1, PixelBox{0.0f, 0.0f, 1.0f, 1.0f}, Polygon{Point{0.0f, 0.0f}});
    labels.add(2, PixelBox{0.0f, 0.0f, 2.0f, 2.0f}, Polygon{Point{2.0f, 2.0f}, Point{2.0f, 0.0f}});
    labels.add(3, PixelBox{0.0f, 0.0f, 3.0f, 3.0f});
    labels.add(4, PixelBox{0.0f, 0.0f, 4.0f, 4.0f}, Polygon{Point{4.0f, 4.0f}});

    labels.keep_if([&labels](std::size_t index)
    {
        return labels.classes()[index] % 2 == 0;
    });

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{2, 4}));
    ASSERT_EQ(labels.boxes().size(), 2u);
    ASSERT_EQ(labels.polygons().size(), 2u);
    expect_pixel_box(labels.boxes()[0], 0.0f, 0.0f, 2.0f, 2.0f);
    expect_pixel_box(labels.boxes()[1], 0.0f, 0.0f, 4.0f, 4.0f);
    ASSERT_EQ(labels.polygons()[0].size(), 2u);
    ASSERT_EQ(labels.polygons()[1].size(), 1u);
    expect_point(labels.polygons()[1][0], 4.0f, 4.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that to_pixel_labels converts every box and keeps classes and order. */
TEST(Unit_ImagingLabelSet, ToPixelLabelsKeepsClassesAndOrder)
{
    const FileLabels labels({7, 2, 7}, {NormalizedBox{0.5f, 0.25f, 0.5f, 0.25f},
                                        NormalizedBox{0.25f, 0.5f, 0.25f, 0.5f},
                                        NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}});

    const Labels pixels = to_pixel_labels(labels, K_SIZE);

    ASSERT_EQ(pixels.size(), 3u);
    EXPECT_EQ(pixels.classes(), (std::vector<std::int32_t>{7, 2, 7}));
    expect_pixel_box(pixels.boxes()[0], 160.0f, 60.0f, 480.0f, 180.0f);
    expect_pixel_box(pixels.boxes()[1], 80.0f, 120.0f, 240.0f, 360.0f);
    expect_pixel_box(pixels.boxes()[2], 0.0f, 0.0f, 640.0f, 480.0f);
}

/** @brief Tests that to_pixel_labels multiplies a vertex's x by the width and y by the height. */
TEST(Unit_ImagingLabelSet, ToPixelLabelsScalesVerticesByWidthAndHeight)
{
    FileLabels labels;
    labels.add(3, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    labels.add(4, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});

    const Labels pixels = to_pixel_labels(labels, K_SIZE);

    ASSERT_EQ(pixels.size(), 2u);
    ASSERT_EQ(pixels.polygons()[0].size(), 3u);
    expect_point(pixels.polygons()[0][0], 160.0f, 120.0f, K_PIXEL_TOLERANCE);
    expect_point(pixels.polygons()[0][1], 480.0f, 120.0f, K_PIXEL_TOLERANCE);
    expect_point(pixels.polygons()[0][2], 320.0f, 360.0f, K_PIXEL_TOLERANCE);
    EXPECT_TRUE(pixels.polygons()[1].empty());
}

/** @brief Tests that to_normalized_labels converts boxes and vertices and keeps the order. */
TEST(Unit_ImagingLabelSet, ToNormalizedLabelsKeepsClassesAndOrder)
{
    Labels labels;
    labels.add(4, PixelBox{80.0f, 120.0f, 240.0f, 360.0f},
               Polygon{Point{80.0f, 120.0f}, Point{240.0f, 360.0f}, Point{80.0f, 360.0f}});
    labels.add(9, PixelBox{160.0f, 60.0f, 480.0f, 180.0f});

    const FileLabels normalized = to_normalized_labels(labels, K_SIZE);

    ASSERT_EQ(normalized.size(), 2u);
    EXPECT_EQ(normalized.classes(), (std::vector<std::int32_t>{4, 9}));
    expect_normalized_box(normalized.boxes()[0], 0.25f, 0.5f, 0.25f, 0.5f);
    expect_normalized_box(normalized.boxes()[1], 0.5f, 0.25f, 0.5f, 0.25f);
    ASSERT_EQ(normalized.polygons()[0].size(), 3u);
    expect_point(normalized.polygons()[0][0], 0.125f, 0.25f, K_FRACTION_TOLERANCE);
    expect_point(normalized.polygons()[0][1], 0.375f, 0.75f, K_FRACTION_TOLERANCE);
    EXPECT_TRUE(normalized.polygons()[1].empty());
}

/** @brief Tests that both conversions of a background image give an empty set. */
TEST(Unit_ImagingLabelSet, LabelConversionsOfABackgroundImageAreEmpty)
{
    EXPECT_TRUE(to_pixel_labels(FileLabels{}, K_SIZE).is_empty());
    EXPECT_TRUE(to_normalized_labels(Labels{}, K_SIZE).is_empty());
}

/** @brief Tests that clip_labels clips every box, clamps every vertex and removes no object. */
TEST(Unit_ImagingLabelSet, ClipLabelsClipsBoxesAndVerticesAndRemovesNone)
{
    Labels labels;
    labels.add(1, PixelBox{-10.0f, 20.0f, 700.0f, 500.0f},
               Polygon{Point{-10.0f, 20.0f}, Point{700.0f, 30.0f}, Point{320.0f, 500.0f}});
    labels.add(2, PixelBox{700.0f, 10.0f, 800.0f, 20.0f});

    clip_labels(labels, K_SIZE);

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2}));
    expect_pixel_box(labels.boxes()[0], 0.0f, 20.0f, 640.0f, 480.0f);
    expect_pixel_box(labels.boxes()[1], 640.0f, 10.0f, 640.0f, 20.0f);
    ASSERT_EQ(labels.polygons()[0].size(), 3u);
    expect_point(labels.polygons()[0][0], 0.0f, 20.0f, K_PIXEL_TOLERANCE);
    expect_point(labels.polygons()[0][1], 640.0f, 30.0f, K_PIXEL_TOLERANCE);
    expect_point(labels.polygons()[0][2], 320.0f, 480.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that remove_empty_boxes drops zero-area objects with their polygons, in order. */
TEST(Unit_ImagingLabelSet, RemoveEmptyBoxesDropsZeroAreaAndKeepsOrderClassesAndPolygons)
{
    Labels labels;
    labels.add(10, PixelBox{0.0f, 0.0f, 5.0f, 5.0f});
    labels.add(11, PixelBox{640.0f, 10.0f, 640.0f, 20.0f}, Polygon{Point{640.0f, 10.0f}});
    labels.add(12, PixelBox{1.0f, 1.0f, 2.0f, 3.0f}, Polygon{Point{1.0f, 1.0f}, Point{2.0f, 3.0f}});
    labels.add(13, PixelBox{10.0f, 0.0f, 20.0f, 0.0f});
    labels.add(14, PixelBox{7.0f, 7.0f, 9.0f, 8.0f});

    remove_empty_boxes(labels);

    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{10, 12, 14}));
    expect_pixel_box(labels.boxes()[0], 0.0f, 0.0f, 5.0f, 5.0f);
    expect_pixel_box(labels.boxes()[1], 1.0f, 1.0f, 2.0f, 3.0f);
    expect_pixel_box(labels.boxes()[2], 7.0f, 7.0f, 9.0f, 8.0f);
    ASSERT_EQ(labels.polygons().size(), 3u);
    EXPECT_TRUE(labels.polygons()[0].empty());
    EXPECT_EQ(labels.polygons()[1].size(), 2u);
    EXPECT_TRUE(labels.polygons()[2].empty());
}

/** @brief Tests that remove_empty_boxes drops a box whose corners are in the wrong order. */
TEST(Unit_ImagingLabelSet, RemoveEmptyBoxesDropsABoxOfNegativeArea)
{
    Labels labels({3}, {PixelBox{5.0f, 0.0f, 2.0f, 4.0f}});

    remove_empty_boxes(labels);

    EXPECT_TRUE(labels.is_empty());
    EXPECT_TRUE(labels.classes().empty());
    EXPECT_TRUE(labels.polygons().empty());
}

/** @brief Tests that resample_polygons gives every polygon 1000 points and skips empty ones. */
TEST(Unit_ImagingLabelSet, ResamplePolygonsGivesEachPolygonAThousandPoints)
{
    FileLabels labels;
    labels.add(0, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    labels.add(1, NormalizedBox{0.5f, 0.5f, 0.2f, 0.2f});
    labels.add(2, NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}, make_diagonal(40));

    const FileLabels resampled = resample_polygons(labels);

    ASSERT_EQ(resampled.size(), 3u);
    EXPECT_EQ(resampled.classes(), labels.classes());
    EXPECT_EQ(resampled.polygons()[0].size(), K_POLYGON_POINTS);
    EXPECT_TRUE(resampled.polygons()[1].empty());
    EXPECT_EQ(resampled.polygons()[2].size(), K_POLYGON_POINTS);
    expect_point(resampled.polygons()[0][0], 0.25f, 0.25f, K_FRACTION_TOLERANCE);
    expect_normalized_box(resampled.boxes()[2], 0.5f, 0.5f, 1.0f, 1.0f);
}

/** @brief Tests that a polygon of over 1000 vertices raises the count to its length plus one. */
TEST(Unit_ImagingLabelSet, ResamplePolygonsUsesTheLongestPolygonPlusOneAboveAThousand)
{
    FileLabels labels;
    labels.add(0, NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}, make_diagonal(1200));
    labels.add(1, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});

    const FileLabels resampled = resample_polygons(labels);

    EXPECT_EQ(resampled.polygons()[0].size(), 1201u);
    EXPECT_EQ(resampled.polygons()[1].size(), 1201u);
}

/** @brief Tests that a set without any polygon comes back from resample_polygons unchanged. */
TEST(Unit_ImagingLabelSet, ResamplePolygonsLeavesASetWithoutPolygonsAlone)
{
    const FileLabels labels({6}, {NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}});

    const FileLabels resampled = resample_polygons(labels);

    ASSERT_EQ(resampled.size(), 1u);
    EXPECT_TRUE(resampled.polygons()[0].empty());
    expect_normalized_box(resampled.boxes()[0], 0.5f, 0.5f, 0.5f, 0.5f);
}

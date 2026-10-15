/**************************************************************************/
/* test_polygon_mask.cpp                                                  */
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
 * @file test_polygon_mask.cpp
 * @brief Tests the painting of polygons into a mask: the pixels, the overlap and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/polygon_mask.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::PixelBox;
using Vision::Imaging::Polygon;

namespace
{
    /** @brief Holds the size of every mask below. */
    constexpr ImageSize K_SIZE{4, 4};

    /** @brief Returns the rectangle between two corners as a polygon and as its box. */
    void add_rectangle(Labels& labels, std::int32_t class_index, float x1, float y1, float x2,
                       float y2)
    {
        labels.add(class_index, PixelBox{x1, y1, x2, y2},
                   Polygon{{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}});
    }

    /** @brief Returns the bytes of a mask, row by row. */
    std::vector<int> bytes_of(const Image& mask)
    {
        return std::vector<int>(mask.data(), mask.data() + mask.byte_count());
    }
} // namespace

/** @brief Tests that a polygon paints the pixels whose centres it holds, and no other. */
TEST(Small_DatasetsPolygonMask, APolygonPaintsThePixelsWhoseCentresItHolds)
{
    Labels labels;
    add_rectangle(labels, 2, 1.0f, 1.0f, 3.0f, 3.0f);

    Image mask;
    SA_EXPECT_OK(paint_polygon_mask(labels, K_SIZE, K_MASK_VOID, mask));

    ASSERT_EQ(mask.size(), K_SIZE);
    ASSERT_EQ(mask.channels(), 1);
    EXPECT_EQ(bytes_of(mask), (std::vector<int>{255, 255, 255, 255, 255, 2, 2, 255, 255, 2, 2,
                                                255, 255, 255, 255, 255}));
}

/** @brief Tests that the later of two objects that overlap paints over the earlier one. */
TEST(Small_DatasetsPolygonMask, TheLaterOfTwoObjectsWinsAnOverlap)
{
    Labels labels;
    add_rectangle(labels, 0, 0.0f, 0.0f, 3.0f, 2.0f);
    add_rectangle(labels, 1, 2.0f, 0.0f, 4.0f, 4.0f);

    Image mask;
    SA_EXPECT_OK(paint_polygon_mask(labels, K_SIZE, 7, mask));

    EXPECT_EQ(bytes_of(mask),
              (std::vector<int>{0, 0, 1, 1, 0, 0, 1, 1, 7, 7, 1, 1, 7, 7, 1, 1}));
}

/** @brief Tests that a set of no object is the background everywhere. */
TEST(Small_DatasetsPolygonMask, NoObjectIsTheBackgroundEverywhere)
{
    Image mask;
    SA_EXPECT_OK(paint_polygon_mask(Labels(), K_SIZE, 3, mask));

    EXPECT_EQ(bytes_of(mask), std::vector<int>(16, 3));
}

/** @brief Tests each refusal: an object with no polygon and a class no byte names. */
TEST(Small_DatasetsPolygonMask, RefusesAnObjectItCannotPaint)
{
    Image mask;

    Labels boxed;
    boxed.add(1, PixelBox{0.0f, 0.0f, 2.0f, 2.0f});
    SA_EXPECT_FAILURE(paint_polygon_mask(boxed, K_SIZE, K_MASK_VOID, mask),
                      StatusCode::InvalidDescription, "paint_polygon_mask", "object 0",
                      "no polygon");

    Labels wide;
    add_rectangle(wide, 255, 0.0f, 0.0f, 2.0f, 2.0f);
    SA_EXPECT_FAILURE(paint_polygon_mask(wide, K_SIZE, K_MASK_VOID, mask),
                      StatusCode::InvalidDescription, "the class 255", "0 to 254");
}

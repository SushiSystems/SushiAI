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
 * @file test_polygon_mask.cpp
 * @brief Tests the painting of polygons into a mask: the pixels, the overlap and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
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

    /** @brief Holds a way to add an object the mask cannot paint and the message it earns. */
    struct RefusalRow
    {
        std::string name;                 /**< The row's name. */
        void (*add)(Labels&);             /**< Adds the object to a label set. */
        std::string first;                /**< The first text the message holds. */
        std::string second;               /**< The second text the message holds. */
        std::string third;                /**< The third text the message holds; may be empty. */
    };

    /** @brief Runs paint_polygon_mask over each object it cannot paint. */
    class Small_DatasetsPolygonMaskRefusal : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Names a test after its row. */
    std::string refusal_row_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Adds a box of class 1 that has no polygon. */
    void add_boxed(Labels& labels)
    {
        labels.add(1, PixelBox{0.0f, 0.0f, 2.0f, 2.0f});
    }

    /** @brief Adds a square polygon of class 255. */
    void add_class_255(Labels& labels)
    {
        add_rectangle(labels, 255, 0.0f, 0.0f, 2.0f, 2.0f);
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

/** @brief Tests that an object the mask cannot paint is refused, naming the cause. */
TEST_P(Small_DatasetsPolygonMaskRefusal, RefusesAnObjectItCannotPaint)
{
    Labels labels;
    GetParam().add(labels);
    Image mask;

    SA_EXPECT_FAILURE(paint_polygon_mask(labels, K_SIZE, K_MASK_VOID, mask),
                      StatusCode::InvalidDescription, GetParam().first, GetParam().second, GetParam().third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsPolygonMaskRefusal,
    ::testing::Values(
        RefusalRow{"ObjectWithoutAPolygon", add_boxed, "paint_polygon_mask", "object 0", "no polygon"},
        RefusalRow{"ClassNoByteNames", add_class_255, "the class 255", "0 to 254", ""}),
    refusal_row_name);

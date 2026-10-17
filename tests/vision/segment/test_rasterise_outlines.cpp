/**************************************************************************/
/* test_rasterise_outlines.cpp                                            */
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
 * @file test_rasterise_outlines.cpp
 * @brief Tests the mask painted from class outlines.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>

using namespace SushiAI;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Polygon;
using Vision::Segment::ClassOutline;

namespace
{
    /** @brief Names the bytes of a mask, row after row. */
    using Bytes = std::vector<std::uint8_t>;

    /** @brief Names the parts of one outline. */
    using Parts = std::vector<Polygon>;

    /** @brief Holds the byte of a pixel no outline covers. */
    constexpr std::uint8_t K_BACKGROUND = 9;

    /** @brief Returns the bytes of @p mask, row after row. */
    Bytes bytes_of(const Image& mask)
    {
        return Bytes(mask.data(), mask.data() + mask.byte_count());
    }

    /** @brief Returns the rectangle with corners (x1, y1) and (x2, y2). */
    Polygon rectangle(float x1, float y1, float x2, float y2)
    {
        return Polygon{{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
    }

    /** @brief Views @p parts as the span an outline holds. */
    SushiRuntime::span<const Polygon> view(const Parts& parts)
    {
        return SushiRuntime::span<const Polygon>(parts.data(), parts.size());
    }

} // namespace

/** @brief Tests that a later outline overwrites an earlier one where they overlap. */
TEST(Small_SegmentRasterise, TheLaterOutlineWinsAnOverlap)
{
    const Parts first{rectangle(0.0f, 0.0f, 3.0f, 1.0f)};
    const Parts second{rectangle(2.0f, 0.0f, 4.0f, 1.0f)};
    const std::vector<ClassOutline> outlines{ClassOutline{1, view(first)},
                                             ClassOutline{2, view(second)}};
    Image mask;

    SA_EXPECT_OK(Vision::Segment::rasterise_outlines(
        SushiRuntime::span<const ClassOutline>(outlines.data(), outlines.size()),
        ImageSize{1, 5}, K_BACKGROUND, mask));

    EXPECT_EQ(mask.size(), (ImageSize{1, 5}));
    EXPECT_EQ(mask.channels(), 1);
    EXPECT_EQ(bytes_of(mask), (Bytes{1, 1, 2, 2, 9}));
}

/** @brief Tests that rasterise_outlines refuses a bad size and names the outline that failed. */
TEST(Small_SegmentRasterise, RasteriseRefusesABadSizeAndNamesTheOutline)
{
    const Parts good{rectangle(0.0f, 0.0f, 1.0f, 1.0f)};
    const Parts bad{Polygon{{std::numeric_limits<float>::infinity(), 0.0f}}};
    const std::vector<ClassOutline> outlines{ClassOutline{1, view(good)},
                                             ClassOutline{2, view(bad)}};
    const SushiRuntime::span<const ClassOutline> span(outlines.data(), outlines.size());
    Image mask;

    SA_EXPECT_FAILURE(
        Vision::Segment::rasterise_outlines(span, ImageSize{0, 4}, K_BACKGROUND, mask),
        StatusCode::InvalidDescription, "rasterise_outlines", "0 by 4");
    SA_EXPECT_FAILURE(
        Vision::Segment::rasterise_outlines(span, ImageSize{2, 2}, K_BACKGROUND, mask),
        StatusCode::InvalidDescription, "outline 1", "vertex 0 of part 0");
}

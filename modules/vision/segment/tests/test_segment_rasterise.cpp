/**************************************************************************/
/* test_segment_rasterise.cpp                                             */
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
 * @file test_segment_rasterise.cpp
 * @brief Tests the outline fill and the mask painted from class outlines on small masks.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>
#include <common/test_status.hpp>

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

    /** @brief Returns a mask of @p size filled with K_BACKGROUND. */
    Image blank(ImageSize size)
    {
        Image mask(size, 1);
        mask.fill(K_BACKGROUND);

        return mask;
    }
} // namespace

/** @brief Tests that a rectangle on pixel edges fills exactly the pixels it bounds. */
TEST(Unit_SegmentRasterise, ARectangleOnPixelEdgesFillsThePixelsItBounds)
{
    Image mask = blank(ImageSize{4, 5});
    const Parts parts{rectangle(1.0f, 1.0f, 4.0f, 3.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{9, 9, 9, 9, 9,
                                      9, 1, 1, 1, 9,
                                      9, 1, 1, 1, 9,
                                      9, 9, 9, 9, 9}));
}

/** @brief Tests that a pixel is filled only when its centre is inside the outline. */
TEST(Unit_SegmentRasterise, APixelIsFilledWhenItsCentreIsInside)
{
    // The rectangle spans x in [0.6, 2.4]: it holds the centre 1.5 and neither 0.5 nor 2.5.
    Image mask = blank(ImageSize{1, 3});
    const Parts parts{rectangle(0.6f, 0.0f, 2.4f, 1.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{9, 1, 9}));
}

/** @brief Tests that two rectangles sharing an edge through pixel centres fill no pixel twice. */
TEST(Unit_SegmentRasterise, ACentreOnASharedEdgeBelongsToOneSide)
{
    // The shared edge x = 1.5 passes through the centre of column 1; the right side takes it.
    Image mask = blank(ImageSize{1, 3});
    const Parts left{rectangle(0.0f, 0.0f, 1.5f, 1.0f)};
    const Parts right{rectangle(1.5f, 0.0f, 3.0f, 1.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(left), 1, mask));
    EXPECT_EQ(bytes_of(mask), (Bytes{1, 9, 9}));

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(right), 2, mask));
    EXPECT_EQ(bytes_of(mask), (Bytes{1, 2, 2}));
}

/** @brief Tests that a triangle fills the centres under its slanted edge. */
TEST(Unit_SegmentRasterise, ATriangleFillsTheCentresUnderItsSlope)
{
    // Row y is filled where x + 0.5 is below 4 - (y + 0.5): 3, 2, 1 and 0 columns.
    Image mask = blank(ImageSize{4, 4});
    const Parts parts{Polygon{{0.0f, 0.0f}, {4.0f, 0.0f}, {0.0f, 4.0f}}};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 1, 1, 9,
                                      1, 1, 9, 9,
                                      1, 9, 9, 9,
                                      9, 9, 9, 9}));
}

/** @brief Tests that both parts of a two-part outline are filled. */
TEST(Unit_SegmentRasterise, EveryPartOfAnOutlineIsFilled)
{
    Image mask = blank(ImageSize{1, 5});
    const Parts parts{rectangle(0.0f, 0.0f, 1.0f, 1.0f), rectangle(3.0f, 0.0f, 5.0f, 1.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 9, 9, 1, 1}));
}

/** @brief Tests that a part inside another cuts a hole, by the even-odd rule. */
TEST(Unit_SegmentRasterise, APartInsideAnotherCutsAHole)
{
    Image mask = blank(ImageSize{3, 3});
    const Parts parts{rectangle(0.0f, 0.0f, 3.0f, 3.0f), rectangle(1.0f, 1.0f, 2.0f, 2.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 1, 1, 1, 9, 1, 1, 1, 1}));
}

/** @brief Tests that an outline reaching past the mask is clipped to it. */
TEST(Unit_SegmentRasterise, AnOutlineIsClippedToTheMask)
{
    Image mask = blank(ImageSize{2, 2});
    const Parts parts{rectangle(-5.0f, -5.0f, 1.0f, 50.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 9, 1, 9}));
}

/** @brief Tests that an outline of no part, or of a part with two vertices, fills nothing. */
TEST(Unit_SegmentRasterise, AnOutlineWithoutAreaFillsNothing)
{
    Image mask = blank(ImageSize{2, 2});
    const Parts none{};
    const Parts line{Polygon{{0.0f, 0.0f}, {2.0f, 2.0f}}};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(none), 1, mask));
    SA_EXPECT_OK(Vision::Segment::fill_outline(view(line), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{9, 9, 9, 9}));
}

/** @brief Tests that fill_outline refuses an unsized mask, three channels and a NaN vertex. */
TEST(Unit_SegmentRasterise, FillRefusesABadMaskAndAVertexThatIsNotFinite)
{
    Image unsized;
    Image colour(ImageSize{2, 2}, 3);
    Image mask = blank(ImageSize{2, 2});
    const Parts square{rectangle(0.0f, 0.0f, 2.0f, 2.0f)};
    const Parts broken{rectangle(0.0f, 0.0f, 2.0f, 2.0f),
                       Polygon{{0.0f, 0.0f}, {std::numeric_limits<float>::quiet_NaN(), 1.0f}}};

    SA_EXPECT_FAILURE(Vision::Segment::fill_outline(view(square), 1, unsized),
                      StatusCode::InvalidDescription, "fill_outline", "0 by 0");
    SA_EXPECT_FAILURE(Vision::Segment::fill_outline(view(square), 1, colour),
                      StatusCode::InvalidDescription, "3 channels");
    SA_EXPECT_FAILURE(Vision::Segment::fill_outline(view(broken), 1, mask),
                      StatusCode::InvalidDescription, "vertex 1 of part 1");

    // The refusal comes before the first pixel is written.
    EXPECT_EQ(bytes_of(mask), (Bytes{9, 9, 9, 9}));
}

/** @brief Tests that a later outline overwrites an earlier one where they overlap. */
TEST(Unit_SegmentRasterise, TheLaterOutlineWinsAnOverlap)
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
TEST(Unit_SegmentRasterise, RasteriseRefusesABadSizeAndNamesTheOutline)
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

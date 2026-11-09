/**************************************************************************/
/* test_fill_outline.cpp                                                  */
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
 * @file test_fill_outline.cpp
 * @brief Tests the outline fill on small masks.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>
#include <support/test_names.hpp>

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

    /** @brief Returns an unsized mask. */
    Image make_unsized()
    {
        return Image();
    }

    /** @brief Returns a 2 by 2 mask of three channels. */
    Image make_colour()
    {
        return Image(ImageSize{2, 2}, 3);
    }

    /** @brief Names an outline, the mask it fills and the bytes the fill leaves. */
    struct FillRow
    {
        std::string_view name; /**< The name of the row. */
        ImageSize size;        /**< The size of the mask, filled with K_BACKGROUND. */
        Parts parts;           /**< The parts of the outline, filled with class 1. */
        Bytes expected;        /**< The bytes of the mask after the fill. */
    };

    /** @brief Gives each outline that fills pixels to a test. */
    class Small_SegmentFill : public ::testing::TestWithParam<FillRow> {};

    /** @brief Gives each outline without area to a test. */
    class Small_SegmentFillNothing : public ::testing::TestWithParam<Parts> {};

    /** @brief Names a mask a fill refuses, how to make it and what the refusal holds. */
    struct BadMaskRow
    {
        std::string_view name;   /**< The name of the row. */
        Image (*make)();         /**< Makes the mask. */
        std::string_view first;  /**< A text the refusal holds. */
        std::string_view second; /**< Another text the refusal holds, or empty. */
    };

    /** @brief Gives each mask a fill refuses to a test. */
    class Small_SegmentFillBadMask : public ::testing::TestWithParam<BadMaskRow> {};

    /** @brief Names a fill row in the name of its test. */
    std::string fill_name(const ::testing::TestParamInfo<FillRow>& info)
    {
        return Test::parameter_name(info.param.name);
    }

    /** @brief Names an outline without area in the name of its test. */
    std::string empty_name(const ::testing::TestParamInfo<Parts>& info)
    {
        return info.param.empty() ? "NoPart" : "TwoVertices";
    }

    /** @brief Names a bad mask in the name of its test. */
    std::string bad_mask_name(const ::testing::TestParamInfo<BadMaskRow>& info)
    {
        return Test::parameter_name(info.param.name);
    }
} // namespace

/**
 * @brief Tests that the fill writes each pixel whose centre the outline holds: a rectangle on
 *        pixel edges, a rectangle that holds one centre of three and a slanted edge.
 */
TEST_P(Small_SegmentFill, WritesEachPixelWhoseCentreTheOutlineHolds)
{
    const FillRow& row = GetParam();
    Image mask = blank(row.size);

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(row.parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Outlines, Small_SegmentFill,
    ::testing::Values(
        FillRow{"ARectangleOnPixelEdges", ImageSize{4, 5}, {rectangle(1.0f, 1.0f, 4.0f, 3.0f)},
                {9, 9, 9, 9, 9,
                 9, 1, 1, 1, 9,
                 9, 1, 1, 1, 9,
                 9, 9, 9, 9, 9}},
        // The rectangle spans x in [0.6, 2.4]: it holds the centre 1.5 and neither 0.5 nor 2.5.
        FillRow{"APixelWhoseCentreIsInside", ImageSize{1, 3}, {rectangle(0.6f, 0.0f, 2.4f, 1.0f)},
                {9, 1, 9}},
        // Row y is filled where x + 0.5 is below 4 - (y + 0.5): 3, 2, 1 and 0 columns.
        FillRow{"ATriangleUnderItsSlope", ImageSize{4, 4},
                {Polygon{{0.0f, 0.0f}, {4.0f, 0.0f}, {0.0f, 4.0f}}},
                {1, 1, 1, 9,
                 1, 1, 9, 9,
                 1, 9, 9, 9,
                 9, 9, 9, 9}}),
    fill_name);

/**
 * @brief Tests that two rectangles sharing an edge through a pixel centre give that centre to
 *        one side only. The shared edge x = 1.5 passes through the centre of column 1. The right
 *        outline comes first, so a left side that also took the centre would overwrite it: the
 *        mask is 1, 2, 2 only when the right side takes the centre and the left side does not.
 */
TEST(Small_SegmentRasterise, ACentreOnASharedEdgeBelongsToOneSide)
{
    const Parts left{rectangle(0.0f, 0.0f, 1.5f, 1.0f)};
    const Parts right{rectangle(1.5f, 0.0f, 3.0f, 1.0f)};
    const std::vector<ClassOutline> outlines{ClassOutline{2, view(right)},
                                             ClassOutline{1, view(left)}};
    Image mask;

    SA_EXPECT_OK(Vision::Segment::rasterise_outlines(
        SushiRuntime::span<const ClassOutline>(outlines.data(), outlines.size()),
        ImageSize{1, 3}, K_BACKGROUND, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 2, 2}));
}

/** @brief Tests that both parts of a two-part outline are filled. */
TEST(Small_SegmentRasterise, EveryPartOfAnOutlineIsFilled)
{
    Image mask = blank(ImageSize{1, 5});
    const Parts parts{rectangle(0.0f, 0.0f, 1.0f, 1.0f), rectangle(3.0f, 0.0f, 5.0f, 1.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 9, 9, 1, 1}));
}

/** @brief Tests that a part inside another cuts a hole, by the even-odd rule. */
TEST(Small_SegmentRasterise, APartInsideAnotherCutsAHole)
{
    Image mask = blank(ImageSize{3, 3});
    const Parts parts{rectangle(0.0f, 0.0f, 3.0f, 3.0f), rectangle(1.0f, 1.0f, 2.0f, 2.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 1, 1, 1, 9, 1, 1, 1, 1}));
}

/** @brief Tests that an outline reaching past the mask is clipped to it. */
TEST(Small_SegmentRasterise, AnOutlineIsClippedToTheMask)
{
    Image mask = blank(ImageSize{2, 2});
    const Parts parts{rectangle(-5.0f, -5.0f, 1.0f, 50.0f)};

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(parts), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{1, 9, 1, 9}));
}

/** @brief Tests that an outline of no part, or of a part with two vertices, fills nothing. */
TEST_P(Small_SegmentFillNothing, AnOutlineWithoutAreaFillsNothing)
{
    Image mask = blank(ImageSize{2, 2});

    SA_EXPECT_OK(Vision::Segment::fill_outline(view(GetParam()), 1, mask));

    EXPECT_EQ(bytes_of(mask), (Bytes{9, 9, 9, 9}));
}

INSTANTIATE_TEST_SUITE_P(Outlines, Small_SegmentFillNothing,
                         ::testing::Values(Parts{},
                                           Parts{Polygon{{0.0f, 0.0f}, {2.0f, 2.0f}}}),
                         empty_name);

/** @brief Tests that fill_outline refuses an unsized mask and a mask of three channels. */
TEST_P(Small_SegmentFillBadMask, FillRefusesABadMask)
{
    const BadMaskRow& row = GetParam();
    Image mask = row.make();
    const Parts square{rectangle(0.0f, 0.0f, 2.0f, 2.0f)};

    SA_EXPECT_FAILURE(Vision::Segment::fill_outline(view(square), 1, mask),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Masks, Small_SegmentFillBadMask,
    ::testing::Values(BadMaskRow{"Unsized", &make_unsized, "fill_outline", "0 by 0"},
                      BadMaskRow{"ThreeChannels", &make_colour, "3 channels", ""}),
    bad_mask_name);

/** @brief Tests that fill_outline refuses a vertex that is not finite before it writes a pixel. */
TEST(Small_SegmentRasterise, FillRefusesAVertexThatIsNotFiniteAndWritesNoPixelFirst)
{
    Image mask = blank(ImageSize{2, 2});
    const Parts broken{rectangle(0.0f, 0.0f, 2.0f, 2.0f),
                       Polygon{{0.0f, 0.0f}, {std::numeric_limits<float>::quiet_NaN(), 1.0f}}};

    SA_EXPECT_FAILURE(Vision::Segment::fill_outline(view(broken), 1, mask),
                      StatusCode::InvalidDescription, "vertex 1 of part 1");

    // The refusal comes before the first pixel is written.
    EXPECT_EQ(bytes_of(mask), (Bytes{9, 9, 9, 9}));
}

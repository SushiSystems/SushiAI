/**************************************************************************/
/* test_imaging_mosaic.cpp                                                */
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
 * @file test_imaging_mosaic.cpp
 * @brief Tests the mosaic's draws, its placement rectangles, its canvas and its joined labels.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/imaging/dataset_partners.hpp>
#include <SushiAI/imaging/mosaic.hpp>

#include "../../common/test_throws.hpp"

using namespace SushiAI;
using namespace SushiAI::Imaging;

namespace
{
    /** @brief Holds the training size of the small mosaics below; the canvas is 16 by 16. */
    constexpr std::int32_t K_SIDE = 8;

    /** @brief Holds the absolute tolerance, in pixels, of a box edge made of small integers. */
    constexpr float K_BOX_TOLERANCE = 1e-4f;

    /** @brief Holds the four patch sizes of a mosaic, patch 0 first. */
    using PatchSizes = std::array<ImageSize, K_MOSAIC_PATCHES>;

    /** @brief Holds the four hand-computed placements a plan is compared with. */
    using Placements = std::array<MosaicPlacement, K_MOSAIC_PATCHES>;

    /** @brief Holds a centre that cuts patch 0 on the left and patches 2 and 3 at the bottom. */
    constexpr std::int32_t K_LOW_CENTER_X = 5;

    /** @brief Holds the row of that centre. */
    constexpr std::int32_t K_LOW_CENTER_Y = 11;

    /** @brief Holds the patch sizes used with the low centre. */
    constexpr PatchSizes K_LOW_SIZES = {ImageSize{8, 8}, ImageSize{6, 8}, ImageSize{8, 4},
                                        ImageSize{8, 8}};

    /** @brief Holds data/augment.py:525-538 worked by hand for the low centre. */
    constexpr Placements K_LOW_PLACEMENTS = {
        MosaicPlacement{0, 3, 5, 11, 3, 0, 8, 8, -3, 3},
        MosaicPlacement{5, 5, 13, 11, 0, 0, 8, 6, 5, 5},
        MosaicPlacement{1, 11, 5, 16, 0, 0, 4, 5, 1, 11},
        MosaicPlacement{5, 11, 13, 16, 0, 0, 8, 5, 5, 11}};

    /** @brief Holds a centre that cuts patches 0 and 1 at the top and 1 and 3 on the right. */
    constexpr std::int32_t K_HIGH_CENTER_X = 10;

    /** @brief Holds the row of that centre. */
    constexpr std::int32_t K_HIGH_CENTER_Y = 6;

    /** @brief Holds the patch sizes used with the high centre. */
    constexpr PatchSizes K_HIGH_SIZES = {ImageSize{8, 8}, ImageSize{8, 8}, ImageSize{8, 8},
                                         ImageSize{8, 8}};

    /** @brief Holds data/augment.py:525-538 worked by hand for the high centre. */
    constexpr Placements K_HIGH_PLACEMENTS = {
        MosaicPlacement{2, 0, 10, 6, 0, 2, 8, 8, 2, -2},
        MosaicPlacement{10, 0, 16, 6, 0, 2, 6, 8, 10, -2},
        MosaicPlacement{2, 6, 10, 14, 0, 0, 8, 8, 2, 6},
        MosaicPlacement{10, 6, 16, 14, 0, 0, 6, 8, 10, 6}};

    /** @brief Returns an enabled draw with the given centre and partners 0, 0, 0. */
    MosaicDraw make_draw(std::int32_t center_x, std::int32_t center_y)
    {
        MosaicDraw draw;
        draw.is_enabled = true;
        draw.center_x = center_x;
        draw.center_y = center_y;
        return draw;
    }

    /** @brief Returns a three-channel image with every pixel the same colour. */
    Image make_constant(ImageSize size, std::uint8_t red, std::uint8_t green, std::uint8_t blue)
    {
        Image image(size, 3);
        for (std::int32_t y = 0; y < size.height; ++y)
        {
            std::uint8_t* row = image.row(y);
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                row[3 * x] = red;
                row[3 * x + 1] = green;
                row[3 * x + 2] = blue;
            }
        }

        return image;
    }

    /** @brief Returns a three-channel image whose every byte depends on its row and column. */
    Image make_textured(ImageSize size, std::int32_t salt)
    {
        Image image(size, 3);
        for (std::int32_t y = 0; y < size.height; ++y)
        {
            std::uint8_t* row = image.row(y);
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                for (std::int32_t c = 0; c < 3; ++c)
                    row[3 * x + c] = static_cast<std::uint8_t>(31 * y + 7 * x + 53 * c + salt);
            }
        }

        return image;
    }

    /** @brief Returns the views of four images in patch order. */
    std::array<ImageView, K_MOSAIC_PATCHES> view_all(
        const std::array<Image, K_MOSAIC_PATCHES>& images)
    {
        return {images[0].view(), images[1].view(), images[2].view(), images[3].view()};
    }

    /** @brief Counts the canvas bytes that differ from the patch byte the placements name. */
    std::size_t count_wrong_bytes(const Image& canvas,
                                  const std::array<Image, K_MOSAIC_PATCHES>& patches,
                                  const Placements& placements, std::uint8_t fill_value)
    {
        std::size_t wrong = 0;
        for (std::int32_t y = 0; y < canvas.height(); ++y)
        {
            for (std::int32_t x = 0; x < canvas.width(); ++x)
            {
                for (std::int32_t c = 0; c < canvas.channels(); ++c)
                {
                    std::uint8_t expected = fill_value;
                    for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
                    {
                        const MosaicPlacement& p = placements[i];
                        if (x >= p.canvas_x1 && x < p.canvas_x2 && y >= p.canvas_y1 &&
                            y < p.canvas_y2)
                            expected = patches[i].row(y - p.offset_y)[3 * (x - p.offset_x) + c];
                    }

                    if (canvas.row(y)[3 * x + c] != expected)
                        ++wrong;
                }
            }
        }

        return wrong;
    }

    /** @brief Checks every field of four placements against the hand-computed ones. */
    void expect_placements(const MosaicPlan& plan, const Placements& expected)
    {
        for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
        {
            SCOPED_TRACE(::testing::Message() << "patch " << i);
            const MosaicPlacement& actual = plan.placements[i];
            EXPECT_EQ(actual.canvas_x1, expected[i].canvas_x1);
            EXPECT_EQ(actual.canvas_y1, expected[i].canvas_y1);
            EXPECT_EQ(actual.canvas_x2, expected[i].canvas_x2);
            EXPECT_EQ(actual.canvas_y2, expected[i].canvas_y2);
            EXPECT_EQ(actual.patch_x1, expected[i].patch_x1);
            EXPECT_EQ(actual.patch_y1, expected[i].patch_y1);
            EXPECT_EQ(actual.patch_x2, expected[i].patch_x2);
            EXPECT_EQ(actual.patch_y2, expected[i].patch_y2);
            EXPECT_EQ(actual.offset_x, expected[i].offset_x);
            EXPECT_EQ(actual.offset_y, expected[i].offset_y);
        }
    }

    /** @brief Checks the four edges of one box within K_BOX_TOLERANCE. */
    void expect_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_NEAR(actual.x1, x1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y1, y1, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.x2, x2, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y2, y2, K_BOX_TOLERANCE);
    }

    /** @brief Checks one vertex within K_BOX_TOLERANCE. */
    void expect_point(const Point& actual, float x, float y)
    {
        EXPECT_NEAR(actual.x, x, K_BOX_TOLERANCE);
        EXPECT_NEAR(actual.y, y, K_BOX_TOLERANCE);
    }

    /** @brief Gives fixed partners, makes no draw and records every index it is asked for. */
    class RecordingPartners final : public PartnerSource
    {
        public:
            /** @brief Records the index and returns the partners 4, 5 and 6. */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override
            {
                static_cast<void>(random);
                indices_.push_back(index);
                return MosaicPartners{4, 5, 6};
            }

            /** @brief Returns the indices choose was called with, in call order. */
            [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept
            {
                return indices_;
            }

        private:
            /// @brief Holds the indices choose was called with.
            std::vector<std::size_t> indices_;
    };

    /** @brief Returns the three bytes of one canvas pixel. */
    std::array<std::uint8_t, 3> pixel_at(const Image& image, std::int32_t x, std::int32_t y)
    {
        const std::uint8_t* row = image.row(y);
        return {row[3 * x], row[3 * x + 1], row[3 * x + 2]};
    }
} // namespace

TEST(Unit_ImagingMosaic, SettingsDefaultToTheReferenceProbabilityAndFill)
{
    const MosaicSettings settings;

    EXPECT_NEAR(settings.probability, 1.0f, 1e-7f);
    EXPECT_EQ(settings.fill_value, 114);
    EXPECT_EQ(K_MOSAIC_PATCHES, 4u);
}

TEST(Unit_ImagingMosaic, DrawMakesPartnersThenGateThenRowThenColumn)
{
    const std::int32_t side = 640;
    const std::size_t pool = 1000;
    DatasetPartners partners(pool);
    Random random(20261007u);
    Random replica = random;

    const MosaicDraw draw = draw_mosaic(MosaicSettings{}, side, 0, partners, random);

    const std::uint32_t first = replica.index(static_cast<std::uint32_t>(pool));
    const std::uint32_t second = replica.index(static_cast<std::uint32_t>(pool));
    const std::uint32_t third = replica.index(static_cast<std::uint32_t>(pool));
    const bool gate = replica.bernoulli(1.0f);
    const std::int32_t center_y = static_cast<std::int32_t>(replica.uniform(320.0f, 960.0f));
    const std::int32_t center_x = static_cast<std::int32_t>(replica.uniform(320.0f, 960.0f));

    EXPECT_TRUE(gate);
    EXPECT_TRUE(draw.is_enabled);
    EXPECT_EQ(draw.partners[0], first);
    EXPECT_EQ(draw.partners[1], second);
    EXPECT_EQ(draw.partners[2], third);
    EXPECT_EQ(draw.center_y, center_y);
    EXPECT_EQ(draw.center_x, center_x);
    EXPECT_TRUE(random == replica) << "draw_mosaic made a number of draws other than six";
}

TEST(Unit_ImagingMosaic, DisabledDrawMakesFourDrawsAndLeavesTheFieldsZero)
{
    MosaicSettings settings;
    settings.probability = 0.0f;
    DatasetPartners partners(50);
    Random random(7u);
    Random replica = random;

    const MosaicDraw draw = draw_mosaic(settings, 640, 0, partners, random);

    for (std::int32_t i = 0; i < 4; ++i)
        static_cast<void>(replica.next_u64());

    EXPECT_FALSE(draw.is_enabled);
    EXPECT_EQ(draw.partners[0], 0u);
    EXPECT_EQ(draw.partners[1], 0u);
    EXPECT_EQ(draw.partners[2], 0u);
    EXPECT_EQ(draw.center_x, 0);
    EXPECT_EQ(draw.center_y, 0);
    EXPECT_TRUE(random == replica) << "a closed gate made a number of draws other than four";
}

TEST(Unit_ImagingMosaic, CentreStaysInTheReferenceRangeOverManySeeds)
{
    // int(uniform(320, 960)) at s = 640 (data/augment.py:480, 520) is a value in 320 to 959.
    const std::int32_t side = 640;
    const std::uint64_t seeds = 5000;
    std::int32_t smallest = 2 * side;
    std::int32_t largest = 0;
    double sum = 0.0;

    DatasetPartners partners(100);
    for (std::uint64_t seed = 0; seed < seeds; ++seed)
    {
        Random random(seed);
        const MosaicDraw draw = draw_mosaic(MosaicSettings{}, side, 0, partners, random);

        ASSERT_TRUE(draw.is_enabled) << "seed " << seed;
        for (const std::int32_t center : {draw.center_x, draw.center_y})
        {
            ASSERT_GE(center, 320) << "seed " << seed;
            ASSERT_LE(center, 959) << "seed " << seed;
            smallest = std::min(smallest, center);
            largest = std::max(largest, center);
            sum += center;
        }
    }

    // 10000 values of standard deviation 185 give a mean within 1.85 of 639.5; 10 is 5 sigma.
    EXPECT_NEAR(sum / static_cast<double>(2 * seeds), 639.5, 10.0);
    EXPECT_LE(smallest, 330) << "the low end of the range is never drawn";
    EXPECT_GE(largest, 949) << "the high end of the range is never drawn";
}

TEST(Unit_ImagingMosaic, CentreOfAnOddSizeUsesTheFlooredBorder)
{
    // -5 // 2 is -3 in the reference, so the range is int(uniform(3, 7)): 3 to 6.
    std::array<bool, 8> seen{};
    DatasetPartners partners(3);
    for (std::uint64_t seed = 0; seed < 400; ++seed)
    {
        Random random(seed);
        const MosaicDraw draw = draw_mosaic(MosaicSettings{}, 5, 0, partners, random);

        ASSERT_GE(draw.center_x, 3);
        ASSERT_LE(draw.center_x, 6);
        ASSERT_GE(draw.center_y, 3);
        ASSERT_LE(draw.center_y, 6);
        seen[static_cast<std::size_t>(draw.center_x)] = true;
    }

    EXPECT_TRUE(seen[3] && seen[4] && seen[5] && seen[6]);
}

TEST(Unit_ImagingMosaic, PartnersCoverThePoolAndNeverLeaveIt)
{
    const std::size_t pool = 7;
    std::array<bool, 7> seen{};
    DatasetPartners partners(pool);
    for (std::uint64_t seed = 0; seed < 300; ++seed)
    {
        Random random(seed);
        const MosaicDraw draw = draw_mosaic(MosaicSettings{}, 640, 0, partners, random);

        for (const std::size_t partner : draw.partners)
        {
            ASSERT_LT(partner, pool) << "seed " << seed;
            seen[partner] = true;
        }
    }

    for (std::size_t i = 0; i < pool; ++i)
        EXPECT_TRUE(seen[i]) << "index " << i << " was never a partner";
}

TEST(Unit_ImagingMosaic, GateOpensWithItsProbability)
{
    MosaicSettings settings;
    settings.probability = 0.25f;
    const std::uint64_t seeds = 4000;
    std::uint64_t enabled = 0;
    DatasetPartners partners(10);

    for (std::uint64_t seed = 0; seed < seeds; ++seed)
    {
        Random random(seed);
        if (draw_mosaic(settings, 640, 0, partners, random).is_enabled)
            ++enabled;
    }

    // The standard deviation of the fraction is 0.0068; 0.04 is almost 6 sigma.
    EXPECT_NEAR(static_cast<double>(enabled) / static_cast<double>(seeds), 0.25, 0.04);
}

TEST(Unit_ImagingMosaic, PlanCutsPatchesAtTheLeftAndBottomEdges)
{
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);

    EXPECT_EQ(plan.image_size, K_SIDE);
    expect_placements(plan, K_LOW_PLACEMENTS);
}

TEST(Unit_ImagingMosaic, PlanCutsPatchesAtTheTopAndRightEdges)
{
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_HIGH_CENTER_X, K_HIGH_CENTER_Y), K_SIDE, K_HIGH_SIZES);

    expect_placements(plan, K_HIGH_PLACEMENTS);
}

TEST(Unit_ImagingMosaic, PlanCarriesTheFillValueOfTheSettings)
{
    MosaicSettings settings;
    settings.fill_value = 9;

    const MosaicPlan plan = plan_mosaic(
        settings, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);

    EXPECT_EQ(plan.fill_value, 9);
    EXPECT_EQ(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE, K_LOW_SIZES).fill_value, 114);
}

TEST(Unit_ImagingMosaic, CanvasIsTwiceTheSizeAndFilledWhereNoPatchLands)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 255, 0, 0), make_constant(K_LOW_SIZES[1], 0, 255, 0),
        make_constant(K_LOW_SIZES[2], 0, 0, 255), make_constant(K_LOW_SIZES[3], 255, 255, 0)};
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    Image canvas;

    apply_mosaic(view_all(patches), plan, canvas);

    ASSERT_EQ(canvas.height(), 2 * K_SIDE);
    ASSERT_EQ(canvas.width(), 2 * K_SIDE);
    ASSERT_EQ(canvas.channels(), 3);
    const std::array<std::uint8_t, 3> fill = {114, 114, 114};
    EXPECT_EQ(pixel_at(canvas, 0, 0), fill);
    EXPECT_EQ(pixel_at(canvas, 15, 15), fill);
    EXPECT_EQ(pixel_at(canvas, 7, 4), fill) << "the row above the shorter patch 1";
    EXPECT_EQ(pixel_at(canvas, 0, 12), fill) << "the column left of the narrower patch 2";
}

TEST(Unit_ImagingMosaic, EachColourLandsInItsQuadrantAroundTheCentre)
{
    const std::array<std::uint8_t, 3> red = {255, 0, 0};
    const std::array<std::uint8_t, 3> green = {0, 255, 0};
    const std::array<std::uint8_t, 3> blue = {0, 0, 255};
    const std::array<std::uint8_t, 3> yellow = {255, 255, 0};
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 255, 0, 0), make_constant(K_LOW_SIZES[1], 0, 255, 0),
        make_constant(K_LOW_SIZES[2], 0, 0, 255), make_constant(K_LOW_SIZES[3], 255, 255, 0)};
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    Image canvas;

    apply_mosaic(view_all(patches), plan, canvas);

    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(pixel_at(canvas, K_LOW_CENTER_X - 1, K_LOW_CENTER_Y - 1), red);
    EXPECT_EQ(pixel_at(canvas, K_LOW_CENTER_X, K_LOW_CENTER_Y - 1), green);
    EXPECT_EQ(pixel_at(canvas, K_LOW_CENTER_X - 1, K_LOW_CENTER_Y), blue);
    EXPECT_EQ(pixel_at(canvas, K_LOW_CENTER_X, K_LOW_CENTER_Y), yellow);
    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_LOW_PLACEMENTS, 114), 0u);
}

TEST(Unit_ImagingMosaic, CutPatchesKeepThePixelsNearestTheCentre)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_textured(K_HIGH_SIZES[0], 1), make_textured(K_HIGH_SIZES[1], 60),
        make_textured(K_HIGH_SIZES[2], 120), make_textured(K_HIGH_SIZES[3], 180)};
    MosaicSettings settings;
    settings.fill_value = 3;
    const MosaicPlan plan = plan_mosaic(
        settings, make_draw(K_HIGH_CENTER_X, K_HIGH_CENTER_Y), K_SIDE, K_HIGH_SIZES);
    Image canvas;

    apply_mosaic(view_all(patches), plan, canvas);

    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_HIGH_PLACEMENTS, 3), 0u);
}

TEST(Unit_ImagingMosaic, ApplyOverwritesWhatTheDestinationHeld)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_textured(K_LOW_SIZES[0], 5), make_textured(K_LOW_SIZES[1], 6),
        make_textured(K_LOW_SIZES[2], 7), make_textured(K_LOW_SIZES[3], 8)};
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    Image canvas(ImageSize{40, 40}, 3);
    canvas.fill(200);

    apply_mosaic(view_all(patches), plan, canvas);

    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_LOW_PLACEMENTS, 114), 0u);
}

TEST(Unit_ImagingMosaic, LabelsAreShiftedJoinedInPatchOrderClippedAndPruned)
{
    MosaicPlan plan;
    plan.image_size = K_SIDE;
    plan.placements = K_LOW_PLACEMENTS;
    std::array<Labels, K_MOSAIC_PATCHES> labels;
    labels[0] = Labels({3, 1, 2}, {PixelBox{4.0f, 1.0f, 7.0f, 5.0f},
                                   PixelBox{0.0f, 2.0f, 3.0f, 6.0f},
                                   PixelBox{1.0f, 0.0f, 5.0f, 2.0f}});
    labels[1] = Labels({7}, {PixelBox{1.0f, 1.0f, 2.0f, 3.0f}});
    labels[3] = Labels({5}, {PixelBox{2.0f, 2.0f, 6.0f, 7.0f}});

    const Labels joined = apply_mosaic(labels, plan);

    // Patch 0's second box lies wholly left of the canvas and has no area after the clip.
    ASSERT_EQ(joined.size(), 4u);
    EXPECT_EQ(joined.classes(), (std::vector<std::int32_t>{3, 2, 7, 5}));
    expect_box(joined.boxes()[0], 1.0f, 4.0f, 4.0f, 8.0f);
    expect_box(joined.boxes()[1], 0.0f, 3.0f, 2.0f, 5.0f);
    expect_box(joined.boxes()[2], 6.0f, 6.0f, 7.0f, 8.0f);
    expect_box(joined.boxes()[3], 7.0f, 13.0f, 11.0f, 16.0f);
}

TEST(Unit_ImagingMosaic, LabelsOfFourBackgroundPatchesAreEmpty)
{
    MosaicPlan plan;
    plan.image_size = K_SIDE;
    plan.placements = K_LOW_PLACEMENTS;

    const Labels joined = apply_mosaic(std::array<Labels, K_MOSAIC_PATCHES>{}, plan);

    EXPECT_TRUE(joined.is_empty());
    EXPECT_TRUE(joined.classes().empty());
}

TEST(Unit_ImagingMosaic, DrawRejectsANonPositiveSizeBeforeAnyDraw)
{
    DatasetPartners partners(10);
    Random random(1u);

    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(MosaicSettings{}, 0, 0, partners, random)),
                    "validate_mosaic_size", "image_size of 0");
    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(MosaicSettings{}, -4, 0, partners, random)),
                    "validate_mosaic_size", "image_size of -4");
    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
}

TEST(Unit_ImagingMosaic, PlanRejectsADisabledDrawAndAnEmptyPatch)
{
    PatchSizes sizes = K_LOW_SIZES;
    sizes[2] = ImageSize{0, 4};

    SA_EXPECT_ERROR(
        static_cast<void>(plan_mosaic(MosaicSettings{}, MosaicDraw{}, K_SIDE, K_LOW_SIZES)),
        "not enabled");
    SA_EXPECT_ERROR(
        static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE, sizes)),
        "patch 2");
    SA_EXPECT_ERROR(
        static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(17, 8), K_SIDE, K_LOW_SIZES)),
        "centre");
}

TEST(Unit_ImagingMosaic, ApplyRejectsPatchesThePlanWasNotMadeFor)
{
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 1, 1, 1), make_constant(K_LOW_SIZES[1], 2, 2, 2),
        make_constant(K_LOW_SIZES[2], 3, 3, 3), make_constant(K_LOW_SIZES[3], 4, 4, 4)};
    const Image gray(K_LOW_SIZES[1], 1);
    const Image small(ImageSize{4, 4}, 3);
    Image canvas;

    std::array<ImageView, K_MOSAIC_PATCHES> mixed = view_all(patches);
    mixed[1] = gray.view();
    SA_EXPECT_ERROR(apply_mosaic(mixed, plan, canvas), "patch 1", "channels");

    std::array<ImageView, K_MOSAIC_PATCHES> shrunk = view_all(patches);
    shrunk[3] = small.view();
    SA_EXPECT_ERROR(apply_mosaic(shrunk, plan, canvas), "patch 3");
}

TEST(Unit_ImagingMosaic, DrawnCentreIsWhereTheFourColoursMeet)
{
    const std::array<std::uint8_t, 3> red = {255, 0, 0};
    const std::array<std::uint8_t, 3> green = {0, 255, 0};
    const std::array<std::uint8_t, 3> blue = {0, 0, 255};
    const std::array<std::uint8_t, 3> yellow = {255, 255, 0};
    const ImageSize full{K_SIDE, K_SIDE};
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(full, 255, 0, 0), make_constant(full, 0, 255, 0),
        make_constant(full, 0, 0, 255), make_constant(full, 255, 255, 0)};

    DatasetPartners partners(4);
    for (std::uint64_t seed = 0; seed < 32; ++seed)
    {
        SCOPED_TRACE(::testing::Message() << "seed " << seed);
        Random random(seed);
        const MosaicDraw draw = draw_mosaic(MosaicSettings{}, K_SIDE, 0, partners, random);
        const MosaicPlan plan =
            plan_mosaic(MosaicSettings{}, draw, K_SIDE, {full, full, full, full});
        Image canvas;

        apply_mosaic(view_all(patches), plan, canvas);

        // int(uniform(4, 12)) at s = 8 keeps the centre and its left and upper neighbours inside.
        ASSERT_GE(draw.center_x, 4);
        ASSERT_LE(draw.center_x, 11);
        ASSERT_GE(draw.center_y, 4);
        ASSERT_LE(draw.center_y, 11);
        ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
        EXPECT_EQ(pixel_at(canvas, draw.center_x - 1, draw.center_y - 1), red);
        EXPECT_EQ(pixel_at(canvas, draw.center_x, draw.center_y - 1), green);
        EXPECT_EQ(pixel_at(canvas, draw.center_x - 1, draw.center_y), blue);
        EXPECT_EQ(pixel_at(canvas, draw.center_x, draw.center_y), yellow);
    }
}

TEST(Unit_ImagingMosaic, DrawRejectsAProbabilityOutsideZeroToOne)
{
    DatasetPartners partners(10);
    Random random(1u);
    MosaicSettings above;
    above.probability = 1.5f;
    MosaicSettings below;
    below.probability = -0.25f;
    MosaicSettings undefined;
    undefined.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(above, 640, 0, partners, random)),
                    "validate_mosaic", "probability of 1.5");
    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(below, 640, 0, partners, random)),
                    "validate_mosaic", "probability");
    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(undefined, 640, 0, partners, random)),
                    "validate_mosaic", "probability");
    SA_EXPECT_ERROR(validate_mosaic(above), "validate_mosaic", "probability");
    EXPECT_NO_THROW(validate_mosaic(MosaicSettings{}));
    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
}

TEST(Unit_ImagingMosaic, DrawRejectsASizeWhoseCanvasIsTooLarge)
{
    DatasetPartners partners(10);
    Random random(1u);

    SA_EXPECT_ERROR(static_cast<void>(draw_mosaic(MosaicSettings{}, K_MAX_IMAGE_SIDE / 2 + 1, 0,
                                                  partners, random)),
                    "validate_mosaic_size", "image_size of 8193", "[1, 8192]");
    EXPECT_TRUE(
        draw_mosaic(MosaicSettings{}, K_MAX_IMAGE_SIDE / 2, 0, partners, random).is_enabled);
}

/** @brief Tests that draw_mosaic asks the partner source once per call, gate open or closed. */
TEST(Unit_ImagingMosaic, DrawAsksThePartnerSourceForEverySample)
{
    MosaicSettings closed;
    closed.probability = 0.0f;
    RecordingPartners partners;
    Random random(3u);

    const MosaicDraw open_draw = draw_mosaic(MosaicSettings{}, 640, 17, partners, random);
    const MosaicDraw closed_draw = draw_mosaic(closed, 640, 23, partners, random);

    EXPECT_EQ(partners.indices(), (std::vector<std::size_t>{17, 23}));
    ASSERT_TRUE(open_draw.is_enabled);
    EXPECT_EQ(open_draw.partners, (MosaicPartners{4, 5, 6}));
    ASSERT_FALSE(closed_draw.is_enabled);
    EXPECT_EQ(closed_draw.partners, (MosaicPartners{0, 0, 0}));
}

/** @brief Tests that polygons are shifted with their patch, clamped to the canvas and pruned. */
TEST(Unit_ImagingMosaic, PolygonsAreShiftedClampedAndPrunedWithTheirObjects)
{
    MosaicPlan plan;
    plan.image_size = K_SIDE;
    plan.placements = K_LOW_PLACEMENTS;
    std::array<Labels, K_MOSAIC_PATCHES> labels;
    labels[0].add(3, PixelBox{4.0f, 1.0f, 7.0f, 5.0f},
                  Polygon{Point{4.0f, 1.0f}, Point{7.0f, 5.0f}, Point{1.0f, 3.0f}});
    labels[0].add(1, PixelBox{0.0f, 2.0f, 3.0f, 6.0f},
                  Polygon{Point{0.0f, 2.0f}, Point{3.0f, 2.0f}, Point{3.0f, 6.0f}});
    labels[0].add(2, PixelBox{1.0f, 0.0f, 5.0f, 2.0f});
    labels[3].add(5, PixelBox{2.0f, 2.0f, 6.0f, 7.0f},
                  Polygon{Point{2.0f, 2.0f}, Point{6.0f, 7.0f}, Point{6.0f, 2.0f}});

    const Labels joined = apply_mosaic(labels, plan);

    // Patch 0 moves by (-3, 3) and patch 3 by (5, 11); the canvas is 16 by 16.
    ASSERT_EQ(joined.size(), 3u);
    EXPECT_EQ(joined.classes(), (std::vector<std::int32_t>{3, 2, 5}));
    ASSERT_EQ(joined.polygons()[0].size(), 3u);
    expect_point(joined.polygons()[0][0], 1.0f, 4.0f);
    expect_point(joined.polygons()[0][1], 4.0f, 8.0f);
    expect_point(joined.polygons()[0][2], 0.0f, 6.0f);
    EXPECT_TRUE(joined.polygons()[1].empty());
    ASSERT_EQ(joined.polygons()[2].size(), 3u);
    expect_point(joined.polygons()[2][0], 7.0f, 13.0f);
    expect_point(joined.polygons()[2][1], 11.0f, 16.0f);
    expect_point(joined.polygons()[2][2], 11.0f, 13.0f);
    expect_box(joined.boxes()[2], 7.0f, 13.0f, 11.0f, 16.0f);
}

TEST(Unit_ImagingMosaic, PlanRejectsASizeOutOfRangeAndAPatchSideAboveTheLimit)
{
    PatchSizes sizes = K_LOW_SIZES;
    sizes[1] = ImageSize{K_MAX_IMAGE_SIDE + 1, 8};

    SA_EXPECT_ERROR(
        static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(0, 0), 0, K_LOW_SIZES)),
        "image_size");
    SA_EXPECT_ERROR(static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(8, 8),
                                                  K_MAX_IMAGE_SIDE / 2 + 1, K_LOW_SIZES)),
                    "image_size");
    SA_EXPECT_ERROR(
        static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE, sizes)),
        "patch 1");
}

TEST(Unit_ImagingMosaic, ApplyRejectsAPlanWithoutAValidSizeForPixelsAndForLabels)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 1, 1, 1), make_constant(K_LOW_SIZES[1], 2, 2, 2),
        make_constant(K_LOW_SIZES[2], 3, 3, 3), make_constant(K_LOW_SIZES[3], 4, 4, 4)};
    std::array<Labels, K_MOSAIC_PATCHES> labels;
    labels[0] = Labels({3}, {PixelBox{1.0f, 1.0f, 5.0f, 5.0f}});
    MosaicPlan oversized;
    oversized.image_size = K_MAX_IMAGE_SIDE / 2 + 1;
    Image canvas;

    SA_EXPECT_ERROR(apply_mosaic(view_all(patches), MosaicPlan{}, canvas), "image_size");
    SA_EXPECT_ERROR(apply_mosaic(view_all(patches), oversized, canvas), "image_size");
    SA_EXPECT_ERROR(static_cast<void>(apply_mosaic(labels, MosaicPlan{})), "image_size");
    SA_EXPECT_ERROR(static_cast<void>(apply_mosaic(labels, oversized)), "image_size");
}

TEST(Unit_ImagingMosaic, ApplyRejectsAPatchWithoutPixels)
{
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 1, 1, 1), make_constant(K_LOW_SIZES[1], 2, 2, 2),
        make_constant(K_LOW_SIZES[2], 3, 3, 3), make_constant(K_LOW_SIZES[3], 4, 4, 4)};
    std::array<ImageView, K_MOSAIC_PATCHES> views = view_all(patches);
    views[2] = ImageView();
    Image canvas;

    SA_EXPECT_ERROR(apply_mosaic(views, plan, canvas), "apply_mosaic: patch 2", "no pixels");
}

TEST(Unit_ImagingMosaic, ApplyRejectsAPatchThatLiesAnywhereInTheDestination)
{
    const MosaicPlan plan = plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES);
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 1, 1, 1), make_constant(K_LOW_SIZES[1], 2, 2, 2),
        make_constant(K_LOW_SIZES[2], 3, 3, 3), make_constant(K_LOW_SIZES[3], 4, 4, 4)};
    Image canvas(ImageSize{40, 40}, 3);
    canvas.fill(200);

    std::array<ImageView, K_MOSAIC_PATCHES> whole = view_all(patches);
    whole[0] = ImageView(canvas.data(), K_LOW_SIZES[0], 3);
    SA_EXPECT_ERROR(apply_mosaic(whole, plan, canvas), "apply_mosaic: patch 0",
                    "destination holds the source's pixels");

    std::array<ImageView, K_MOSAIC_PATCHES> inner = view_all(patches);
    inner[2] = ImageView(canvas.row(20), K_LOW_SIZES[2], 3);
    SA_EXPECT_ERROR(apply_mosaic(inner, plan, canvas), "apply_mosaic: patch 2",
                    "destination holds the source's pixels");

    EXPECT_EQ(canvas.size(), (ImageSize{40, 40})) << "a rejected call reset the destination";
    EXPECT_EQ(canvas.row(20)[0], 200) << "a rejected call wrote the destination";
}

TEST(Unit_ImagingMosaic, ApplyRejectsAPlacementOutsideTheCanvasAndOneThatResamples)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_constant(K_LOW_SIZES[0], 1, 1, 1), make_constant(K_LOW_SIZES[1], 2, 2, 2),
        make_constant(K_LOW_SIZES[2], 3, 3, 3), make_constant(K_LOW_SIZES[3], 4, 4, 4)};
    MosaicPlan outside;
    outside.image_size = K_SIDE;
    outside.placements = K_LOW_PLACEMENTS;
    outside.placements[1].canvas_x2 = 2 * K_SIDE + 1;
    MosaicPlan narrower;
    narrower.image_size = K_SIDE;
    narrower.placements = K_LOW_PLACEMENTS;
    narrower.placements[3].patch_x2 = 7;
    Image canvas;

    SA_EXPECT_ERROR(apply_mosaic(view_all(patches), outside, canvas), "patch 1",
                    "outside a canvas");
    SA_EXPECT_ERROR(apply_mosaic(view_all(patches), narrower, canvas), "patch 3",
                    "resamples nothing");
}

/**************************************************************************/
/* test_mosaic.cpp                                                        */
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
 * @file test_mosaic.cpp
 * @brief Tests the mosaic's draws, its placement rectangles, its canvas and its joined labels.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/in_memory_partner_source.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the chance that a correct run of draws still misses a statistical bound. */
    constexpr double K_FALSE_FAILURE_RATE = 1e-9;

    /**
     * @brief Returns the distance from its expectation that a mean of draws within a range stays
     *        within.
     *
     * Hoeffding's inequality gives P(|mean - expectation| >= t) <= 2 exp(-2 n t^2 / range^2), so t
     * is range * sqrt(ln(2 / rate) / (2 n)) for n draws and K_FALSE_FAILURE_RATE.
     */
    double mean_tolerance(double range, std::uint64_t draws)
    {
        return range *
               std::sqrt(std::log(2.0 / K_FALSE_FAILURE_RATE) / (2.0 * static_cast<double>(draws)));
    }

    /** @brief Holds the training size of the small mosaics below; the canvas is 16 by 16. */
    constexpr std::int32_t K_SIDE = 8;

    /** @brief Holds the four patch sizes of a mosaic, patch 0 first. */
    using PatchSizes = std::array<ImageSize, K_MOSAIC_PATCHES>;

    /** @brief Holds the four hand-computed placements a plan is compared with. */
    using Placements = std::array<MosaicPlacement, K_MOSAIC_PATCHES>;

    /** @brief Holds a centre that cuts patch 0 on the left and patches 2 and 3 at the bottom. */
    constexpr std::int32_t K_LOW_CENTER_X = 5;

    /** @brief Holds the row of that centre. */
    constexpr std::int32_t K_LOW_CENTER_Y = 11;

    /** @brief Holds the patch sizes used with the low centre. */
    constexpr PatchSizes K_LOW_SIZES =
        {ImageSize{8, 8}, ImageSize{6, 8}, ImageSize{8, 4}, ImageSize{8, 8}};

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
    constexpr PatchSizes K_HIGH_SIZES =
        {ImageSize{8, 8}, ImageSize{8, 8}, ImageSize{8, 8}, ImageSize{8, 8}};

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

    /** @brief Returns the patches red, green, blue and yellow, each of the size given for it. */
    std::array<Image, K_MOSAIC_PATCHES> make_colour_patches(const PatchSizes& sizes)
    {
        return {make_constant(sizes[0], 255, 0, 0), make_constant(sizes[1], 0, 255, 0),
                make_constant(sizes[2], 0, 0, 255), make_constant(sizes[3], 255, 255, 0)};
    }

    /** @brief Returns four patches of other, distinct constant colours, each of its size. */
    std::array<Image, K_MOSAIC_PATCHES> make_dark_patches(const PatchSizes& sizes)
    {
        return {make_constant(sizes[0], 1, 1, 1), make_constant(sizes[1], 2, 2, 2),
                make_constant(sizes[2], 3, 3, 3), make_constant(sizes[3], 4, 4, 4)};
    }

    /** @brief Returns the views of four images in patch order. */
    std::array<ImageView, K_MOSAIC_PATCHES> view_all(
        const std::array<Image, K_MOSAIC_PATCHES>& images)
    {
        return {images[0].view(), images[1].view(), images[2].view(), images[3].view()};
    }

    /** @brief Counts the canvas bytes that differ from the patch byte the placements name. */
    std::size_t count_wrong_bytes(const Image& canvas, const std::array<Image,
                                  K_MOSAIC_PATCHES>& patches, const Placements& placements,
                                  std::uint8_t fill_value)
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

    /** @brief Checks the four edges of one box, which are small integers, exactly. */
    void expect_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_EQ(actual.x1, x1);
        EXPECT_EQ(actual.y1, y1);
        EXPECT_EQ(actual.x2, x2);
        EXPECT_EQ(actual.y2, y2);
    }

    /** @brief Checks one vertex, whose coordinates are small integers, exactly. */
    void expect_point(const Point& actual, float x, float y)
    {
        EXPECT_EQ(actual.x, x);
        EXPECT_EQ(actual.y, y);
    }

    /** @brief Returns the three bytes of one canvas pixel. */
    std::array<std::uint8_t, 3> pixel_at(const Image& image, std::int32_t x, std::int32_t y)
    {
        const std::uint8_t* row = image.row(y);
        return {row[3 * x], row[3 * x + 1], row[3 * x + 2]};
    }

    /** @brief Advances a generator by @p draws values of next_u64. */
    void skip_draws(Random& random, std::int32_t draws)
    {
        for (std::int32_t i = 0; i < draws; ++i)
            static_cast<void>(random.next_u64());
    }

    /** @brief Holds what draw_mosaic gave over a run of seeds at the reference size. */
    struct CentreStatistics
    {
        /** @brief Tells whether every draw had its gate open. */
        bool is_every_gate_open;

        /** @brief Holds the smallest centre coordinate drawn. */
        std::int32_t smallest;

        /** @brief Holds the largest centre coordinate drawn. */
        std::int32_t largest;

        /** @brief Holds the mean of the centre coordinates drawn. */
        double mean;
    };

    /** @brief Draws a mosaic of size 640 for each seed in [0, seeds) and summarises the centres. */
    CentreStatistics measure_centres(std::uint64_t seeds)
    {
        const std::unique_ptr<DatasetPartners> partners_owner =
            require_value(DatasetPartners::make(100));
        CentreStatistics statistics{true, 1280, 0, 0.0};
        double sum = 0.0;
        for (std::uint64_t seed = 0; seed < seeds; ++seed)
        {
            Random random(seed);
            const MosaicDraw draw =
                require_value(draw_mosaic(MosaicSettings{}, 640, 0, *partners_owner, random));

            statistics.is_every_gate_open = statistics.is_every_gate_open && draw.is_enabled;
            for (const std::int32_t center : {draw.center_x, draw.center_y})
            {
                statistics.smallest = std::min(statistics.smallest, center);
                statistics.largest = std::max(statistics.largest, center);
                sum += center;
            }
        }

        statistics.mean = sum / static_cast<double>(2 * seeds);
        return statistics;
    }

    /** @brief Holds the centre coordinates that draw_mosaic gave over a run of seeds. */
    struct CentreSets
    {
        /** @brief Holds every x coordinate drawn. */
        std::set<std::int32_t> xs;

        /** @brief Holds every y coordinate drawn. */
        std::set<std::int32_t> ys;
    };

    /** @brief Draws a mosaic of size 5 for each seed in [0, seeds) and collects the centres. */
    CentreSets collect_centres_of_size_five(std::uint64_t seeds)
    {
        const std::unique_ptr<DatasetPartners> partners_owner =
            require_value(DatasetPartners::make(3));
        CentreSets sets;
        for (std::uint64_t seed = 0; seed < seeds; ++seed)
        {
            Random random(seed);
            const MosaicDraw draw =
                require_value(draw_mosaic(MosaicSettings{}, 5, 0, *partners_owner, random));
            sets.xs.insert(draw.center_x);
            sets.ys.insert(draw.center_y);
        }

        return sets;
    }

    /** @brief Draws a mosaic at probability 0.25 for each seed in [0, seeds); counts open gates. */
    std::uint64_t count_open_gates(std::uint64_t seeds)
    {
        MosaicSettings settings;
        settings.probability = 0.25f;
        const std::unique_ptr<DatasetPartners> partners_owner =
            require_value(DatasetPartners::make(10));
        std::uint64_t enabled = 0;
        for (std::uint64_t seed = 0; seed < seeds; ++seed)
        {
            Random random(seed);
            if (require_value(draw_mosaic(settings, 640, 0, *partners_owner, random)).is_enabled)
                ++enabled;
        }

        return enabled;
    }

    /** @brief Describes a hand-worked centre, the patch sizes used with it and its placements. */
    struct PlacementRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the centre column. */
        std::int32_t center_x;

        /** @brief Holds the centre row. */
        std::int32_t center_y;

        /** @brief Holds the patch sizes. */
        PatchSizes sizes;

        /** @brief Holds the placements worked by hand. */
        Placements placements;
    };

    /** @brief Runs each placement case. */
    class Small_ImagingMosaicPlacement : public ::testing::TestWithParam<PlacementRow>
    {
    };

    /** @brief Describes a centre and the patch sizes used with it. */
    struct MeetRow
    {
        /** @brief Names the row in the test report. */
        std::string name;

        /** @brief Holds the centre column. */
        std::int32_t center_x;

        /** @brief Holds the centre row. */
        std::int32_t center_y;

        /** @brief Holds the patch sizes. */
        PatchSizes sizes;
    };

    /** @brief Returns the fixed low centre and the centres of 32 draws at the size K_SIDE. */
    std::vector<MeetRow> meet_rows()
    {
        std::vector<MeetRow> rows;
        rows.push_back(MeetRow{"TheFixedLowCentre", K_LOW_CENTER_X, K_LOW_CENTER_Y, K_LOW_SIZES});

        const ImageSize full{K_SIDE, K_SIDE};
        const std::unique_ptr<DatasetPartners> partners_owner =
            require_value(DatasetPartners::make(4));
        for (std::uint64_t seed = 0; seed < 32; ++seed)
        {
            Random random(seed);
            const MosaicDraw draw =
                require_value(draw_mosaic(MosaicSettings{}, K_SIDE, 0, *partners_owner, random));
            rows.push_back(MeetRow{"DrawnCentreOfSeed" + std::to_string(seed), draw.center_x,
                                   draw.center_y, PatchSizes{full, full, full, full}});
        }

        return rows;
    }

    /** @brief Runs each four-colour case. */
    class Small_ImagingMosaicColours : public ::testing::TestWithParam<MeetRow>
    {
    };

    /** @brief Builds the three label sets the join cases use: patches 0, 1 and 3 hold objects. */
    std::array<Labels, K_MOSAIC_PATCHES> make_patch_labels()
    {
        std::array<Labels, K_MOSAIC_PATCHES> labels;
        labels[0] =
            Labels({3, 1, 2},
                   {PixelBox{4.0f, 1.0f, 7.0f, 5.0f}, PixelBox{0.0f, 2.0f, 3.0f, 6.0f},
                    PixelBox{1.0f, 0.0f, 5.0f, 2.0f}});
        labels[1] = Labels({7}, {PixelBox{1.0f, 1.0f, 2.0f, 3.0f}});
        labels[3] = Labels({5}, {PixelBox{2.0f, 2.0f, 6.0f, 7.0f}});
        return labels;
    }

    /** @brief Returns a plan whose placements are the hand-worked ones of the low centre. */
    MosaicPlan make_low_plan()
    {
        MosaicPlan plan;
        plan.image_size = K_SIDE;
        plan.placements = K_LOW_PLACEMENTS;
        return plan;
    }
} // namespace

TEST(Small_ImagingMosaic, DrawMakesPartnersThenGateThenRowThenColumn)
{
    const std::int32_t side = 640;
    const std::size_t pool = 1000;
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(pool));
    DatasetPartners& partners = *partners_owner;
    Random random(20261007u);
    Random replica = random;

    const MosaicDraw draw = require_value(draw_mosaic(MosaicSettings{}, side, 0, partners, random));

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

/** @brief Tests that a closed gate leaves the partners and the centre at zero. */
TEST(Small_ImagingMosaic, DisabledDrawLeavesTheFieldsZero)
{
    MosaicSettings settings;
    settings.probability = 0.0f;
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(50));
    DatasetPartners& partners = *partners_owner;
    Random random(7u);

    const MosaicDraw draw = require_value(draw_mosaic(settings, 640, 0, partners, random));

    EXPECT_FALSE(draw.is_enabled);
    EXPECT_EQ(draw.partners[0], 0u);
    EXPECT_EQ(draw.partners[1], 0u);
    EXPECT_EQ(draw.partners[2], 0u);
    EXPECT_EQ(draw.center_x, 0);
    EXPECT_EQ(draw.center_y, 0);
}

/** @brief Tests that a closed gate still makes the three partner draws and the gate draw. */
TEST(Small_ImagingMosaic, DisabledDrawMakesFourDraws)
{
    MosaicSettings settings;
    settings.probability = 0.0f;
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(50));
    DatasetPartners& partners = *partners_owner;
    Random random(7u);
    Random replica = random;

    static_cast<void>(require_value(draw_mosaic(settings, 640, 0, partners, random)));

    skip_draws(replica, 4);
    EXPECT_TRUE(random == replica) << "a closed gate made a number of draws other than four";
}

/** @brief Tests that, over 5000 seeds, the centre covers the reference range and its mean. */
TEST(Small_ImagingMosaic, CentreStaysInTheReferenceRangeOverManySeeds)
{
    // int(uniform(320, 960)) at s = 640 (data/augment.py:480, 520) is a value in 320 to 959.
    const std::uint64_t seeds = 5000;

    const CentreStatistics statistics = measure_centres(seeds);

    EXPECT_TRUE(statistics.is_every_gate_open);
    EXPECT_GE(statistics.smallest, 320);
    EXPECT_LE(statistics.largest, 959);
    EXPECT_NEAR(statistics.mean, 639.5, mean_tolerance(640.0, 2 * seeds));
    EXPECT_LE(statistics.smallest, 330) << "the low end of the range is never drawn";
    EXPECT_GE(statistics.largest, 949) << "the high end of the range is never drawn";
}

/** @brief Tests that, for a size of 5, the centre is drawn from 3 to 6 on both axes. */
TEST(Small_ImagingMosaic, CentreOfAnOddSizeUsesTheFlooredBorder)
{
    // -5 // 2 is -3 in the reference, so the range is int(uniform(3, 7)): 3 to 6.
    const CentreSets sets = collect_centres_of_size_five(400);

    EXPECT_EQ(sets.xs, (std::set<std::int32_t>{3, 4, 5, 6}));
    EXPECT_EQ(sets.ys, (std::set<std::int32_t>{3, 4, 5, 6}));
}

/** @brief Tests that the gate opens with the probability of the settings. */
TEST(Small_ImagingMosaic, GateOpensWithItsProbability)
{
    const std::uint64_t seeds = 4000;

    const std::uint64_t enabled = count_open_gates(seeds);

    EXPECT_NEAR(static_cast<double>(enabled) / static_cast<double>(seeds), 0.25,
                mean_tolerance(1.0, seeds));
}

/** @brief Tests that the plan cuts patches against the canvas edges as the reference does. */
TEST_P(Small_ImagingMosaicPlacement, PlanCutsPatchesAtTheEdgesOfTheCanvas)
{
    const PlacementRow& row = GetParam();

    const MosaicPlan plan = require_value(
        plan_mosaic(MosaicSettings{}, make_draw(row.center_x, row.center_y), K_SIDE, row.sizes));

    EXPECT_EQ(plan.image_size, K_SIDE);
    expect_placements(plan, row.placements);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingMosaicPlacement,
    ::testing::Values(PlacementRow{"AtTheLeftAndBottomEdges", K_LOW_CENTER_X, K_LOW_CENTER_Y,
                                   K_LOW_SIZES, K_LOW_PLACEMENTS},
                      PlacementRow{"AtTheTopAndRightEdges", K_HIGH_CENTER_X, K_HIGH_CENTER_Y,
                                   K_HIGH_SIZES, K_HIGH_PLACEMENTS}),
    [](const ::testing::TestParamInfo<PlacementRow>& info)
    {
        return std::string(info.param.name);
    });

TEST(Small_ImagingMosaic, PlanCarriesTheFillValueOfTheSettings)
{
    MosaicSettings settings;
    settings.fill_value = 9;

    const MosaicPlan plan = require_value(plan_mosaic(
        settings, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));

    EXPECT_EQ(plan.fill_value, 9);
    EXPECT_EQ(require_value(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE,
                                        K_LOW_SIZES)).fill_value,
              114);
}

/** @brief Tests that the canvas is twice the training size, in three channels. */
TEST(Small_ImagingMosaic, CanvasIsTwiceTheSizeOfTheTrainingSize)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_colour_patches(K_LOW_SIZES);
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    Image canvas;

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    ASSERT_EQ(canvas.height(), 2 * K_SIDE);
    ASSERT_EQ(canvas.width(), 2 * K_SIDE);
    ASSERT_EQ(canvas.channels(), 3);
}

/** @brief Tests that canvas pixels no patch lands on take the fill value. */
TEST(Small_ImagingMosaic, UncoveredCanvasTakesTheFillValue)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_colour_patches(K_LOW_SIZES);
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    Image canvas;

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    const std::array<std::uint8_t, 3> fill = {114, 114, 114};
    EXPECT_EQ(pixel_at(canvas, 0, 0), fill);
    EXPECT_EQ(pixel_at(canvas, 15, 15), fill);
    EXPECT_EQ(pixel_at(canvas, 7, 4), fill) << "the row above the shorter patch 1";
    EXPECT_EQ(pixel_at(canvas, 0, 12), fill) << "the column left of the narrower patch 2";
}

/** @brief Tests that each patch lands on the canvas where its hand-worked placement puts it. */
TEST(Small_ImagingMosaic, EachPatchLandsWhereItsPlacementPutsIt)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_colour_patches(K_LOW_SIZES);
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    Image canvas;

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_LOW_PLACEMENTS, 114), 0u);
}

/** @brief Tests that the four colours meet at the centre, for a fixed centre and drawn ones. */
TEST_P(Small_ImagingMosaicColours, EachColourLandsInItsQuadrantAroundTheCentre)
{
    const MeetRow& row = GetParam();
    const std::array<std::uint8_t, 3> red = {255, 0, 0};
    const std::array<std::uint8_t, 3> green = {0, 255, 0};
    const std::array<std::uint8_t, 3> blue = {0, 0, 255};
    const std::array<std::uint8_t, 3> yellow = {255, 255, 0};
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_colour_patches(row.sizes);
    const MosaicPlan plan = require_value(
        plan_mosaic(MosaicSettings{}, make_draw(row.center_x, row.center_y), K_SIDE, row.sizes));
    Image canvas;

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    // int(uniform(4, 12)) at s = 8 keeps the centre and its left and upper neighbours inside.
    ASSERT_GE(row.center_x, 4);
    ASSERT_LE(row.center_x, 11);
    ASSERT_GE(row.center_y, 4);
    ASSERT_LE(row.center_y, 11);
    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(pixel_at(canvas, row.center_x - 1, row.center_y - 1), red);
    EXPECT_EQ(pixel_at(canvas, row.center_x, row.center_y - 1), green);
    EXPECT_EQ(pixel_at(canvas, row.center_x - 1, row.center_y), blue);
    EXPECT_EQ(pixel_at(canvas, row.center_x, row.center_y), yellow);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingMosaicColours, ::testing::ValuesIn(meet_rows()),
                         [](const ::testing::TestParamInfo<MeetRow>& info)
                         {
                             return info.param.name;
                         });

TEST(Small_ImagingMosaic, CutPatchesKeepThePixelsNearestTheCentre)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_textured(K_HIGH_SIZES[0], 1), make_textured(K_HIGH_SIZES[1], 60),
        make_textured(K_HIGH_SIZES[2], 120), make_textured(K_HIGH_SIZES[3], 180)};
    MosaicSettings settings;
    settings.fill_value = 3;
    const MosaicPlan plan = require_value(plan_mosaic(
        settings, make_draw(K_HIGH_CENTER_X, K_HIGH_CENTER_Y), K_SIDE, K_HIGH_SIZES));
    Image canvas;

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_HIGH_PLACEMENTS, 3), 0u);
}

TEST(Small_ImagingMosaic, ApplyOverwritesWhatTheDestinationHeld)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = {
        make_textured(K_LOW_SIZES[0], 5), make_textured(K_LOW_SIZES[1], 6),
        make_textured(K_LOW_SIZES[2], 7), make_textured(K_LOW_SIZES[3], 8)};
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    Image canvas(ImageSize{40, 40}, 3);
    canvas.fill(200);

    SA_EXPECT_OK(apply_mosaic(view_all(patches), plan, canvas));

    ASSERT_EQ(canvas.size(), (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(count_wrong_bytes(canvas, patches, K_LOW_PLACEMENTS, 114), 0u);
}

/** @brief Tests that labels are joined in patch order and an object clipped to nothing goes. */
TEST(Small_ImagingMosaic, LabelsAreJoinedInPatchOrderAndPrunedWhenClippedToNothing)
{
    const Labels joined = apply_mosaic(make_patch_labels(), make_low_plan());

    // Patch 0's second box lies wholly left of the canvas and has no area after the clip.
    ASSERT_EQ(joined.size(), 4u);
    EXPECT_EQ(joined.classes(), (std::vector<std::int32_t>{3, 2, 7, 5}));
}

/** @brief Tests that label boxes are shifted by the offset of their patch and clipped. */
TEST(Small_ImagingMosaic, LabelBoxesAreShiftedByTheirPatchOffsetAndClippedToTheCanvas)
{
    const Labels joined = apply_mosaic(make_patch_labels(), make_low_plan());

    ASSERT_EQ(joined.size(), 4u);
    expect_box(joined.boxes()[0], 1.0f, 4.0f, 4.0f, 8.0f);
    expect_box(joined.boxes()[1], 0.0f, 3.0f, 2.0f, 5.0f);
    expect_box(joined.boxes()[2], 6.0f, 6.0f, 7.0f, 8.0f);
    expect_box(joined.boxes()[3], 7.0f, 13.0f, 11.0f, 16.0f);
}

TEST(Small_ImagingMosaic, LabelsOfFourBackgroundPatchesAreEmpty)
{
    const Labels joined = apply_mosaic(std::array<Labels, K_MOSAIC_PATCHES>{}, make_low_plan());

    EXPECT_TRUE(joined.is_empty());
    EXPECT_TRUE(joined.classes().empty());
}

/** @brief Tests that a size that is not positive is refused, naming the size. */
TEST(Small_ImagingMosaic, DrawRejectsANonPositiveSize)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);

    SA_EXPECT_FAILURE(draw_mosaic(MosaicSettings{}, 0, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic_size", "image_size of 0");
    SA_EXPECT_FAILURE(draw_mosaic(MosaicSettings{}, -4, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic_size", "image_size of -4");
}

/** @brief Tests that a draw refused for its size makes no random draw. */
TEST(Small_ImagingMosaic, DrawRefusedForItsSizeMakesNoDraw)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);

    static_cast<void>(draw_mosaic(MosaicSettings{}, 0, 0, partners, random));

    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
}

/** @brief Tests that a disabled draw is refused by plan_mosaic. */
TEST(Medium_ImagingMosaic, PlanRejectsADisabledDraw)
{
    SA_EXPECT_FATAL(static_cast<void>(plan_mosaic(MosaicSettings{}, MosaicDraw{}, K_SIDE,
                                                  K_LOW_SIZES)),
                    "not enabled");
}

/** @brief Tests that a patch with no pixel is refused by plan_mosaic, naming the patch. */
TEST(Small_ImagingMosaic, PlanRejectsAnEmptyPatch)
{
    PatchSizes sizes = K_LOW_SIZES;
    sizes[2] = ImageSize{0, 4};

    SA_EXPECT_FAILURE(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE, sizes),
                      StatusCode::InvalidDescription, "patch 2");
}

/** @brief Tests that a centre outside the canvas is refused by plan_mosaic. */
TEST(Medium_ImagingMosaic, PlanRejectsACentreOutsideTheCanvas)
{
    SA_EXPECT_FATAL(static_cast<void>(plan_mosaic(MosaicSettings{}, make_draw(17, 8), K_SIDE,
                                                  K_LOW_SIZES)),
                    "centre");
}

/** @brief Tests that a patch with another channel count than the others is refused. */
TEST(Small_ImagingMosaic, ApplyRejectsAPatchWithOtherChannels)
{
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    const Image gray(K_LOW_SIZES[1], 1);
    Image canvas;
    std::array<ImageView, K_MOSAIC_PATCHES> mixed = view_all(patches);
    mixed[1] = gray.view();

    SA_EXPECT_FAILURE(apply_mosaic(mixed, plan, canvas), StatusCode::InvalidDescription, "patch 1",
                      "channels");
}

/** @brief Tests that a patch of another size than the plan was made for is refused. */
TEST(Medium_ImagingMosaic, ApplyRejectsAPatchOfAnotherSizeThanThePlan)
{
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    const Image small(ImageSize{4, 4}, 3);
    Image canvas;
    std::array<ImageView, K_MOSAIC_PATCHES> shrunk = view_all(patches);
    shrunk[3] = small.view();

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(shrunk, plan, canvas)), "patch 3");
}

/** @brief Tests that a probability outside [0, 1], or NaN, is refused by draw_mosaic. */
TEST(Small_ImagingMosaic, DrawRejectsAProbabilityOutsideZeroToOne)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);
    MosaicSettings above;
    above.probability = 1.5f;
    MosaicSettings below;
    below.probability = -0.25f;
    MosaicSettings undefined;
    undefined.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_FAILURE(draw_mosaic(above, 640, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic", "probability of 1.5");
    SA_EXPECT_FAILURE(draw_mosaic(below, 640, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic", "probability");
    SA_EXPECT_FAILURE(draw_mosaic(undefined, 640, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic", "probability");
}

/** @brief Tests that validate_mosaic refuses a probability above 1 and accepts the defaults. */
TEST(Small_ImagingMosaic, ValidateRejectsAProbabilityOutsideZeroToOne)
{
    MosaicSettings above;
    above.probability = 1.5f;

    SA_EXPECT_FAILURE(validate_mosaic(above), StatusCode::InvalidDescription, "validate_mosaic",
                      "probability");
    SA_EXPECT_OK(validate_mosaic(MosaicSettings{}));
}

/** @brief Tests that a draw refused for its probability makes no random draw. */
TEST(Small_ImagingMosaic, DrawRefusedForItsProbabilityMakesNoDraw)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);
    MosaicSettings above;
    above.probability = 1.5f;

    static_cast<void>(draw_mosaic(above, 640, 0, partners, random));

    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
}

/** @brief Tests that a size whose canvas would pass the image bound is refused. */
TEST(Small_ImagingMosaic, DrawRejectsASizeWhoseCanvasIsTooLarge)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);

    SA_EXPECT_FAILURE(draw_mosaic(MosaicSettings{}, K_MAX_IMAGE_SIDE / 2 + 1, 0, partners, random),
                      StatusCode::InvalidDescription, "validate_mosaic_size", "image_size of 8193",
                      "[1, 8192]");
}

/** @brief Tests that the largest size whose canvas fits the image bound draws. */
TEST(Small_ImagingMosaic, DrawAcceptsTheLargestSizeWhoseCanvasFits)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(10));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);

    EXPECT_TRUE(
        require_value(draw_mosaic(MosaicSettings{}, K_MAX_IMAGE_SIDE / 2, 0, partners,
        random)).is_enabled);
}

/** @brief Tests that draw_mosaic asks the partner source once per call, gate open or closed. */
TEST(Small_ImagingMosaic, DrawAsksThePartnerSourceForEverySample)
{
    MosaicSettings closed;
    closed.probability = 0.0f;
    InMemoryPartnerSource partners({MosaicPartners{4, 5, 6}});
    Random random(3u);

    const MosaicDraw open_draw =
        require_value(draw_mosaic(MosaicSettings{}, 640, 17, partners, random));
    const MosaicDraw closed_draw = require_value(draw_mosaic(closed, 640, 23, partners, random));

    EXPECT_EQ(partners.asked(), (std::vector<std::size_t>{17, 23}));
    ASSERT_TRUE(open_draw.is_enabled);
    EXPECT_EQ(open_draw.partners, (MosaicPartners{4, 5, 6}));
    ASSERT_FALSE(closed_draw.is_enabled);
    EXPECT_EQ(closed_draw.partners, (MosaicPartners{0, 0, 0}));
}

/** @brief Tests that polygons are shifted with their patch, clamped to the canvas and pruned. */
TEST(Small_ImagingMosaic, PolygonsAreShiftedClampedAndPrunedWithTheirObjects)
{
    const MosaicPlan plan = make_low_plan();
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

/** @brief Tests that plan_mosaic refuses a size that is zero or whose canvas is too large. */
TEST(Small_ImagingMosaic, PlanRejectsASizeOutOfRange)
{
    SA_EXPECT_FAILURE(plan_mosaic(MosaicSettings{}, make_draw(0, 0), 0, K_LOW_SIZES),
                      StatusCode::InvalidDescription, "image_size");
    SA_EXPECT_FAILURE(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_MAX_IMAGE_SIDE / 2 + 1,
                                  K_LOW_SIZES),
                      StatusCode::InvalidDescription, "image_size");
}

/** @brief Tests that plan_mosaic refuses a patch side above the image bound. */
TEST(Small_ImagingMosaic, PlanRejectsAPatchSideAboveTheLimit)
{
    PatchSizes sizes = K_LOW_SIZES;
    sizes[1] = ImageSize{K_MAX_IMAGE_SIDE + 1, 8};

    SA_EXPECT_FAILURE(plan_mosaic(MosaicSettings{}, make_draw(8, 8), K_SIDE, sizes),
                      StatusCode::InvalidDescription, "patch 1");
}

/** @brief Tests that the pixel overload of apply_mosaic refuses a plan without a valid size. */
TEST(Medium_ImagingMosaic, ApplyToPixelsRejectsAPlanWithoutAValidSize)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    MosaicPlan oversized;
    oversized.image_size = K_MAX_IMAGE_SIDE / 2 + 1;
    Image canvas;

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(view_all(patches), MosaicPlan{}, canvas)),
                    "image_size");
    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(view_all(patches), oversized, canvas)),
                    "image_size");
}

/** @brief Tests that the label overload of apply_mosaic refuses a plan without a valid size. */
TEST(Medium_ImagingMosaic, ApplyToLabelsRejectsAPlanWithoutAValidSize)
{
    std::array<Labels, K_MOSAIC_PATCHES> labels;
    labels[0] = Labels({3}, {PixelBox{1.0f, 1.0f, 5.0f, 5.0f}});
    MosaicPlan oversized;
    oversized.image_size = K_MAX_IMAGE_SIDE / 2 + 1;

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(labels, MosaicPlan{})), "image_size");
    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(labels, oversized)), "image_size");
}

/** @brief Tests that apply_mosaic refuses a patch without pixels. */
TEST(Medium_ImagingMosaic, ApplyRejectsAPatchWithoutPixels)
{
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    std::array<ImageView, K_MOSAIC_PATCHES> views = view_all(patches);
    views[2] = ImageView();
    Image canvas;

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(views, plan, canvas)), "apply_mosaic: patch 2",
                    "no pixels");
}

/** @brief Tests that apply_mosaic refuses a patch that lies anywhere in the destination. */
TEST(Medium_ImagingMosaic, ApplyRejectsAPatchThatLiesAnywhereInTheDestination)
{
    const MosaicPlan plan = require_value(plan_mosaic(
        MosaicSettings{}, make_draw(K_LOW_CENTER_X, K_LOW_CENTER_Y), K_SIDE, K_LOW_SIZES));
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    Image canvas(ImageSize{40, 40}, 3);
    canvas.fill(200);

    std::array<ImageView, K_MOSAIC_PATCHES> whole = view_all(patches);
    whole[0] = ImageView(canvas.data(), K_LOW_SIZES[0], 3);
    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(whole, plan, canvas)), "apply_mosaic: patch 0",
                    "destination holds the source's pixels");

    std::array<ImageView, K_MOSAIC_PATCHES> inner = view_all(patches);
    inner[2] = ImageView(canvas.row(20), K_LOW_SIZES[2], 3);
    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(inner, plan, canvas)), "apply_mosaic: patch 2",
                    "destination holds the source's pixels");
}

/** @brief Tests that apply_mosaic refuses a placement that reaches outside the canvas. */
TEST(Medium_ImagingMosaic, ApplyRejectsAPlacementOutsideTheCanvas)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    MosaicPlan outside = make_low_plan();
    outside.placements[1].canvas_x2 = 2 * K_SIDE + 1;
    Image canvas;

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(view_all(patches), outside, canvas)), "patch 1",
                    "outside a canvas");
}

/** @brief Tests that apply_mosaic refuses a placement that would resample its patch. */
TEST(Medium_ImagingMosaic, ApplyRejectsAPlacementThatResamples)
{
    const std::array<Image, K_MOSAIC_PATCHES> patches = make_dark_patches(K_LOW_SIZES);
    MosaicPlan narrower = make_low_plan();
    narrower.placements[3].patch_x2 = 7;
    Image canvas;

    SA_EXPECT_FATAL(static_cast<void>(apply_mosaic(view_all(patches), narrower, canvas)), "patch 3",
                    "resamples nothing");
}

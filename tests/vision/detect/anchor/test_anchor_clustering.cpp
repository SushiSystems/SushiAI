/**************************************************************************/
/* test_anchor_clustering.cpp                                             */
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
 * @file test_anchor_clustering.cpp
 * @brief Tests the anchors derived from seeded label boxes.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <support/splitmix64.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorClustering;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::cluster_anchors;
    using SushiAI::Vision::Detect::Anchor::ClusteringOptions;

    /** @brief Holds the seed of the jitter every ladder of this file is generated with. */
    constexpr std::uint64_t K_LADDER_SEED = 0xA03ull;

    /** @brief Holds the number of rungs of the ladder. */
    constexpr std::int64_t K_RUNGS = 6;

    /** @brief Holds the number of boxes generated at each rung. */
    constexpr std::size_t K_BOXES_PER_RUNG = 400;

    /** @brief Holds the largest share a side is jittered by. */
    constexpr double K_JITTER = 0.02;

    /** @brief Holds 2^53, the divisor that turns the upper 53 bits of a number into a unit. */
    constexpr double K_UNIT_DIVISOR = 9007199254740992.0;

    /**
     * @brief Holds the relative tolerance of a recovered centre's side: 2.5 times the 0.02 jitter,
     *        not a rounding bound.
     */
    constexpr double K_CENTRE_TOLERANCE = 0.05;

    /**
     * @brief Holds the absolute tolerance of a mean summed in double: 16777216 times the double
     *        unit roundoff, the smallest power of two not tighter than the earlier 1e-9.
     */
    constexpr double K_MEAN_TOLERANCE = 16777216.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Holds the absolute tolerance of an anchor that equals a box: 32 times the float unit
     *        roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_SIDE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds a ladder seed under test and the name of its row. */
    struct LadderSeed
    {
        std::string name;   /**< The row's name. */
        std::uint64_t seed; /**< The seed of the clustering. */
    };

    /** @brief Holds one options set the clustering refuses and the name of its row. */
    struct RefusedOptions
    {
        std::string name;          /**< The row's name. */
        ClusteringOptions options; /**< The options outside their range. */
    };

    /** @brief Runs the ladder under one clustering seed. */
    class Small_DetectAnchorClusteringLadder : public ::testing::TestWithParam<LadderSeed>
    {

    };

    /** @brief Runs one options set the clustering refuses. */
    class Small_DetectAnchorClusteringOptions : public ::testing::TestWithParam<RefusedOptions>
    {

    };

    /** @brief Names a ladder row by its name field. */
    std::string ladder_name(const ::testing::TestParamInfo<LadderSeed>& info)
    {
        return info.param.name;
    }

    /** @brief Names an options row by its name field. */
    std::string options_name(const ::testing::TestParamInfo<RefusedOptions>& info)
    {
        return info.param.name;
    }

    /** @brief Returns draw @p index of the ladder jitter as a double in [0, 1). */
    double draw_jitter_unit(std::uint64_t index)
    {
        return static_cast<double>(TestSupport::reference_splitmix64(K_LADDER_SEED, index) >> 11) /
               K_UNIT_DIVISOR;
    }

    /** @brief Returns the centre of rung @p rung: 0.03 x 0.05 times 1.8 per rung. */
    AnchorSize make_rung(std::int64_t rung)
    {
        const double scale = std::pow(1.8, static_cast<double>(rung));
        return AnchorSize{static_cast<float>(0.03 * scale), static_cast<float>(0.05 * scale)};
    }

    /** @brief Returns K_BOXES_PER_RUNG jittered boxes at each of K_RUNGS rungs, rung by rung. */
    std::vector<AnchorSize> make_ladder()
    {
        std::vector<AnchorSize> boxes;
        boxes.reserve(static_cast<std::size_t>(K_RUNGS) * K_BOXES_PER_RUNG);
        std::uint64_t draw = 0;
        for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
        {
            const double scale = std::pow(1.8, static_cast<double>(rung));
            for (std::size_t box = 0; box < K_BOXES_PER_RUNG; ++box)
            {
                const double width_factor =
                    1.0 + K_JITTER * (2.0 * draw_jitter_unit(draw++) - 1.0);
                const double height_factor =
                    1.0 + K_JITTER * (2.0 * draw_jitter_unit(draw++) - 1.0);
                boxes.push_back(AnchorSize{static_cast<float>(0.03 * scale * width_factor),
                                           static_cast<float>(0.05 * scale * height_factor)});
            }
        }
        return boxes;
    }

    /** @brief Returns the options of a clustering to K_RUNGS anchors from @p seed. */
    ClusteringOptions make_ladder_options(std::uint64_t seed)
    {
        ClusteringOptions options;
        options.count = K_RUNGS;
        options.seed = seed;
        return options;
    }

    /** @brief Returns the options of a clustering to one anchor, which the refusal rows alter. */
    ClusteringOptions make_single_options()
    {
        ClusteringOptions options;
        options.count = 1;
        return options;
    }

    /** @brief Expects anchor k of @p anchors within K_CENTRE_TOLERANCE of rung k, per side. */
    void expect_rungs_recovered(const std::vector<AnchorSize>& anchors)
    {
        ASSERT_EQ(anchors.size(), static_cast<std::size_t>(K_RUNGS));
        for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
        {
            const AnchorSize centre = make_rung(rung);
            const AnchorSize anchor = anchors[static_cast<std::size_t>(rung)];
            EXPECT_NEAR(anchor.width, centre.width, K_CENTRE_TOLERANCE * centre.width)
                << "rung " << rung;
            EXPECT_NEAR(anchor.height, centre.height, K_CENTRE_TOLERANCE * centre.height)
                << "rung " << rung;
        }
    }

    /** @brief Expects @p anchor to equal @p box on both sides within K_SIDE_TOLERANCE. */
    void expect_anchor_equals(const AnchorSize& anchor, const AnchorSize& box)
    {
        EXPECT_NEAR(anchor.width, box.width, K_SIDE_TOLERANCE);
        EXPECT_NEAR(anchor.height, box.height, K_SIDE_TOLERANCE);
    }

    /** @brief Returns the area of every anchor of @p anchors, in order. */
    std::vector<float> measure_areas(const std::vector<AnchorSize>& anchors)
    {
        std::vector<float> areas;
        areas.reserve(anchors.size());
        for (const AnchorSize& anchor : anchors)
            areas.push_back(anchor.width * anchor.height);
        return areas;
    }

    /** @brief Returns the bit pattern of @p value. */
    std::uint64_t read_bits(double value)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Returns the bit pattern of @p value. */
    std::uint32_t read_bits(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Returns the bits of every width and height of @p anchors, in order. */
    std::vector<std::uint32_t> read_anchor_bits(const std::vector<AnchorSize>& anchors)
    {
        std::vector<std::uint32_t> bits;
        bits.reserve(anchors.size() * 2);
        for (const AnchorSize& anchor : anchors)
        {
            bits.push_back(read_bits(anchor.width));
            bits.push_back(read_bits(anchor.height));
        }
        return bits;
    }

    /** @brief Returns the options outside their range, one row each. */
    std::vector<RefusedOptions> make_refused_options()
    {
        std::vector<RefusedOptions> rows;
        rows.push_back({"ACountOfZero", make_single_options()});
        rows.back().options.count = 0;
        rows.push_back({"ZeroIterations", make_single_options()});
        rows.back().options.max_iterations = 0;
        rows.push_back({"ANegativeSweepCount", make_single_options()});
        rows.back().options.max_refinement_sweeps = -1;
        rows.push_back({"ANegativeSmallestSide", make_single_options()});
        rows.back().options.min_side = -0.5f;
        rows.push_back({"ANaNSmallestSide", make_single_options()});
        rows.back().options.min_side = std::numeric_limits<float>::quiet_NaN();
        return rows;
    }
} // namespace

/** @brief Tests that the default seed and another seed each recover every rung within 5 %. */
TEST_P(Small_DetectAnchorClusteringLadder, EverySeedRecoversEveryRung)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const ClusteringOptions options = make_ladder_options(GetParam().seed);

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorClustering& clustering = outcome.value();
    expect_rungs_recovered(clustering.anchors);
    EXPECT_GT(clustering.mean_best_iou, 0.95);
    EXPECT_GE(clustering.mean_best_iou, clustering.mean_best_iou_before_refinement);
    EXPECT_EQ(clustering.boxes_used, 2400);
    EXPECT_EQ(clustering.boxes_rejected, 0);
    EXPECT_GE(clustering.iterations, 1);
    EXPECT_LE(clustering.iterations, options.max_iterations);
    EXPECT_LE(clustering.refinement_sweeps, options.max_refinement_sweeps);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DetectAnchorClusteringLadder,
                         ::testing::Values(LadderSeed{"TheDefaultSeed", ClusteringOptions{}.seed},
                                           LadderSeed{"AnotherSeed", 0x1234ull}),
                         ladder_name);

/** @brief Tests that the anchors come back in ascending area. */
TEST(Small_DetectAnchorClustering, AnchorsAreInAscendingArea)
{
    const std::vector<AnchorSize> boxes = make_ladder();

    const Result<AnchorClustering> outcome =
        cluster_anchors(boxes, make_ladder_options(0x5355534849ull));

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const std::vector<float> areas = measure_areas(outcome.value().anchors);
    EXPECT_EQ(std::adjacent_find(areas.begin(), areas.end(), std::greater_equal<float>()),
              areas.end());
}

/** @brief Tests that 1 % of the boxes at 0.9 x 0.9 leave every centre within 5 %. */
TEST(Small_DetectAnchorClustering, OutliersLeaveEveryCentreWithinFivePercent)
{
    std::vector<AnchorSize> boxes = make_ladder();
    boxes.insert(boxes.end(), boxes.size() / 100, AnchorSize{0.9f, 0.9f});

    const Result<AnchorClustering> outcome =
        cluster_anchors(boxes, make_ladder_options(0x5355534849ull));

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().boxes_used, 2424);
    expect_rungs_recovered(outcome.value().anchors);
    EXPECT_GE(outcome.value().mean_best_iou, outcome.value().mean_best_iou_before_refinement);
}

/** @brief Tests that two runs on the same boxes and options agree in every bit. */
TEST(Small_DetectAnchorClustering, TwoRunsAreBitIdentical)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const ClusteringOptions options = make_ladder_options(0x5355534849ull);

    const Result<AnchorClustering> first = cluster_anchors(boxes, options);
    const Result<AnchorClustering> second = cluster_anchors(boxes, options);

    ASSERT_TRUE(first.has_value()) << first.status().message();
    ASSERT_TRUE(second.has_value()) << second.status().message();
    EXPECT_EQ(read_anchor_bits(first.value().anchors), read_anchor_bits(second.value().anchors));
    EXPECT_EQ(read_bits(first.value().mean_best_iou), read_bits(second.value().mean_best_iou));
    EXPECT_EQ(read_bits(first.value().mean_best_iou_before_refinement),
              read_bits(second.value().mean_best_iou_before_refinement));
    EXPECT_EQ(first.value().iterations, second.value().iterations);
    EXPECT_EQ(first.value().refinement_sweeps, second.value().refinement_sweeps);
}

/** @brief Tests that no refinement sweep leaves the fit of the settled centres. */
TEST(Small_DetectAnchorClustering, ZeroSweepsKeepTheFitBeforeRefinement)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    ClusteringOptions options = make_ladder_options(0x5355534849ull);
    options.max_refinement_sweeps = 0;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().refinement_sweeps, 0);
    EXPECT_NEAR(outcome.value().mean_best_iou, outcome.value().mean_best_iou_before_refinement,
                K_MEAN_TOLERANCE);
}

/** @brief Tests that an empty box list is Incompatible. */
TEST(Small_DetectAnchorClustering, RefusesAnEmptyBoxList)
{
    const std::vector<AnchorSize> boxes;
    ClusteringOptions options;
    options.count = 1;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that count - 1 distinct boxes are Incompatible however often they repeat. */
TEST(Small_DetectAnchorClustering, RefusesFewerDistinctBoxesThanAnchors)
{
    const AnchorSize distinct[2] = {{0.1f, 0.2f}, {0.4f, 0.3f}};
    std::vector<AnchorSize> boxes;
    for (const AnchorSize& box : distinct)
        boxes.insert(boxes.end(), 10, box);
    ClusteringOptions options;
    options.count = 3;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that a zero side and a NaN side are left out and counted. */
TEST(Small_DetectAnchorClustering, LeavesOutAZeroAndANaNSide)
{
    const AnchorSize kept[3] = {{0.0625f, 0.125f}, {0.25f, 0.25f}, {0.5f, 0.75f}};
    std::vector<AnchorSize> boxes;
    for (const AnchorSize& box : kept)
        boxes.insert(boxes.end(), 5, box);
    boxes.push_back(AnchorSize{0.0f, 0.5f});
    boxes.push_back(AnchorSize{0.5f, std::numeric_limits<float>::quiet_NaN()});
    ClusteringOptions options;
    options.count = 3;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    // Three sizes and three anchors: each anchor is one size and every best IoU is 1.
    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorClustering& clustering = outcome.value();
    EXPECT_EQ(clustering.boxes_rejected, 2);
    EXPECT_EQ(clustering.boxes_used, 15);
    ASSERT_EQ(clustering.anchors.size(), 3u);
    expect_anchor_equals(clustering.anchors[0], kept[0]);
    expect_anchor_equals(clustering.anchors[1], kept[1]);
    expect_anchor_equals(clustering.anchors[2], kept[2]);
    EXPECT_NEAR(clustering.mean_best_iou, 1.0, K_MEAN_TOLERANCE);
}

/** @brief Tests that a box with a side at min_side is left out and one above it is kept. */
TEST(Small_DetectAnchorClustering, LeavesOutABoxAtTheSmallestSide)
{
    const std::vector<AnchorSize> boxes = {{0.125f, 0.5f}, {0.25f, 0.5f}, {0.5f, 0.125f}};
    ClusteringOptions options;
    options.count = 1;
    options.min_side = 0.125f;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().boxes_rejected, 2);
    EXPECT_EQ(outcome.value().boxes_used, 1);
    ASSERT_EQ(outcome.value().anchors.size(), 1u);
    expect_anchor_equals(outcome.value().anchors[0], AnchorSize{0.25f, 0.5f});
}

/** @brief Tests that an option outside its range is InvalidDescription. */
TEST_P(Small_DetectAnchorClusteringOptions, AnOptionOutsideItsRangeIsRefused)
{
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}, {0.5f, 0.25f}};

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, GetParam().options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DetectAnchorClusteringOptions,
                         ::testing::ValuesIn(make_refused_options()), options_name);

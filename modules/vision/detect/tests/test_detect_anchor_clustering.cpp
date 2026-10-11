/**************************************************************************/
/* test_detect_anchor_clustering.cpp                                      */
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
 * @file test_detect_anchor_clustering.cpp
 * @brief Tests the anchors derived from seeded label boxes and the sequence that seeds them.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

#include "../source/anchor/seeded_sequence.hpp"

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorClustering;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::cluster_anchors;
    using SushiAI::Vision::Detect::Anchor::ClusteringOptions;
    using SushiAI::Vision::Detect::Anchor::Detail::SeededSequence;

    /** @brief Holds the seed of the jitter every ladder of this file is generated with. */
    constexpr std::uint64_t K_LADDER_SEED = 0xA03ull;

    /** @brief Holds the number of rungs of the ladder. */
    constexpr std::int64_t K_RUNGS = 6;

    /** @brief Holds the number of boxes generated at each rung. */
    constexpr std::size_t K_BOXES_PER_RUNG = 400;

    /** @brief Holds the largest share a side is jittered by. */
    constexpr double K_JITTER = 0.02;

    /** @brief Holds the relative tolerance of a recovered centre's side. */
    constexpr double K_CENTRE_TOLERANCE = 0.05;

    /** @brief Holds the absolute tolerance of a mean summed in double. */
    constexpr double K_MEAN_TOLERANCE = 1e-9;

    /** @brief Holds the absolute tolerance of an anchor that equals a box. */
    constexpr float K_SIDE_TOLERANCE = 1e-6f;

    /** @brief Returns the centre of rung @p rung: 0.03 x 0.05 times 1.8 per rung. */
    AnchorSize make_rung(std::int64_t rung)
    {
        const double scale = std::pow(1.8, static_cast<double>(rung));
        return AnchorSize{static_cast<float>(0.03 * scale), static_cast<float>(0.05 * scale)};
    }

    /** @brief Returns K_BOXES_PER_RUNG jittered boxes at each of K_RUNGS rungs, rung by rung. */
    std::vector<AnchorSize> make_ladder()
    {
        SeededSequence jitter(K_LADDER_SEED);
        std::vector<AnchorSize> boxes;
        boxes.reserve(static_cast<std::size_t>(K_RUNGS) * K_BOXES_PER_RUNG);
        for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
        {
            const double scale = std::pow(1.8, static_cast<double>(rung));
            for (std::size_t box = 0; box < K_BOXES_PER_RUNG; ++box)
            {
                const double width_factor = 1.0 + K_JITTER * (2.0 * jitter.next_unit() - 1.0);
                const double height_factor = 1.0 + K_JITTER * (2.0 * jitter.next_unit() - 1.0);
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
} // namespace

/** @brief Tests the first eight numbers of seed 0 against the published SplitMix64 sequence. */
TEST(Unit_DetectAnchorClustering, SequenceOfSeedZeroMatchesThePublishedNumbers)
{
    const std::uint64_t published[8] = {
        0xE220A8397B1DCDAFull, 0x6E789E6AA1B965F4ull, 0x06C45D188009454Full,
        0xF88BB8A8724C81ECull, 0x1B39896A51A8749Bull, 0x53CB9F0C747EA2EAull,
        0x2C829ABE1F4532E1ull, 0xC584133AC916AB3Cull};

    SeededSequence sequence(0);

    for (const std::uint64_t expected : published)
        EXPECT_EQ(sequence.next(), expected);
}

/** @brief Tests that the unit draw is the upper 53 bits of the next number over 2^53. */
TEST(Unit_DetectAnchorClustering, UnitDrawIsTheUpperBitsOfTheNextNumber)
{
    SeededSequence sequence(0);

    const double unit = sequence.next_unit();

    // 0xE220A8397B1DCDAF is the first published number; 53 bits over 2^53 is exact in double.
    EXPECT_EQ(read_bits(unit), read_bits(static_cast<double>(0xE220A8397B1DCDAFull >> 11) /
                                         9007199254740992.0));
    EXPECT_GE(unit, 0.0);
    EXPECT_LT(unit, 1.0);
}

/** @brief Tests that the default seed recovers every rung of the ladder within 5 %. */
TEST(Unit_DetectAnchorClustering, LadderIsRecoveredWithinFivePercent)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    ClusteringOptions options;
    options.count = K_RUNGS;

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

/** @brief Tests that the anchors come back in ascending area. */
TEST(Unit_DetectAnchorClustering, AnchorsAreInAscendingArea)
{
    const std::vector<AnchorSize> boxes = make_ladder();

    const Result<AnchorClustering> outcome =
        cluster_anchors(boxes, make_ladder_options(0x5355534849ull));

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const std::vector<AnchorSize>& anchors = outcome.value().anchors;
    for (std::size_t index = 1; index < anchors.size(); ++index)
    {
        EXPECT_LT(anchors[index - 1].width * anchors[index - 1].height,
                  anchors[index].width * anchors[index].height);
    }
}

/** @brief Tests that another seed still recovers every rung within 5 %. */
TEST(Unit_DetectAnchorClustering, AnotherSeedStillRecoversTheLadder)
{
    const std::vector<AnchorSize> boxes = make_ladder();

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, make_ladder_options(0x1234));

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    expect_rungs_recovered(outcome.value().anchors);
    EXPECT_GT(outcome.value().mean_best_iou, 0.95);
}

/** @brief Tests that 1 % of the boxes at 0.9 x 0.9 leave every centre within 5 %. */
TEST(Unit_DetectAnchorClustering, OutliersLeaveEveryCentreWithinFivePercent)
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
TEST(Unit_DetectAnchorClustering, TwoRunsAreBitIdentical)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const ClusteringOptions options = make_ladder_options(0x5355534849ull);

    const Result<AnchorClustering> first = cluster_anchors(boxes, options);
    const Result<AnchorClustering> second = cluster_anchors(boxes, options);

    ASSERT_TRUE(first.has_value()) << first.status().message();
    ASSERT_TRUE(second.has_value()) << second.status().message();
    ASSERT_EQ(first.value().anchors.size(), second.value().anchors.size());
    for (std::size_t index = 0; index < first.value().anchors.size(); ++index)
    {
        EXPECT_EQ(read_bits(first.value().anchors[index].width),
                  read_bits(second.value().anchors[index].width));
        EXPECT_EQ(read_bits(first.value().anchors[index].height),
                  read_bits(second.value().anchors[index].height));
    }
    EXPECT_EQ(read_bits(first.value().mean_best_iou), read_bits(second.value().mean_best_iou));
    EXPECT_EQ(read_bits(first.value().mean_best_iou_before_refinement),
              read_bits(second.value().mean_best_iou_before_refinement));
    EXPECT_EQ(first.value().iterations, second.value().iterations);
    EXPECT_EQ(first.value().refinement_sweeps, second.value().refinement_sweeps);
}

/** @brief Tests that no refinement sweep leaves the fit of the settled centres. */
TEST(Unit_DetectAnchorClustering, ZeroSweepsKeepTheFitBeforeRefinement)
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
TEST(Unit_DetectAnchorClustering, RefusesAnEmptyBoxList)
{
    const std::vector<AnchorSize> boxes;
    ClusteringOptions options;
    options.count = 1;

    const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that count - 1 distinct boxes are Incompatible however often they repeat. */
TEST(Unit_DetectAnchorClustering, RefusesFewerDistinctBoxesThanAnchors)
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
TEST(Unit_DetectAnchorClustering, LeavesOutAZeroAndANaNSide)
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
    for (std::size_t index = 0; index < 3; ++index)
    {
        EXPECT_NEAR(clustering.anchors[index].width, kept[index].width, K_SIDE_TOLERANCE);
        EXPECT_NEAR(clustering.anchors[index].height, kept[index].height, K_SIDE_TOLERANCE);
    }
    EXPECT_NEAR(clustering.mean_best_iou, 1.0, K_MEAN_TOLERANCE);
}

/** @brief Tests that a box with a side at min_side is left out and one above it is kept. */
TEST(Unit_DetectAnchorClustering, LeavesOutABoxAtTheSmallestSide)
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
    EXPECT_NEAR(outcome.value().anchors[0].width, 0.25f, K_SIDE_TOLERANCE);
    EXPECT_NEAR(outcome.value().anchors[0].height, 0.5f, K_SIDE_TOLERANCE);
}

/** @brief Tests that each option outside its range is InvalidDescription. */
TEST(Unit_DetectAnchorClustering, RefusesOptionsOutsideTheirRange)
{
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}, {0.5f, 0.25f}};
    ClusteringOptions valid;
    valid.count = 1;
    std::vector<ClusteringOptions> refused(5, valid);
    refused[0].count = 0;
    refused[1].max_iterations = 0;
    refused[2].max_refinement_sweeps = -1;
    refused[3].min_side = -0.5f;
    refused[4].min_side = std::numeric_limits<float>::quiet_NaN();

    for (const ClusteringOptions& options : refused)
    {
        const Result<AnchorClustering> outcome = cluster_anchors(boxes, options);

        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    }
}

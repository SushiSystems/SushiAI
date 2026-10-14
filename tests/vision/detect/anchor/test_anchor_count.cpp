/**************************************************************************/
/* test_anchor_count.cpp                                                  */
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
 * @file test_anchor_count.cpp
 * @brief Tests the number of anchors per head chosen from seeded label boxes.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_count.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorClustering;
    using SushiAI::Vision::Detect::Anchor::AnchorCountChoice;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::choose_anchor_count;
    using SushiAI::Vision::Detect::Anchor::cluster_anchors;
    using SushiAI::Vision::Detect::Anchor::ClusteringOptions;
    using SushiAI::Vision::Detect::Anchor::CountOptions;
    using SushiAI::Vision::Detect::Anchor::shape_iou;

    /** @brief Holds the seed of the jitter the ladder is generated with. */
    constexpr std::uint64_t K_LADDER_SEED = 0xA05ull;

    /** @brief Holds the seed the blob is generated with. */
    constexpr std::uint64_t K_BLOB_SEED = 0xB10Bull;

    /** @brief Holds the number of rungs of the ladder. */
    constexpr std::int64_t K_RUNGS = 6;

    /** @brief Holds the number of boxes generated at each rung. */
    constexpr std::size_t K_BOXES_PER_RUNG = 400;

    /** @brief Holds the largest share a side of the ladder is jittered by. */
    constexpr double K_JITTER = 0.02;

    /** @brief Holds the number of boxes of the blob. */
    constexpr std::size_t K_BLOB_BOXES = 2000;

    /** @brief Holds the largest factor a side of the blob lies from its centre by. */
    constexpr double K_BLOB_FACTOR = 1.1;

    /** @brief Holds the absolute tolerance of a mean summed in double. */
    constexpr double K_MEAN_TOLERANCE = 1e-9;

    /** @brief Holds the absolute tolerance of a mean of shape_iou values computed in float. */
    constexpr double K_FLOAT_MEAN_TOLERANCE = 1e-6;

    /** @brief Returns the next draw of @p engine as a double in [0, 1), from its upper 53 bits. */
    double draw_unit(std::mt19937_64& engine)
    {
        return static_cast<double>(engine() >> 11) / 9007199254740992.0;
    }

    /** @brief Returns K_BOXES_PER_RUNG jittered boxes at each of K_RUNGS rungs, rung by rung. */
    std::vector<AnchorSize> make_ladder()
    {
        std::mt19937_64 engine(K_LADDER_SEED);
        std::vector<AnchorSize> boxes;
        boxes.reserve(static_cast<std::size_t>(K_RUNGS) * K_BOXES_PER_RUNG);
        for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
        {
            const double scale = std::pow(1.8, static_cast<double>(rung));
            for (std::size_t box = 0; box < K_BOXES_PER_RUNG; ++box)
            {
                const double width_factor = 1.0 + K_JITTER * (2.0 * draw_unit(engine) - 1.0);
                const double height_factor = 1.0 + K_JITTER * (2.0 * draw_unit(engine) - 1.0);
                boxes.push_back(AnchorSize{static_cast<float>(0.03 * scale * width_factor),
                                           static_cast<float>(0.05 * scale * height_factor)});
            }
        }
        return boxes;
    }

    /** @brief Returns K_BLOB_BOXES boxes, each side log-uniform within 10 % of 0.2 x 0.3. */
    std::vector<AnchorSize> make_blob()
    {
        std::mt19937_64 engine(K_BLOB_SEED);
        const double reach = std::log(K_BLOB_FACTOR);
        std::vector<AnchorSize> boxes;
        boxes.reserve(K_BLOB_BOXES);
        for (std::size_t box = 0; box < K_BLOB_BOXES; ++box)
        {
            const double width = 0.2 * std::exp(reach * (2.0 * draw_unit(engine) - 1.0));
            const double height = 0.3 * std::exp(reach * (2.0 * draw_unit(engine) - 1.0));
            boxes.push_back(AnchorSize{static_cast<float>(width), static_cast<float>(height)});
        }
        return boxes;
    }

    /** @brief Returns ten boxes of 0.1 x 0.1 followed by eleven of 0.4 x 0.4. */
    std::vector<AnchorSize> make_two_sizes()
    {
        std::vector<AnchorSize> boxes(10, AnchorSize{0.1f, 0.1f});
        boxes.insert(boxes.end(), 11, AnchorSize{0.4f, 0.4f});
        return boxes;
    }

    /** @brief Returns the mean over @p boxes of the best shape_iou with any of @p anchors. */
    double measure_mean_best_iou(const std::vector<AnchorSize>& boxes,
                                 const std::vector<AnchorSize>& anchors)
    {
        double sum = 0.0;
        for (const AnchorSize& box : boxes)
        {
            float best = 0.0f;
            for (const AnchorSize& anchor : anchors)
                best = std::max(best, shape_iou(box, anchor));
            sum += static_cast<double>(best);
        }
        return sum / static_cast<double>(boxes.size());
    }
} // namespace

/** @brief Tests that six rungs and two heads give three anchors per head after four trials. */
TEST(Unit_DetectAnchorCount, LadderWithTwoHeadsChoosesThreePerHead)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const CountOptions options;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 2, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    EXPECT_EQ(choice.per_head, 3);
    EXPECT_EQ(choice.clustering.anchors.size(), 6u);
    ASSERT_EQ(choice.trials.size(), 4u);
    for (std::size_t trial = 0; trial < choice.trials.size(); ++trial)
        EXPECT_EQ(choice.trials[trial].per_head, static_cast<std::int64_t>(trial) + 1);
    EXPECT_GE(choice.trials[1].mean_best_iou - choice.trials[0].mean_best_iou, options.min_gain);
    EXPECT_GE(choice.trials[2].mean_best_iou - choice.trials[1].mean_best_iou, options.min_gain);
    EXPECT_LT(choice.trials[3].mean_best_iou - choice.trials[2].mean_best_iou, options.min_gain);
}

/** @brief Tests that each trial holds the fit of a clustering to that count made directly. */
TEST(Unit_DetectAnchorCount, TrialsHoldTheFitOfEachCountClusteredDirectly)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::int64_t heads = 2;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, heads, CountOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    for (const auto& trial : choice.trials)
    {
        ClusteringOptions direct;
        direct.count = trial.per_head * heads;
        const Result<AnchorClustering> clustering = cluster_anchors(boxes, direct);

        ASSERT_TRUE(clustering.has_value()) << clustering.status().message();
        EXPECT_NEAR(trial.mean_best_iou, clustering.value().mean_best_iou, K_MEAN_TOLERANCE)
            << "per head " << trial.per_head;
    }
    EXPECT_NEAR(choice.clustering.mean_best_iou,
                measure_mean_best_iou(boxes, choice.clustering.anchors), K_MEAN_TOLERANCE);
    EXPECT_NEAR(choice.clustering.mean_best_iou,
                choice.trials[static_cast<std::size_t>(choice.per_head) - 1].mean_best_iou,
                K_MEAN_TOLERANCE);
}

/** @brief Tests that one blob and three heads give one anchor per head after two trials. */
TEST(Unit_DetectAnchorCount, BlobChoosesOnePerHead)
{
    const std::vector<AnchorSize> boxes = make_blob();
    const CountOptions options;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 3, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    EXPECT_EQ(choice.per_head, 1);
    EXPECT_EQ(choice.clustering.anchors.size(), 3u);
    ASSERT_EQ(choice.trials.size(), 2u);
    EXPECT_EQ(choice.trials[0].per_head, 1);
    EXPECT_EQ(choice.trials[1].per_head, 2);
    EXPECT_LT(choice.trials[1].mean_best_iou - choice.trials[0].mean_best_iou, options.min_gain);
}

/** @brief Tests that a max_per_head of 2 stops the ladder at two anchors per head. */
TEST(Unit_DetectAnchorCount, MaxPerHeadOfTwoStopsAtTwo)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    CountOptions options;
    options.max_per_head = 2;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 2, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    EXPECT_EQ(choice.per_head, 2);
    EXPECT_EQ(choice.clustering.anchors.size(), 4u);
    ASSERT_EQ(choice.trials.size(), 2u);
    EXPECT_EQ(choice.trials[1].per_head, 2);
    EXPECT_NEAR(choice.clustering.mean_best_iou, choice.trials[1].mean_best_iou,
                K_MEAN_TOLERANCE);
}

/** @brief Tests that equal bounds make one trial and try no successor. */
TEST(Unit_DetectAnchorCount, EqualBoundsMakeOneTrial)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    CountOptions options;
    options.min_per_head = 2;
    options.max_per_head = 2;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 3, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().per_head, 2);
    EXPECT_EQ(outcome.value().clustering.anchors.size(), 6u);
    ASSERT_EQ(outcome.value().trials.size(), 1u);
    EXPECT_EQ(outcome.value().trials[0].per_head, 2);
}

/** @brief Tests two sizes and one head: a second anchor gains 9.375 / 21 and is taken. */
TEST(Unit_DetectAnchorCount, TwoSizesTakeTheSecondAnchorAtTheDefaultGain)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();
    CountOptions options;
    options.max_per_head = 2;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, options);

    // One anchor is the median 0.4 x 0.4: eleven boxes at 1 and ten at 0.01 / 0.16.
    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    EXPECT_EQ(choice.per_head, 2);
    ASSERT_EQ(choice.trials.size(), 2u);
    EXPECT_NEAR(choice.trials[0].mean_best_iou, 11.625 / 21.0, K_FLOAT_MEAN_TOLERANCE);
    EXPECT_NEAR(choice.trials[1].mean_best_iou, 1.0, K_FLOAT_MEAN_TOLERANCE);
    EXPECT_NEAR(choice.clustering.mean_best_iou, 1.0, K_FLOAT_MEAN_TOLERANCE);
}

/** @brief Tests that a min_gain above 9.375 / 21 keeps one anchor and returns both trials. */
TEST(Unit_DetectAnchorCount, TwoSizesKeepOneAnchorWhenTheGainIsBelowMinGain)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();
    CountOptions options;
    options.max_per_head = 2;
    options.min_gain = 0.5;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorCountChoice& choice = outcome.value();
    EXPECT_EQ(choice.per_head, 1);
    ASSERT_EQ(choice.clustering.anchors.size(), 1u);
    ASSERT_EQ(choice.trials.size(), 2u);
    EXPECT_NEAR(choice.clustering.mean_best_iou, 11.625 / 21.0, K_FLOAT_MEAN_TOLERANCE);
    EXPECT_NEAR(choice.trials[1].mean_best_iou, 1.0, K_FLOAT_MEAN_TOLERANCE);
}

/** @brief Tests that the count of the clustering options is not read. */
TEST(Unit_DetectAnchorCount, IgnoresTheCountOfTheClusteringOptions)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();
    CountOptions options;
    options.max_per_head = 2;
    options.clustering.count = 17;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().per_head, 2);
    EXPECT_EQ(outcome.value().clustering.anchors.size(), 2u);
}

/** @brief Tests that a min_per_head above max_per_head is InvalidDescription. */
TEST(Unit_DetectAnchorCount, RefusesAMinPerHeadAboveMaxPerHead)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();
    CountOptions options;
    options.min_per_head = 3;
    options.max_per_head = 2;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

/** @brief Tests that each option outside its range is InvalidDescription. */
TEST(Unit_DetectAnchorCount, RefusesOptionsOutsideTheirRange)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();
    std::vector<CountOptions> refused(4);
    refused[0].min_per_head = 0;
    refused[1].min_gain = -0.01;
    refused[2].min_gain = std::numeric_limits<double>::quiet_NaN();
    refused[3].clustering.max_iterations = 0;

    for (const CountOptions& options : refused)
    {
        const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, options);

        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    }
}

/** @brief Tests that no head and a head count whose product overflows are InvalidDescription. */
TEST(Unit_DetectAnchorCount, RefusesAHeadCountOutsideItsRange)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();

    for (const std::int64_t heads : {std::int64_t{0}, std::int64_t{-1},
                                     std::numeric_limits<std::int64_t>::max()})
    {
        const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, heads, CountOptions{});

        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    }
}

/** @brief Tests that fewer distinct boxes than the first count tried are Incompatible. */
TEST(Unit_DetectAnchorCount, RefusesFewerDistinctBoxesThanTheFirstCount)
{
    const std::vector<AnchorSize> boxes = make_two_sizes();

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 3, CountOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that an empty box list is Incompatible. */
TEST(Unit_DetectAnchorCount, RefusesAnEmptyBoxList)
{
    const std::vector<AnchorSize> boxes;

    const Result<AnchorCountChoice> outcome = choose_anchor_count(boxes, 1, CountOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/**************************************************************************/
/* test_detect_anchor_plan.cpp                                            */
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
 * @file test_detect_anchor_plan.cpp
 * @brief Tests the anchoring planned for a new model from seeded label boxes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_count.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_plan.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorClustering;
    using SushiAI::Vision::Detect::Anchor::AnchorCountChoice;
    using SushiAI::Vision::Detect::Anchor::AnchorFit;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorPlan;
    using SushiAI::Vision::Detect::Anchor::AnchorPlanOptions;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::assign_anchors_to_heads;
    using SushiAI::Vision::Detect::Anchor::AssignmentThresholds;
    using SushiAI::Vision::Detect::Anchor::choose_anchor_count;
    using SushiAI::Vision::Detect::Anchor::cluster_anchors;
    using SushiAI::Vision::Detect::Anchor::ClusteringOptions;
    using SushiAI::Vision::Detect::Anchor::CountTrial;
    using SushiAI::Vision::Detect::Anchor::derive_assignment_thresholds;
    using SushiAI::Vision::Detect::Anchor::measure_anchor_fit;
    using SushiAI::Vision::Detect::Anchor::plan_anchors;

    /** @brief Holds the seed of the jitter the ladder is generated with. */
    constexpr std::uint64_t K_LADDER_SEED = 0xA07ull;

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

    /** @brief Holds the absolute tolerance between a plan and the same steps made by hand. */
    constexpr double K_SAME_TOLERANCE = 1e-9;

    /** @brief Holds the absolute tolerance between two plans of the same arguments: none. */
    constexpr double K_IDENTICAL = 0.0;

    /** @brief Holds the absolute tolerance of a threshold, a float32. */
    constexpr double K_THRESHOLD_TOLERANCE = 1e-6;

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

    /** @brief Returns the bit pattern of @p value. */
    std::uint32_t read_bits(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Expects two layouts to hold the same anchors, bit for bit, and the same heads. */
    void expect_same_layout(const AnchorLayout& actual, const AnchorLayout& expected)
    {
        ASSERT_EQ(actual.anchors.size(), expected.anchors.size());
        for (std::size_t index = 0; index < actual.anchors.size(); ++index)
        {
            EXPECT_EQ(read_bits(actual.anchors[index].width),
                      read_bits(expected.anchors[index].width))
                << "anchor " << index;
            EXPECT_EQ(read_bits(actual.anchors[index].height),
                      read_bits(expected.anchors[index].height))
                << "anchor " << index;
        }
        EXPECT_EQ(actual.head_anchors, expected.head_anchors);
    }

    /** @brief Expects every number of @p actual within @p tolerance of @p expected. */
    void expect_same_fit(const AnchorFit& actual, const AnchorFit& expected, double tolerance)
    {
        EXPECT_EQ(actual.boxes, expected.boxes);
        EXPECT_NEAR(actual.mean_best_iou, expected.mean_best_iou, tolerance);
        EXPECT_NEAR(actual.best_possible_recall, expected.best_possible_recall, tolerance);
        EXPECT_NEAR(actual.unowned_share, expected.unowned_share, tolerance);
        EXPECT_NEAR(actual.best_iou_tenth_percentile, expected.best_iou_tenth_percentile,
                    tolerance);
        ASSERT_EQ(actual.anchor_share.size(), expected.anchor_share.size());
        for (std::size_t index = 0; index < actual.anchor_share.size(); ++index)
        {
            EXPECT_NEAR(actual.anchor_share[index], expected.anchor_share[index], tolerance)
                << "anchor " << index;
        }
    }

    /** @brief Expects both thresholds of @p actual within @p tolerance of @p expected. */
    void expect_same_thresholds(const AssignmentThresholds& actual,
                                const AssignmentThresholds& expected, double tolerance)
    {
        EXPECT_NEAR(actual.match_threshold, expected.match_threshold, tolerance);
        EXPECT_NEAR(actual.ignore_threshold, expected.ignore_threshold, tolerance);
    }

    /** @brief Expects the same counts in the same order, each fit within @p tolerance. */
    void expect_same_trials(const std::vector<CountTrial>& actual,
                            const std::vector<CountTrial>& expected, double tolerance)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            EXPECT_EQ(actual[index].per_head, expected[index].per_head) << "trial " << index;
            EXPECT_NEAR(actual[index].mean_best_iou, expected[index].mean_best_iou, tolerance)
                << "trial " << index;
        }
    }

    /** @brief Expects every part of @p actual to equal that of @p expected, to @p tolerance. */
    void expect_same_plan(const AnchorPlan& actual, const AnchorPlan& expected, double tolerance)
    {
        expect_same_layout(actual.layout, expected.layout);
        expect_same_fit(actual.fit, expected.fit, tolerance);
        expect_same_thresholds(actual.thresholds, expected.thresholds, tolerance);
        expect_same_trials(actual.trials, expected.trials, tolerance);
    }
} // namespace

/** @brief Tests six rungs under strides 16 and 32: two heads of three anchors, all recalled. */
TEST(Unit_DetectAnchorPlan, LadderWithTwoStridesPlansTwoHeadsOfThreeAnchors)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};

    const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, AnchorPlanOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorPlan& plan = outcome.value();
    ASSERT_EQ(plan.layout.anchors.size(), 6u);
    ASSERT_EQ(plan.layout.head_anchors.size(), 2u);
    EXPECT_EQ(plan.layout.head_anchors[0], (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_EQ(plan.layout.head_anchors[1], (std::vector<std::int64_t>{3, 4, 5}));
    EXPECT_EQ(plan.fit.boxes, static_cast<std::int64_t>(boxes.size()));
    EXPECT_NEAR(plan.fit.best_possible_recall, 1.0, K_SAME_TOLERANCE);
    EXPECT_NEAR(plan.fit.unowned_share, 0.0, K_SAME_TOLERANCE);
    EXPECT_GT(plan.fit.mean_best_iou, 0.95);
    expect_same_thresholds(plan.thresholds, derive_assignment_thresholds(plan.fit),
                           K_THRESHOLD_TOLERANCE);
    EXPECT_NEAR(plan.thresholds.ignore_threshold, 0.7, K_THRESHOLD_TOLERANCE);
    EXPECT_NEAR(plan.thresholds.match_threshold, plan.fit.best_iou_tenth_percentile,
                K_THRESHOLD_TOLERANCE);
    ASSERT_EQ(plan.trials.size(), 4u);
    EXPECT_EQ(plan.trials.back().per_head, 4);
}

/** @brief Tests that a chosen count gives what the four steps give when called one by one. */
TEST(Unit_DetectAnchorPlan, ChosenCountEqualsTheStepsMadeByHand)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorPlanOptions options;

    const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, options);

    const Result<AnchorCountChoice> choice = choose_anchor_count(boxes, 2, options.count);
    ASSERT_TRUE(choice.has_value()) << choice.status().message();
    AnchorPlan expected;
    const Result<AnchorLayout> layout =
        assign_anchors_to_heads(choice.value().clustering.anchors, strides, {});
    ASSERT_TRUE(layout.has_value()) << layout.status().message();
    expected.layout = layout.value();
    const Result<AnchorFit> fit = measure_anchor_fit(boxes, expected.layout, options.fit);
    ASSERT_TRUE(fit.has_value()) << fit.status().message();
    expected.fit = fit.value();
    expected.thresholds = derive_assignment_thresholds(expected.fit);
    expected.trials = choice.value().trials;

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    expect_same_plan(outcome.value(), expected, K_SAME_TOLERANCE);
}

/** @brief Tests that a per_head of 3 clusters six anchors once and returns no trial. */
TEST(Unit_DetectAnchorPlan, GivenCountTriesNoCount)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};
    AnchorPlanOptions options;
    options.per_head = 3;

    const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, options);

    ClusteringOptions direct = options.count.clustering;
    direct.count = 6;
    const Result<AnchorClustering> clustering = cluster_anchors(boxes, direct);
    ASSERT_TRUE(clustering.has_value()) << clustering.status().message();
    const Result<AnchorLayout> layout =
        assign_anchors_to_heads(clustering.value().anchors, strides, {});
    ASSERT_TRUE(layout.has_value()) << layout.status().message();

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorPlan& plan = outcome.value();
    EXPECT_TRUE(plan.trials.empty());
    expect_same_layout(plan.layout, layout.value());
    ASSERT_EQ(plan.layout.head_anchors.size(), 2u);
    EXPECT_EQ(plan.layout.head_anchors[0].size(), 3u);
    EXPECT_EQ(plan.layout.head_anchors[1].size(), 3u);
    EXPECT_NEAR(plan.fit.best_possible_recall, 1.0, K_SAME_TOLERANCE);
}

/** @brief Tests that a given count reads none of the options of the count choice. */
TEST(Unit_DetectAnchorPlan, GivenCountIgnoresTheBoundsOfTheCountChoice)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};
    AnchorPlanOptions options;
    options.per_head = 3;
    AnchorPlanOptions unread = options;
    unread.count.min_per_head = 0;
    unread.count.max_per_head = -1;
    unread.count.min_gain = std::numeric_limits<double>::quiet_NaN();
    unread.count.clustering.count = 17;

    const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, unread);
    const Result<AnchorPlan> plain = plan_anchors(boxes, strides, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    ASSERT_TRUE(plain.has_value()) << plain.status().message();
    expect_same_plan(outcome.value(), plain.value(), K_IDENTICAL);
}

/** @brief Tests that a head takes the anchors of its own stride wherever the stride stands. */
TEST(Unit_DetectAnchorPlan, StridesInAnyOrderGiveEachHeadTheAnchorsOfItsStride)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> ascending = {16, 32};
    const std::vector<std::int64_t> descending = {32, 16};
    const std::vector<std::int64_t> mixed = {16, 64, 32};
    AnchorPlanOptions two_per_head;
    two_per_head.per_head = 2;

    const Result<AnchorPlan> forward = plan_anchors(boxes, ascending, AnchorPlanOptions{});
    const Result<AnchorPlan> backward = plan_anchors(boxes, descending, AnchorPlanOptions{});
    const Result<AnchorPlan> three = plan_anchors(boxes, mixed, two_per_head);

    ASSERT_TRUE(forward.has_value()) << forward.status().message();
    ASSERT_TRUE(backward.has_value()) << backward.status().message();
    ASSERT_TRUE(three.has_value()) << three.status().message();
    ASSERT_EQ(backward.value().layout.head_anchors.size(), 2u);
    EXPECT_EQ(backward.value().layout.head_anchors[0], (std::vector<std::int64_t>{3, 4, 5}));
    EXPECT_EQ(backward.value().layout.head_anchors[1], (std::vector<std::int64_t>{0, 1, 2}));
    AnchorPlan swapped = backward.value();
    std::swap(swapped.layout.head_anchors[0], swapped.layout.head_anchors[1]);
    expect_same_plan(swapped, forward.value(), K_IDENTICAL);
    ASSERT_EQ(three.value().layout.head_anchors.size(), 3u);
    EXPECT_EQ(three.value().layout.head_anchors[0], (std::vector<std::int64_t>{0, 1}));
    EXPECT_EQ(three.value().layout.head_anchors[1], (std::vector<std::int64_t>{4, 5}));
    EXPECT_EQ(three.value().layout.head_anchors[2], (std::vector<std::int64_t>{2, 3}));
}

/** @brief Tests that two calls with the same arguments give the same plan, bit for bit. */
TEST(Unit_DetectAnchorPlan, TwoCallsGiveTheSamePlan)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};
    AnchorPlanOptions seeded;
    seeded.count.clustering.seed = 0x5EEDull;

    const Result<AnchorPlan> first = plan_anchors(boxes, strides, seeded);
    const Result<AnchorPlan> second = plan_anchors(boxes, strides, seeded);

    ASSERT_TRUE(first.has_value()) << first.status().message();
    ASSERT_TRUE(second.has_value()) << second.status().message();
    expect_same_plan(first.value(), second.value(), K_IDENTICAL);
}

/** @brief Tests that one blob under three strides plans one anchor per head. */
TEST(Unit_DetectAnchorPlan, BlobPlansOneAnchorPerHead)
{
    const std::vector<AnchorSize> boxes = make_blob();
    const std::vector<std::int64_t> strides = {8, 16, 32};

    const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, AnchorPlanOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorPlan& plan = outcome.value();
    ASSERT_EQ(plan.layout.head_anchors.size(), 3u);
    EXPECT_EQ(plan.layout.head_anchors[0], (std::vector<std::int64_t>{0}));
    EXPECT_EQ(plan.layout.head_anchors[1], (std::vector<std::int64_t>{1}));
    EXPECT_EQ(plan.layout.head_anchors[2], (std::vector<std::int64_t>{2}));
    ASSERT_EQ(plan.trials.size(), 2u);
    EXPECT_NEAR(plan.fit.best_possible_recall, 1.0, K_SAME_TOLERANCE);
}

/** @brief Tests that no stride and a stride of zero are InvalidDescription. */
TEST(Unit_DetectAnchorPlan, RefusesNoStrideAndAStrideOfZero)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> none;
    const std::vector<std::int64_t> zero = {16, 0};
    AnchorPlanOptions one_per_head;
    one_per_head.per_head = 1;

    for (const AnchorPlanOptions& options : {AnchorPlanOptions{}, one_per_head})
    {
        const Result<AnchorPlan> without = plan_anchors(boxes, none, options);
        const Result<AnchorPlan> with_zero = plan_anchors(boxes, zero, options);

        ASSERT_FALSE(without.has_value());
        EXPECT_EQ(without.status().code(), StatusCode::InvalidDescription);
        ASSERT_FALSE(with_zero.has_value());
        EXPECT_EQ(with_zero.status().code(), StatusCode::InvalidDescription);
    }
}

/** @brief Tests that each option outside its range is InvalidDescription. */
TEST(Unit_DetectAnchorPlan, RefusesOptionsOutsideTheirRange)
{
    const std::vector<AnchorSize> boxes = make_ladder();
    const std::vector<std::int64_t> strides = {16, 32};
    std::vector<AnchorPlanOptions> refused(6);
    refused[0].per_head = -1;
    refused[1].per_head = std::numeric_limits<std::int64_t>::max();
    refused[2].count.min_per_head = 0;
    refused[3].count.clustering.max_iterations = 0;
    refused[4].fit.recall_iou = 2.0f;
    refused[5].per_head = 1;
    refused[5].count.clustering.max_iterations = 0;

    for (std::size_t index = 0; index < refused.size(); ++index)
    {
        const Result<AnchorPlan> outcome = plan_anchors(boxes, strides, refused[index]);

        ASSERT_FALSE(outcome.has_value()) << "options " << index;
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription) << "options " << index;
    }
}

/** @brief Tests that no box and fewer distinct boxes than a count needs are Incompatible. */
TEST(Unit_DetectAnchorPlan, RefusesBoxesThatDisagreeWithTheCount)
{
    const std::vector<AnchorSize> empty;
    const std::vector<AnchorSize> two_sizes = make_two_sizes();
    const std::vector<std::int64_t> two_strides = {16, 32};
    const std::vector<std::int64_t> three_strides = {8, 16, 32};
    AnchorPlanOptions two_per_head;
    two_per_head.per_head = 2;

    const Result<AnchorPlan> no_box = plan_anchors(empty, two_strides, AnchorPlanOptions{});
    const Result<AnchorPlan> no_box_given = plan_anchors(empty, two_strides, two_per_head);
    const Result<AnchorPlan> chosen = plan_anchors(two_sizes, three_strides, AnchorPlanOptions{});
    const Result<AnchorPlan> given = plan_anchors(two_sizes, two_strides, two_per_head);

    for (const Result<AnchorPlan>* outcome : {&no_box, &no_box_given, &chosen, &given})
    {
        ASSERT_FALSE(outcome->has_value());
        EXPECT_EQ(outcome->status().code(), StatusCode::Incompatible);
    }
}

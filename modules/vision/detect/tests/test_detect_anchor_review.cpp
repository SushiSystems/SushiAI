/**************************************************************************/
/* test_detect_anchor_review.cpp                                          */
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
 * @file test_detect_anchor_review.cpp
 * @brief Tests the review of an anchor layout on seeded label boxes and its proposal.
 * @author Mustafa Garip
 */

#include <algorithm>
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
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_review.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

#include "../source/anchor/seeded_sequence.hpp"

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorClustering;
    using SushiAI::Vision::Detect::Anchor::AnchorFit;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorReview;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::AnchorVerdict;
    using SushiAI::Vision::Detect::Anchor::assign_anchors_to_heads;
    using SushiAI::Vision::Detect::Anchor::cluster_anchors;
    using SushiAI::Vision::Detect::Anchor::ClusteringOptions;
    using SushiAI::Vision::Detect::Anchor::measure_anchor_fit;
    using SushiAI::Vision::Detect::Anchor::review_anchors;
    using SushiAI::Vision::Detect::Anchor::ReviewOptions;
    using SushiAI::Vision::Detect::Anchor::Detail::SeededSequence;

    /** @brief Holds the seed of the jitter every ladder of this file is generated with. */
    constexpr std::uint64_t K_LADDER_SEED = 0xA06ull;

    /** @brief Holds the number of rungs of the ladder. */
    constexpr std::int64_t K_RUNGS = 6;

    /** @brief Holds the number of boxes generated at each rung. */
    constexpr std::size_t K_BOXES_PER_RUNG = 400;

    /** @brief Holds the largest share a side is jittered by. */
    constexpr double K_JITTER = 0.02;

    /** @brief Holds the factor on every side of the shifted ladder. */
    constexpr double K_SHIFT = 0.75;

    /** @brief Holds the relative tolerance of a proposed anchor's side. */
    constexpr double K_CENTRE_TOLERANCE = 0.05;

    /** @brief Holds the absolute tolerance of a fit against the test's own sums in double. */
    constexpr double K_FIT_TOLERANCE = 1e-6;

    /** @brief Holds the absolute tolerance of two runs of one computation. */
    constexpr double K_SAME_TOLERANCE = 1e-12;

    /** @brief Returns the ladder of SPEC 5.6 with every side times @p factor, rung by rung. */
    std::vector<AnchorSize> make_ladder(double factor)
    {
        SeededSequence jitter(K_LADDER_SEED);
        std::vector<AnchorSize> boxes;
        boxes.reserve(static_cast<std::size_t>(K_RUNGS) * K_BOXES_PER_RUNG);
        for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
        {
            const double scale = factor * std::pow(1.8, static_cast<double>(rung));
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

    /** @brief Returns K_RUNGS anchors clustered from @p boxes and assigned to @p per_head. */
    AnchorLayout fit_layout(const std::vector<AnchorSize>& boxes,
                            const std::vector<std::int64_t>& strides,
                            const std::vector<std::int64_t>& per_head)
    {
        ClusteringOptions options;
        options.count = K_RUNGS;
        const Result<AnchorClustering> clustering = cluster_anchors(boxes, options);
        EXPECT_TRUE(clustering.has_value()) << clustering.status().message();
        if (!clustering.has_value())
            return AnchorLayout{};

        const Result<AnchorLayout> layout =
            assign_anchors_to_heads(clustering.value().anchors, strides, per_head);
        EXPECT_TRUE(layout.has_value()) << layout.status().message();
        return layout.has_value() ? layout.value() : AnchorLayout{};
    }

    /** @brief Returns the IoU in double of two sizes laid on one centre. */
    double measure_overlap(AnchorSize a, AnchorSize b)
    {
        const double shared = static_cast<double>(std::min(a.width, b.width)) *
                              static_cast<double>(std::min(a.height, b.height));
        const double joined = static_cast<double>(a.width) * static_cast<double>(a.height) +
                              static_cast<double>(b.width) * static_cast<double>(b.height) - shared;
        return shared / joined;
    }

    /** @brief Holds the two numbers of a fit the test sums box by box. */
    struct HandFit
    {
        double mean_best_iou = 0.0;        /**< Mean of each box's largest overlap. */
        double best_possible_recall = 0.0; /**< Share of boxes at or above the recall IoU. */
    };

    /**
     * @brief Returns the fit of @p anchors on @p boxes, summed box by box in double.
     * @pre Every anchor belongs to a head, so a box is recallable by its overlap alone.
     */
    HandFit measure_by_hand(const std::vector<AnchorSize>& boxes,
                            const std::vector<AnchorSize>& anchors, double recall_iou)
    {
        double sum = 0.0;
        std::size_t recallable = 0;
        for (const AnchorSize& box : boxes)
        {
            double best = 0.0;
            for (const AnchorSize& anchor : anchors)
                best = std::max(best, measure_overlap(box, anchor));
            sum += best;
            if (best >= recall_iou)
                ++recallable;
        }
        const double count = static_cast<double>(boxes.size());
        return HandFit{sum / count, static_cast<double>(recallable) / count};
    }

    /** @brief Expects every number of @p actual within K_SAME_TOLERANCE of @p expected. */
    void expect_same_fit(const AnchorFit& actual, const AnchorFit& expected)
    {
        EXPECT_EQ(actual.boxes, expected.boxes);
        EXPECT_NEAR(actual.mean_best_iou, expected.mean_best_iou, K_SAME_TOLERANCE);
        EXPECT_NEAR(actual.best_possible_recall, expected.best_possible_recall, K_SAME_TOLERANCE);
        EXPECT_NEAR(actual.unowned_share, expected.unowned_share, K_SAME_TOLERANCE);
        EXPECT_NEAR(actual.best_iou_tenth_percentile, expected.best_iou_tenth_percentile,
                    K_SAME_TOLERANCE);
        ASSERT_EQ(actual.anchor_share.size(), expected.anchor_share.size());
        for (std::size_t index = 0; index < actual.anchor_share.size(); ++index)
        {
            EXPECT_NEAR(actual.anchor_share[index], expected.anchor_share[index], K_SAME_TOLERANCE)
                << "anchor " << index;
        }
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
} // namespace

/** @brief Tests that a layout fitted to the ladder is kept on the ladder. */
TEST(Unit_DetectAnchorReview, FittedLayoutIsKeptOnItsOwnBoxes)
{
    const std::vector<AnchorSize> ladder = make_ladder(1.0);
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorLayout current = fit_layout(ladder, strides, {3, 3});

    const Result<AnchorReview> outcome = review_anchors(ladder, current, strides, ReviewOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorReview& review = outcome.value();
    EXPECT_EQ(review.verdict, AnchorVerdict::Keep);
    // Each box lies within 2 % per side of a rung, so its best overlap is far above 0.25.
    EXPECT_NEAR(review.current.best_possible_recall, 1.0, K_SAME_TOLERANCE);
    // The review clusters the same boxes with the same options as fit_layout did.
    EXPECT_NEAR(review.proposed_fit.mean_best_iou, review.current.mean_best_iou,
                K_SAME_TOLERANCE);
}

/** @brief Tests that the shifted ladder gets Propose with both fits as measured directly. */
TEST(Unit_DetectAnchorReview, ShiftedBoxesGetAProposalWithBothFits)
{
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, {3, 3});
    const std::vector<AnchorSize> shifted = make_ladder(K_SHIFT);
    const ReviewOptions options;

    const Result<AnchorReview> outcome = review_anchors(shifted, current, strides, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorReview& review = outcome.value();
    EXPECT_EQ(review.verdict, AnchorVerdict::Propose);

    const Result<AnchorFit> current_fit = measure_anchor_fit(shifted, current, options.fit);
    const Result<AnchorFit> proposed_fit =
        measure_anchor_fit(shifted, review.proposed, options.fit);
    ASSERT_TRUE(current_fit.has_value()) << current_fit.status().message();
    ASSERT_TRUE(proposed_fit.has_value()) << proposed_fit.status().message();
    expect_same_fit(review.current, current_fit.value());
    expect_same_fit(review.proposed_fit, proposed_fit.value());

    const HandFit current_by_hand = measure_by_hand(shifted, current.anchors, 0.25);
    const HandFit proposed_by_hand = measure_by_hand(shifted, review.proposed.anchors, 0.25);
    EXPECT_NEAR(review.current.mean_best_iou, current_by_hand.mean_best_iou, K_FIT_TOLERANCE);
    EXPECT_NEAR(review.current.best_possible_recall, current_by_hand.best_possible_recall,
                K_FIT_TOLERANCE);
    EXPECT_NEAR(review.proposed_fit.mean_best_iou, proposed_by_hand.mean_best_iou,
                K_FIT_TOLERANCE);
    EXPECT_NEAR(review.proposed_fit.best_possible_recall, proposed_by_hand.best_possible_recall,
                K_FIT_TOLERANCE);
    EXPECT_GT(review.proposed_fit.mean_best_iou - review.current.mean_best_iou, options.min_gain);
}

/** @brief Tests that the proposal's anchors sit within 5 % of the shifted rungs. */
TEST(Unit_DetectAnchorReview, ProposalRecoversTheShiftedRungs)
{
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, {3, 3});

    const Result<AnchorReview> outcome =
        review_anchors(make_ladder(K_SHIFT), current, strides, ReviewOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const std::vector<AnchorSize>& anchors = outcome.value().proposed.anchors;
    ASSERT_EQ(anchors.size(), static_cast<std::size_t>(K_RUNGS));
    for (std::int64_t rung = 0; rung < K_RUNGS; ++rung)
    {
        const double scale = K_SHIFT * std::pow(1.8, static_cast<double>(rung));
        const AnchorSize anchor = anchors[static_cast<std::size_t>(rung)];
        EXPECT_NEAR(anchor.width, 0.03 * scale, K_CENTRE_TOLERANCE * 0.03 * scale)
            << "rung " << rung;
        EXPECT_NEAR(anchor.height, 0.05 * scale, K_CENTRE_TOLERANCE * 0.05 * scale)
            << "rung " << rung;
    }
}

/** @brief Tests that the proposal has the input's anchor count and counts per head. */
TEST(Unit_DetectAnchorReview, ProposalKeepsTheCountsPerHead)
{
    const std::vector<std::int64_t> strides = {32, 16};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, {2, 4});

    const Result<AnchorReview> outcome =
        review_anchors(make_ladder(K_SHIFT), current, strides, ReviewOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorLayout& proposed = outcome.value().proposed;
    EXPECT_EQ(proposed.anchors.size(), current.anchors.size());
    ASSERT_EQ(proposed.head_anchors.size(), 2u);
    // Anchors come in ascending area; stride 16 takes the four smallest, stride 32 the rest.
    EXPECT_EQ(proposed.head_anchors[0], (std::vector<std::int64_t>{4, 5}));
    EXPECT_EQ(proposed.head_anchors[1], (std::vector<std::int64_t>{0, 1, 2, 3}));
}

/** @brief Tests that the proposal is the clustering of the boxes assigned to the heads. */
TEST(Unit_DetectAnchorReview, ProposalIsTheClusteringAssignedToTheHeads)
{
    const std::vector<std::int64_t> strides = {16, 32};
    const std::vector<std::int64_t> per_head = {3, 3};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, per_head);
    const std::vector<AnchorSize> shifted = make_ladder(K_SHIFT);
    ReviewOptions options;
    options.clustering.count = 99;

    const Result<AnchorReview> outcome = review_anchors(shifted, current, strides, options);

    // fit_layout clusters to K_RUNGS with the default seed, as the review must for this layout.
    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    expect_same_layout(outcome.value().proposed, fit_layout(shifted, strides, per_head));
}

/** @brief Tests that the reviewed layout holds the same anchors and heads afterwards. */
TEST(Unit_DetectAnchorReview, InputLayoutIsUnchanged)
{
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, {3, 3});
    const AnchorLayout before = current;

    const Result<AnchorReview> outcome =
        review_anchors(make_ladder(K_SHIFT), current, strides, ReviewOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_EQ(outcome.value().verdict, AnchorVerdict::Propose);
    expect_same_layout(current, before);
}

/** @brief Tests each clause of the verdict with the other one switched off. */
TEST(Unit_DetectAnchorReview, RecallAndGainEachDecideTheVerdict)
{
    const std::vector<std::int64_t> strides = {16, 32};
    const AnchorLayout current = fit_layout(make_ladder(1.0), strides, {3, 3});
    const std::vector<AnchorSize> shifted = make_ladder(K_SHIFT);
    ReviewOptions no_gain_counts;
    no_gain_counts.min_gain = 10.0;
    ReviewOptions strict_recall = no_gain_counts;
    strict_recall.fit.recall_iou = 0.6f;

    const Result<AnchorReview> kept = review_anchors(shifted, current, strides, no_gain_counts);
    const Result<AnchorReview> proposed = review_anchors(shifted, current, strides, strict_recall);

    // A shifted box overlaps its best fitted anchor by 0.5625 before jitter, at most 0.59.
    ASSERT_TRUE(kept.has_value()) << kept.status().message();
    ASSERT_TRUE(proposed.has_value()) << proposed.status().message();
    EXPECT_NEAR(kept.value().current.best_possible_recall, 1.0, K_SAME_TOLERANCE);
    EXPECT_EQ(kept.value().verdict, AnchorVerdict::Keep);
    EXPECT_NEAR(proposed.value().current.best_possible_recall,
                measure_by_hand(shifted, current.anchors, 0.6).best_possible_recall,
                K_FIT_TOLERANCE);
    EXPECT_NEAR(proposed.value().current.best_possible_recall, 0.0, K_SAME_TOLERANCE);
    EXPECT_EQ(proposed.value().verdict, AnchorVerdict::Propose);
}

/** @brief Tests that a min_gain or min_recall outside its range is InvalidDescription. */
TEST(Unit_DetectAnchorReview, RefusesOptionsOutsideTheirRange)
{
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}, {0.5f, 0.25f}};
    const AnchorLayout current{{{0.25f, 0.5f}, {0.5f, 0.25f}}, {{0, 1}}};
    const std::vector<std::int64_t> strides = {32};
    std::vector<ReviewOptions> refused(5);
    refused[0].min_gain = -0.01;
    refused[1].min_gain = std::numeric_limits<double>::quiet_NaN();
    refused[2].min_recall = 1.5;
    refused[3].min_recall = -0.1;
    refused[4].fit.recall_iou = 2.0f;

    for (const ReviewOptions& options : refused)
    {
        const Result<AnchorReview> outcome = review_anchors(boxes, current, strides, options);

        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    }
}

/** @brief Tests that a layout with no head is InvalidDescription. */
TEST(Unit_DetectAnchorReview, RefusesALayoutWithoutHeads)
{
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}, {0.5f, 0.25f}};
    const AnchorLayout current{{{0.25f, 0.5f}}, {}};
    const std::vector<std::int64_t> strides = {32};

    const Result<AnchorReview> outcome = review_anchors(boxes, current, strides, ReviewOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

/** @brief Tests that boxes, strides or head counts that disagree are Incompatible. */
TEST(Unit_DetectAnchorReview, RefusesBoxesStridesAndCountsThatDisagree)
{
    const std::vector<AnchorSize> boxes = {{0.125f, 0.25f}, {0.25f, 0.5f}, {0.5f, 0.75f}};
    const std::vector<AnchorSize> no_boxes;
    const std::vector<AnchorSize> one_box(4, AnchorSize{0.25f, 0.5f});
    const AnchorLayout whole{{{0.125f, 0.25f}, {0.25f, 0.5f}, {0.5f, 0.75f}}, {{0}, {1, 2}}};
    const AnchorLayout shared{{{0.125f, 0.25f}, {0.25f, 0.5f}, {0.5f, 0.75f}}, {{0, 1}, {1, 2}}};
    const std::vector<std::int64_t> two_strides = {16, 32};
    const std::vector<std::int64_t> one_stride = {32};
    const ReviewOptions options;

    const Result<AnchorReview> accepted = review_anchors(boxes, whole, two_strides, options);
    const Result<AnchorReview> empty = review_anchors(no_boxes, whole, two_strides, options);
    const Result<AnchorReview> few = review_anchors(one_box, whole, two_strides, options);
    const Result<AnchorReview> short_strides = review_anchors(boxes, whole, one_stride, options);
    const Result<AnchorReview> four_slots = review_anchors(boxes, shared, two_strides, options);

    EXPECT_TRUE(accepted.has_value()) << accepted.status().message();
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.status().code(), StatusCode::Incompatible);
    ASSERT_FALSE(few.has_value());
    EXPECT_EQ(few.status().code(), StatusCode::Incompatible);
    ASSERT_FALSE(short_strides.has_value());
    EXPECT_EQ(short_strides.status().code(), StatusCode::Incompatible);
    ASSERT_FALSE(four_slots.has_value());
    EXPECT_EQ(four_slots.status().code(), StatusCode::Incompatible);
}

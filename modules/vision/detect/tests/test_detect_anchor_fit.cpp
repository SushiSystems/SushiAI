/**************************************************************************/
/* test_detect_anchor_fit.cpp                                             */
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
 * @file test_detect_anchor_fit.cpp
 * @brief Tests the fit report of an anchor layout on hand-built boxes.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorFit;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::FitOptions;
    using SushiAI::Vision::Detect::Anchor::measure_anchor_fit;
    using SushiAI::Vision::Detect::Anchor::shape_iou;

    /** @brief Holds the absolute tolerance of a share or a mean summed in double. */
    constexpr double K_FIT_TOLERANCE = 1e-9;

    /** @brief Returns three anchors whose half-size boxes fit no other anchor above 0.2. */
    AnchorLayout make_separated_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.0625f, 0.5f}, {0.5f, 0.0625f}, {0.5f, 0.5f}};
        layout.head_anchors = {{0, 1, 2}};
        return layout;
    }

    /** @brief Returns one box per anchor of @p layout with each side times @p factor. */
    std::vector<AnchorSize> make_scaled_boxes(const AnchorLayout& layout, float factor)
    {
        std::vector<AnchorSize> boxes;
        for (const AnchorSize& anchor : layout.anchors)
            boxes.push_back(AnchorSize{anchor.width * factor, anchor.height * factor});
        return boxes;
    }

    /** @brief Returns the six anchors of the yolov4-tiny cfg over its 416 input, two masks. */
    AnchorLayout make_v4_tiny_layout()
    {
        const float pixels[6][2] = {{10.0f, 14.0f}, {23.0f, 27.0f},   {37.0f, 58.0f},
                                    {81.0f, 82.0f}, {135.0f, 169.0f}, {344.0f, 319.0f}};
        AnchorLayout layout;
        for (const auto& anchor : pixels)
            layout.anchors.push_back(AnchorSize{anchor[0] / 416.0f, anchor[1] / 416.0f});
        layout.head_anchors = {{3, 4, 5}, {1, 2, 3}};
        return layout;
    }

    /** @brief Returns @p copies boxes at each anchor size of @p layout, anchor by anchor. */
    std::vector<AnchorSize> repeat_anchor_sizes(const AnchorLayout& layout, std::size_t copies)
    {
        std::vector<AnchorSize> boxes;
        for (const AnchorSize& anchor : layout.anchors)
            boxes.insert(boxes.end(), copies, anchor);
        return boxes;
    }
} // namespace

/** @brief Tests that one box at each anchor's size measures 1 in every number. */
TEST(Unit_DetectAnchorFit, OneBoxPerAnchorGivesOneEverywhere)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 1.0f);

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 3);
    EXPECT_NEAR(fit.mean_best_iou, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_iou_tenth_percentile, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.unowned_share, 0.0, K_FIT_TOLERANCE);
    ASSERT_EQ(fit.anchor_share.size(), 3u);
    for (const double share : fit.anchor_share)
        EXPECT_NEAR(share, 1.0 / 3.0, K_FIT_TOLERANCE);
}

/** @brief Tests that boxes at half an anchor's sides measure 0.25 and are recalled at 0.25. */
TEST(Unit_DetectAnchorFit, HalfSizedBoxesGiveAQuarterAndAreRecalledAtAQuarter)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 0.5f);
    FitOptions at_quarter;
    at_quarter.recall_iou = 0.25f;
    FitOptions above_quarter;
    above_quarter.recall_iou = 0.3f;

    const Result<AnchorFit> reached = measure_anchor_fit(boxes, layout, at_quarter);
    const Result<AnchorFit> missed = measure_anchor_fit(boxes, layout, above_quarter);

    // Every side is a power of two, so each area ratio is 0.25 without rounding.
    ASSERT_TRUE(reached.has_value()) << reached.status().message();
    ASSERT_TRUE(missed.has_value()) << missed.status().message();
    EXPECT_NEAR(reached.value().mean_best_iou, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(reached.value().best_iou_tenth_percentile, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(reached.value().best_possible_recall, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(missed.value().mean_best_iou, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(missed.value().best_possible_recall, 0.0, K_FIT_TOLERANCE);
}

/** @brief Tests that the yolov4-tiny masks leave the boxes of anchor 0 without a head. */
TEST(Unit_DetectAnchorFit, V4TinyMasksLeaveASixthOfTheBoxesUnowned)
{
    const AnchorLayout layout = make_v4_tiny_layout();
    const std::vector<AnchorSize> boxes = repeat_anchor_sizes(layout, 100);

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 600);
    EXPECT_NEAR(fit.unowned_share, 1.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, 5.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.mean_best_iou, 1.0, K_FIT_TOLERANCE);
    ASSERT_EQ(fit.anchor_share.size(), 6u);
    for (const double share : fit.anchor_share)
        EXPECT_NEAR(share, 1.0 / 6.0, K_FIT_TOLERANCE);
}

/** @brief Tests that masks covering every anchor bring the unowned share of that case to 0. */
TEST(Unit_DetectAnchorFit, MasksOverEveryAnchorLeaveNoBoxUnowned)
{
    AnchorLayout layout = make_v4_tiny_layout();
    const std::vector<AnchorSize> boxes = repeat_anchor_sizes(layout, 100);
    layout.head_anchors = {{3, 4, 5}, {0, 1, 2}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().unowned_share, 0.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().best_possible_recall, 1.0, K_FIT_TOLERANCE);
}

/** @brief Tests the four numbers and the shares against sums the test makes box by box. */
TEST(Unit_DetectAnchorFit, FourNumbersEqualTheBoxByBoxSums)
{
    AnchorLayout layout;
    layout.anchors = {{0.1f, 0.1f}, {0.2f, 0.3f}, {0.5f, 0.4f}, {0.8f, 0.9f}};
    layout.head_anchors = {{2, 3}, {1}};
    const std::vector<AnchorSize> boxes = {
        {0.09f, 0.12f}, {0.05f, 0.04f}, {0.21f, 0.28f}, {0.3f, 0.3f},  {0.45f, 0.5f},
        {0.6f, 0.2f},   {0.9f, 0.9f},   {0.7f, 1.0f},   {0.02f, 0.6f}, {0.15f, 0.2f},
        {0.4f, 0.4f},   {0.25f, 0.35f}};
    FitOptions options;
    options.recall_iou = 0.5f;

    std::vector<double> expected_best;
    std::vector<double> expected_share(layout.anchors.size(), 0.0);
    double expected_sum = 0.0;
    double expected_recalled = 0.0;
    double expected_unowned = 0.0;
    for (const AnchorSize& box : boxes)
    {
        std::size_t best_index = 0;
        float best_iou = -1.0f;
        for (std::size_t index = 0; index < layout.anchors.size(); ++index)
        {
            const float iou = shape_iou(box, layout.anchors[index]);
            if (iou > best_iou)
            {
                best_iou = iou;
                best_index = index;
            }
        }
        expected_best.push_back(static_cast<double>(best_iou));
        expected_sum += static_cast<double>(best_iou);
        expected_share[best_index] += 1.0 / 12.0;
        // Anchor 0 is the one no head lists.
        if (best_index == 0)
            expected_unowned += 1.0;
        else if (best_iou >= 0.5f)
            expected_recalled += 1.0;
    }
    std::sort(expected_best.begin(), expected_best.end());

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 12);
    EXPECT_NEAR(fit.mean_best_iou, expected_sum / 12.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, expected_recalled / 12.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.unowned_share, expected_unowned / 12.0, K_FIT_TOLERANCE);
    // Twelve boxes put the tenth percentile at rank ceil(1.2) = 2, the second smallest.
    EXPECT_NEAR(fit.best_iou_tenth_percentile, expected_best[1], K_FIT_TOLERANCE);
    EXPECT_GT(expected_unowned, 0.0);
    EXPECT_GT(expected_recalled, 0.0);
    EXPECT_LT(expected_recalled + expected_unowned, 12.0);
    ASSERT_EQ(fit.anchor_share.size(), expected_share.size());
    for (std::size_t index = 0; index < expected_share.size(); ++index)
        EXPECT_NEAR(fit.anchor_share[index], expected_share[index], K_FIT_TOLERANCE);
}

/** @brief Tests that the tenth percentile of ten boxes is the smallest best IoU. */
TEST(Unit_DetectAnchorFit, TenthPercentileOfTenBoxesIsTheSmallest)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    std::vector<AnchorSize> boxes(9, AnchorSize{0.5f, 0.5f});
    boxes.push_back(AnchorSize{0.25f, 0.25f});

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().best_iou_tenth_percentile, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().mean_best_iou, 0.925, K_FIT_TOLERANCE);
}

/** @brief Tests that of two equal anchors the lower index takes the box. */
TEST(Unit_DetectAnchorFit, EqualAnchorsGiveTheBoxToTheLowerIndex)
{
    AnchorLayout layout;
    layout.anchors = {{0.25f, 0.5f}, {0.25f, 0.5f}};
    layout.head_anchors = {{1}};
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    ASSERT_EQ(outcome.value().anchor_share.size(), 2u);
    EXPECT_NEAR(outcome.value().anchor_share[0], 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().anchor_share[1], 0.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().unowned_share, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().best_possible_recall, 0.0, K_FIT_TOLERANCE);
}

/** @brief Tests that an empty box list is Incompatible. */
TEST(Unit_DetectAnchorFit, RefusesAnEmptyBoxList)
{
    const std::vector<AnchorSize> boxes;

    const Result<AnchorFit> outcome =
        measure_anchor_fit(boxes, make_separated_layout(), FitOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that a layout whose head names a missing anchor is InvalidDescription. */
TEST(Unit_DetectAnchorFit, RefusesALayoutThatDoesNotValidate)
{
    AnchorLayout layout = make_separated_layout();
    layout.head_anchors[0][2] = 3;
    const std::vector<AnchorSize> boxes = {{0.5f, 0.5f}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a recall_iou below 0, above 1 or not a number is InvalidDescription. */
TEST(Unit_DetectAnchorFit, RefusesARecallIoUOutsideZeroToOne)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = {{0.5f, 0.5f}};
    const float refused[3] = {-0.1f, 1.5f, std::numeric_limits<float>::quiet_NaN()};

    for (const float recall_iou : refused)
    {
        FitOptions options;
        options.recall_iou = recall_iou;

        const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, options);

        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    }
}


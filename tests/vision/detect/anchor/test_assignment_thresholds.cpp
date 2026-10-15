/**************************************************************************/
/* test_assignment_thresholds.cpp                                         */
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
 * @file test_assignment_thresholds.cpp
 * @brief Tests the match and ignore thresholds derived from hand-written anchor fits.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::AnchorFit;
    using SushiAI::Vision::Detect::Anchor::AssignmentThresholds;
    using SushiAI::Vision::Detect::Anchor::derive_assignment_thresholds;

    /** @brief Holds the absolute tolerance of a threshold narrowed from double to float. */
    constexpr float K_THRESHOLD_TOLERANCE = 1e-6f;

    /** @brief Holds one fit and the thresholds worked out for it by hand. */
    struct ThresholdCase
    {
        double tenth_percentile = 0.0; /**< Tenth percentile of the best shape_iou. */
        double mean_best_iou = 0.0;    /**< Mean of the best shape_iou. */
        float expected_match = 0.0f;   /**< tenth_percentile clamped to [0.2, 1]. */
        float expected_ignore = 0.0f;  /**< mean_best_iou clamped to [0.5, 0.7]. */
    };

    /** @brief Returns a fit that carries the two numbers the thresholds are derived from. */
    AnchorFit make_fit(double tenth_percentile, double mean_best_iou)
    {
        AnchorFit fit;
        fit.best_iou_tenth_percentile = tenth_percentile;
        fit.mean_best_iou = mean_best_iou;
        return fit;
    }

    /** @brief Expects the thresholds of each case's fit to equal its hand-written pair. */
    template <std::size_t Count>
    void expect_cases(const ThresholdCase (&cases)[Count])
    {
        for (const ThresholdCase& entry : cases)
        {
            const AssignmentThresholds thresholds = derive_assignment_thresholds(
                make_fit(entry.tenth_percentile, entry.mean_best_iou));

            EXPECT_NEAR(thresholds.match_threshold, entry.expected_match, K_THRESHOLD_TOLERANCE)
                << "tenth percentile " << entry.tenth_percentile;
            EXPECT_NEAR(thresholds.ignore_threshold, entry.expected_ignore,
                        K_THRESHOLD_TOLERANCE)
                << "mean best IoU " << entry.mean_best_iou;
        }
    }
} // namespace

/** @brief Tests that without a fit the thresholds are 1 and 0.7. */
TEST(Small_DetectAnchorThresholds, DefaultsAreOneAndSevenTenths)
{
    const AssignmentThresholds thresholds;

    EXPECT_EQ(thresholds.match_threshold, 1.0f);
    EXPECT_EQ(thresholds.ignore_threshold, 0.7f);
}

/** @brief Tests that a tenth percentile of 0.35 and a mean of 0.62 pass through unclamped. */
TEST(Small_DetectAnchorThresholds, FitInsideBothRangesGivesItsOwnNumbers)
{
    const AssignmentThresholds thresholds = derive_assignment_thresholds(make_fit(0.35, 0.62));

    EXPECT_NEAR(thresholds.match_threshold, 0.35f, K_THRESHOLD_TOLERANCE);
    EXPECT_NEAR(thresholds.ignore_threshold, 0.62f, K_THRESHOLD_TOLERANCE);
}

/** @brief Tests the formula on four fits, each number inside its range. */
TEST(Small_DetectAnchorThresholds, FourFitsFollowTheFormula)
{
    const ThresholdCase cases[4] = {{0.35, 0.62, 0.35f, 0.62f},
                                    {0.21, 0.51, 0.21f, 0.51f},
                                    {0.5, 0.69, 0.5f, 0.69f},
                                    {0.25, 0.55, 0.25f, 0.55f}};

    expect_cases(cases);
}

/** @brief Tests that the match threshold stops at 0.2 below and at 1 above. */
TEST(Small_DetectAnchorThresholds, MatchThresholdIsClampedAtBothEnds)
{
    const ThresholdCase cases[5] = {{0.05, 0.6, 0.2f, 0.6f},
                                    {0.0, 0.6, 0.2f, 0.6f},
                                    {0.2, 0.6, 0.2f, 0.6f},
                                    {1.0, 0.6, 1.0f, 0.6f},
                                    {1.3, 0.6, 1.0f, 0.6f}};

    expect_cases(cases);
}

/** @brief Tests that the ignore threshold stops at 0.5 below and at 0.7 above. */
TEST(Small_DetectAnchorThresholds, IgnoreThresholdIsClampedAtBothEnds)
{
    const ThresholdCase cases[5] = {{0.4, 0.3, 0.4f, 0.5f},
                                    {0.4, 0.0, 0.4f, 0.5f},
                                    {0.4, 0.5, 0.4f, 0.5f},
                                    {0.4, 0.7, 0.4f, 0.7f},
                                    {0.4, 0.95, 0.4f, 0.7f}};

    expect_cases(cases);
}

/** @brief Tests that a fit of 1 everywhere gives a match of 1 and an ignore of 0.7. */
TEST(Small_DetectAnchorThresholds, PerfectFitGivesOneAndSevenTenths)
{
    const AssignmentThresholds thresholds = derive_assignment_thresholds(make_fit(1.0, 1.0));

    EXPECT_NEAR(thresholds.match_threshold, 1.0f, K_THRESHOLD_TOLERANCE);
    EXPECT_NEAR(thresholds.ignore_threshold, 0.7f, K_THRESHOLD_TOLERANCE);
}

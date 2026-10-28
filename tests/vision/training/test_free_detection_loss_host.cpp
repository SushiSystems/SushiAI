/**************************************************************************/
/* test_free_detection_loss_host.cpp                                      */
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
 * @file test_free_detection_loss_host.cpp
 * @brief Tests the host evaluation of the anchor-free detection loss against closed forms.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>

namespace
{
    using SushiAI::Vision::Training::evaluate_free_detection_loss;
    using SushiAI::Vision::Training::FreeLevelLossArrays;
    using SushiAI::Vision::Training::FreeLossGains;
    using SushiAI::Vision::Training::FreeLossTerms;

    /** @brief Bounds a term the closed forms below give exactly, up to rounding. */
    constexpr double K_EXACT = 1e-12;

    /** @brief Bounds a box term, whose CIoU guards add epsilons of 1e-7. */
    constexpr double K_BOX_TOLERANCE = 1e-6;

    /**
     * @brief Holds the values of one level of one row, so a case changes only what it tests.
     *
     * The row is an anchor point at (1, 1) whose four distances of 1 decode to the box
     * (0, 0, 2, 2), which is also the target and the safe box, so the box term is near 0.
     */
    struct LevelRow
    {
        std::int64_t reg_max = 1;                        /**< Bins per side. */
        std::int64_t classes = 1;                        /**< Class logits per row. */
        std::vector<double> boxes = {1.0, 1.0, 1.0, 1.0};  /**< Distances, or bin logits. */
        std::vector<double> scores = {0.0};               /**< Class logits. */
        std::vector<double> target_scores = {0.0};        /**< Soft class targets. */
        std::vector<double> box_mask = {1.0, 1.0, 1.0, 1.0}; /**< 1 where the box is decoded. */
        std::vector<double> normalized_weight = {0.5};    /**< Weight of the row's box term. */
        std::vector<double> target_boxes = {0.0, 0.0, 2.0, 2.0}; /**< Target corners. */
        std::vector<double> target_distances = {0.0, 0.0, 0.0, 0.0}; /**< Target sides. */
        std::vector<double> l1_scale = {0.5, 0.5, 0.5, 0.5};         /**< Factor of each side. */
        std::vector<double> safe_boxes = {0.0, 0.0, 2.0, 2.0};       /**< Box of a masked row. */
        std::vector<double> distribution_weights;                    /**< Bin weights. */
        std::vector<double> anchor_points = {1.0, 1.0};              /**< Cell centre. */

        /** @brief Returns the arrays of this level, viewing the vectors of this object. */
        FreeLevelLossArrays arrays() const
        {
            FreeLevelLossArrays level;
            level.boxes = boxes;
            level.scores = scores;
            level.target_scores = target_scores;
            level.box_mask = box_mask;
            level.normalized_weight = normalized_weight;
            level.target_boxes = target_boxes;
            level.target_distances = target_distances;
            level.l1_scale = l1_scale;
            level.safe_boxes = safe_boxes;
            level.distribution_weights = distribution_weights;
            level.anchor_points = anchor_points;
            return level;
        }
    };

    /** @brief Returns a row of two bins per side, all logits 0, with the weights 1/4 and 3/4. */
    LevelRow two_bin_row()
    {
        LevelRow row;
        row.reg_max = 2;
        row.boxes = std::vector<double>(8, 0.0);
        row.distribution_weights = {0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75};
        return row;
    }

    /** @brief Evaluates @p levels of @p row's shape at the batch and the score sum given. */
    SushiAI::Result<FreeLossTerms> evaluate(const std::vector<LevelRow>& levels,
                                            const FreeLossGains& gains, std::int64_t batch = 1,
                                            double target_scores_sum = 1.0)
    {
        std::vector<FreeLevelLossArrays> arrays;
        for (const LevelRow& level : levels)
            arrays.push_back(level.arrays());
        return evaluate_free_detection_loss(
            SushiRuntime::span<const FreeLevelLossArrays>(arrays.data(), arrays.size()),
            levels.front().reg_max, levels.front().classes, batch, target_scores_sum, gains);
    }

    /** @brief Returns the gains with every term on, each a different factor. */
    FreeLossGains distinct_gains()
    {
        FreeLossGains gains;
        gains.box_gain = 2.0f;
        gains.class_gain = 3.0f;
        gains.distribution_gain = 4.0f;
        gains.l1_gain = 5.0f;
        return gains;
    }
} // namespace

/** @brief Tests that the class term is the logistic cross-entropy over the score sum. */
TEST(Small_FreeDetectionLossHost, TheClassTermIsTheCrossEntropyOverTheScoreSum)
{
    LevelRow row;
    row.classes = 2;
    row.scores = {0.0, 2.0};
    row.target_scores = {0.0, 1.0};

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({row}, FreeLossGains{}, 1, 2.0));

    // (log 2 + (log(1 + e^2) - 2)) / 2, computed outside the library.
    EXPECT_NEAR(terms.classes, 0.41003759580145904, K_EXACT);
}

/** @brief Tests that a decoded box equal to its target costs nothing. */
TEST(Small_FreeDetectionLossHost, ABoxEqualToItsTargetCostsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms,
                            evaluate({LevelRow()}, FreeLossGains{}));

    EXPECT_NEAR(terms.box, 0.0, K_BOX_TOLERANCE);
}

/** @brief Tests the box term (1 - CIoU) * weight on a box inside its target. */
TEST(Small_FreeDetectionLossHost, TheBoxTermIsOneMinusCIoUTimesTheWeight)
{
    LevelRow row;
    row.boxes = {1.0, 1.0, 0.0, 0.0};

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({row}, FreeLossGains{}));

    // (0, 0, 1, 1) in (0, 0, 2, 2): IoU 1/4, centre term 1/16, equal aspects; (1 - 3/16) * 0.5.
    EXPECT_NEAR(terms.box, 0.40625, K_BOX_TOLERANCE);
}

/** @brief Tests that a row whose mask is 0 is scored with its safe box, not its output. */
TEST(Small_FreeDetectionLossHost, AMaskedRowTakesTheSafeBoxInPlaceOfItsOutput)
{
    LevelRow row;
    row.boxes = {9.0, 9.0, 9.0, 9.0};
    row.box_mask = {0.0, 0.0, 0.0, 0.0};

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({row}, FreeLossGains{}));

    EXPECT_NEAR(terms.box, 0.0, K_BOX_TOLERANCE);
}

/** @brief Tests that the distribution term is the cross-entropy against the bin weights. */
TEST(Small_FreeDetectionLossHost, TheDistributionTermIsTheCrossEntropyAgainstTheBinWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms,
                            evaluate({two_bin_row()}, FreeLossGains{}));

    // Equal logits give each bin -log 2 and a side's weights sum to 1, so each side adds log 2.
    EXPECT_NEAR(terms.distribution, 2.772588722239781, K_EXACT);
}

/** @brief Tests that one bin per side leaves the distribution term at 0. */
TEST(Small_FreeDetectionLossHost, OneBinPerSideHasNoDistributionTerm)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms,
                            evaluate({LevelRow()}, FreeLossGains{}));

    EXPECT_EQ(terms.distribution, 0.0);
}

/** @brief Tests that the L1 term weighs the error of each side by its scale. */
TEST(Small_FreeDetectionLossHost, TheL1TermWeighsEachSideErrorByItsScale)
{
    LevelRow row;
    row.boxes = {1.0, 1.0, 0.0, 0.0};
    FreeLossGains gains;
    gains.l1_gain = 1.0f;

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({row}, gains));

    // |1 - 0| * 0.5 on two sides and 0 on the others.
    EXPECT_NEAR(terms.l1, 1.0, K_EXACT);
}

/** @brief Tests that a zero L1 gain leaves the term at 0 and the L1 arrays unread. */
TEST(Small_FreeDetectionLossHost, ANoL1GainNeedsNoL1Arrays)
{
    LevelRow row;
    row.target_distances.clear();
    row.l1_scale.clear();

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({row}, FreeLossGains{}));

    EXPECT_EQ(terms.l1, 0.0);
}

/** @brief Tests that the total is the gained terms times the batch. */
TEST(Small_FreeDetectionLossHost, TheTotalIsTheGainedTermsTimesTheBatch)
{
    const FreeLossGains gains = distinct_gains();

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms terms, evaluate({two_bin_row()}, gains, 3));

    const double gained = 2.0 * terms.box + 3.0 * terms.classes + 4.0 * terms.distribution +
                          5.0 * terms.l1;
    EXPECT_GT(terms.l1, 0.0);
    EXPECT_NEAR(terms.total, 3.0 * gained, K_EXACT * terms.total);
}

/** @brief Tests that two equal levels double each term, the score sum held fixed. */
TEST(Small_FreeDetectionLossHost, LevelsAddTheirTerms)
{
    const FreeLossGains gains = distinct_gains();
    LevelRow row = two_bin_row();
    row.boxes = {0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5};

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms one, evaluate({row}, gains));
    SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms two, evaluate({row, row}, gains));

    EXPECT_NEAR(two.box, 2.0 * one.box, K_EXACT);
    EXPECT_NEAR(two.classes, 2.0 * one.classes, K_EXACT);
    EXPECT_NEAR(two.distribution, 2.0 * one.distribution, K_EXACT);
    EXPECT_NEAR(two.l1, 2.0 * one.l1, K_EXACT);
    EXPECT_NEAR(two.total, 2.0 * one.total, K_EXACT * two.total);
}

/** @brief Tests that an array of another size than its rows need is refused naming it. */
TEST(Small_FreeDetectionLossHost, AnArrayOfAnotherSizeIsRefusedNamingIt)
{
    LevelRow scores;
    scores.classes = 2;
    scores.scores = {0.0, 0.0, 0.0};
    scores.target_scores = {0.0, 0.0};
    LevelRow mask;
    mask.box_mask = {1.0, 1.0};

    SA_EXPECT_FAILURE(evaluate({scores}, FreeLossGains{}),
                      SushiAI::StatusCode::InvalidDescription, "scores", "hold 3 values");
    SA_EXPECT_FAILURE(evaluate({mask}, FreeLossGains{}),
                      SushiAI::StatusCode::InvalidDescription, "box mask", "hold 2 values");
}

/** @brief Tests that a setting below one, or a score sum that is not above 0, is refused. */
TEST(Small_FreeDetectionLossHost, ASettingThatIsNotPositiveIsRefused)
{
    LevelRow regression;
    regression.reg_max = 0;
    LevelRow classes;
    classes.classes = 0;

    SA_EXPECT_FAILURE(evaluate({regression}, FreeLossGains{}),
                      SushiAI::StatusCode::InvalidDescription, "reg_max 0", "positive");
    SA_EXPECT_FAILURE(evaluate({classes}, FreeLossGains{}),
                      SushiAI::StatusCode::InvalidDescription, "0 classes", "positive");
    SA_EXPECT_FAILURE(evaluate({LevelRow()}, FreeLossGains{}, 0),
                      SushiAI::StatusCode::InvalidDescription, "batch of 0", "positive");
    SA_EXPECT_FAILURE(evaluate({LevelRow()}, FreeLossGains{}, 1, 0.0),
                      SushiAI::StatusCode::InvalidDescription, "score sum", "positive");
}

/** @brief Tests that a run of no level has every term 0. */
TEST(Small_FreeDetectionLossHost, NoLevelGivesNoLoss)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const FreeLossTerms terms,
        evaluate_free_detection_loss(SushiRuntime::span<const FreeLevelLossArrays>(), 1, 1, 1,
                                     1.0, FreeLossGains{}));

    EXPECT_EQ(terms.total, 0.0);
    EXPECT_EQ(terms.box, 0.0);
    EXPECT_EQ(terms.classes, 0.0);
    EXPECT_EQ(terms.distribution, 0.0);
    EXPECT_EQ(terms.l1, 0.0);
}

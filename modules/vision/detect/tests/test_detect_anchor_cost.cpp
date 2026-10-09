/**************************************************************************/
/* test_detect_anchor_cost.cpp                                            */
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
 * @file test_detect_anchor_cost.cpp
 * @brief Tests the reported cost of an anchor head's training delta.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::BoxLoss;
    using SushiAI::Vision::Detect::Anchor::HeadCost;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::measure_head_cost;

    /** @brief Holds the tolerance of the cost sums; they are exact in double for these inputs. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Holds the classes of the head: two, so a slot has 5 + 2 = 7 entries. */
    constexpr std::int64_t K_CLASSES = 2;

    /**
     * @brief Returns the delta of two images on a 1 by 1 grid with two slots of 7 entries.
     *
     * Rows are image 0 slot 0, image 0 slot 1, image 1 slot 0, image 1 slot 1; an entry is
     * x, y, w, h, objectness, class 0, class 1. The squared box entries sum to
     * 1 + 4 + 0.25 = 5.25 and the squared non-box entries to 9 + 16 + 1 = 26.
     */
    std::vector<float> known_delta()
    {
        return {
            1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f,
            0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };
    }

    /** @brief Returns the shape of the known delta: two images on a 1 by 1 grid. */
    HeadShape known_shape()
    {
        HeadShape shape;
        shape.batch = 2;
        shape.height = 1;
        shape.width = 1;
        return shape;
    }

    /** @brief Returns the head options of the known delta. */
    HeadOptions known_options()
    {
        HeadOptions options;
        options.num_classes = K_CLASSES;
        return options;
    }

    /** @brief Returns the known delta with @p assignments whose IoU sums to @p iou_sum. */
    HeadDelta known_head(std::int64_t assignments, double iou_sum)
    {
        HeadDelta delta;
        delta.delta = known_delta();
        delta.assignments = assignments;
        delta.iou_sum = iou_sum;
        return delta;
    }

} // namespace

/**
 * @brief Tests the four numbers of a squared-error head with an objectness weight of 1.
 *
 * Total (5.25 + 26) / 2 = 15.625, non-box 26 / 2 = 13, box 15.625 - 13 = 2.625, and the stored
 * cost is the undivided 5.25 + 26 = 31.25.
 */
TEST(Unit_DetectAnchorCost, SquaredErrorSplitsBoxFromNonBoxAndDividesByBatch)
{
    HeadTrainingOptions training;
    training.box_loss = BoxLoss::SquaredError;

    const HeadCost cost =
        measure_head_cost(known_head(2, 1.0), known_shape(), known_options(), training);

    EXPECT_NEAR(cost.total, 15.625, K_TOLERANCE);
    EXPECT_NEAR(cost.non_box, 13.0, K_TOLERANCE);
    EXPECT_NEAR(cost.box, 2.625, K_TOLERANCE);
    EXPECT_NEAR(cost.stored, 31.25, K_TOLERANCE);
}

/**
 * @brief Tests that the objectness weight multiplies the non-box share and box is the rest.
 *
 * With a weight of 2 the non-box share is 2 * 26 / 2 = 26 and box is 15.625 - 26 = -10.375;
 * the total and the stored cost of a squared-error head do not read the weight.
 */
TEST(Unit_DetectAnchorCost, ObjectWeightScalesTheNonBoxShare)
{
    HeadTrainingOptions training;
    training.box_loss = BoxLoss::SquaredError;
    training.object_weight = 2.0f;

    const HeadCost cost =
        measure_head_cost(known_head(2, 1.0), known_shape(), known_options(), training);

    EXPECT_NEAR(cost.total, 15.625, K_TOLERANCE);
    EXPECT_NEAR(cost.non_box, 26.0, K_TOLERANCE);
    EXPECT_NEAR(cost.box, -10.375, K_TOLERANCE);
    EXPECT_NEAR(cost.stored, 31.25, K_TOLERANCE);
}

/**
 * @brief Tests the stored cost of a CompleteIou head: the IoU term plus the weighted non-box sum.
 *
 * Two assignments with an IoU sum of 1 have a mean 1 - IoU of 0.5, so the IoU term is
 * 0.75 * 0.5 = 0.375; the non-box term is 2 * 26 = 52; the stored cost is 52.375, undivided.
 */
TEST(Unit_DetectAnchorCost, CompleteIouStoredAddsTheIouTermAndWeightedNonBoxSum)
{
    HeadTrainingOptions training;
    training.box_loss = BoxLoss::CompleteIou;
    training.box_weight = 0.75f;
    training.object_weight = 2.0f;

    const HeadCost cost =
        measure_head_cost(known_head(2, 1.0), known_shape(), known_options(), training);

    EXPECT_NEAR(cost.stored, 52.375, K_TOLERANCE);
    EXPECT_NEAR(cost.total, 15.625, K_TOLERANCE);
    EXPECT_NEAR(cost.non_box, 26.0, K_TOLERANCE);
    EXPECT_NEAR(cost.box, -10.375, K_TOLERANCE);
}

/** @brief Tests that a CompleteIou head without an assignment stores the non-box sum of 26. */
TEST(Unit_DetectAnchorCost, CompleteIouWithoutAssignmentsOmitsTheIouTerm)
{
    HeadTrainingOptions training;
    training.box_loss = BoxLoss::CompleteIou;
    training.box_weight = 0.75f;
    training.object_weight = 1.0f;

    const HeadCost cost =
        measure_head_cost(known_head(0, 0.0), known_shape(), known_options(), training);

    EXPECT_NEAR(cost.stored, 26.0, K_TOLERANCE);
}

/** @brief Tests that a delta of zeros reports zero in all four numbers. */
TEST(Unit_DetectAnchorCost, ZeroDeltaReportsZeroCost)
{
    HeadDelta delta;
    delta.delta.assign(28, 0.0f);

    HeadTrainingOptions training;
    const HeadCost cost = measure_head_cost(delta, known_shape(), known_options(), training);

    EXPECT_NEAR(cost.total, 0.0, K_TOLERANCE);
    EXPECT_NEAR(cost.box, 0.0, K_TOLERANCE);
    EXPECT_NEAR(cost.non_box, 0.0, K_TOLERANCE);
    EXPECT_NEAR(cost.stored, 0.0, K_TOLERANCE);
}

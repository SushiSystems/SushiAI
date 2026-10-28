/**************************************************************************/
/* test_suppression_criterion.cpp                                         */
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
 * @file test_suppression_criterion.cpp
 * @brief Tests the suppression measures and the criterion-aware suppression by hand cases.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace Detect = SushiAI::Vision::Detect;

namespace
{
    /** @brief Holds the tolerance of every measure, absolute, in float32 units. */
    constexpr float K_MEASURE_TOLERANCE = 1e-6f;

    /** @brief Holds the first box of the centre-distance cases: a 10 by 10 square at the origin. */
    const std::vector<float> K_SQUARE{0.0f, 0.0f, 10.0f, 10.0f};

    /** @brief Holds a box with the same area that shares 40 of 100 units with K_SQUARE. */
    const std::vector<float> K_SHIFTED_DOWN{0.0f, 6.0f, 10.0f, 16.0f};

    /** @brief Holds a box with the same IoU with K_SQUARE as K_SHIFTED_DOWN, nearer in centre. */
    const std::vector<float> K_SHIFTED_DIAGONAL{2.0f, 5.0f, 12.0f, 15.0f};
} // namespace

/** @brief Tests that the IoU of two overlapping boxes is the ratio of their areas. */
TEST(Small_DetectSuppressionCriterion, IoUOfOverlappingBoxesIsTheRatioOfAreas)
{
    // Intersection 9 * 10 = 90; union 100 + 100 - 90 = 110.
    const std::vector<float> shifted{1.0f, 0.0f, 11.0f, 10.0f};
    const float measure =
        Detect::measure_suppression(K_SQUARE, shifted, Detect::SuppressionCriterion::IoU, 0.6f);
    EXPECT_NEAR(measure, 90.0 / 110.0, K_MEASURE_TOLERANCE);
}

/** @brief Tests that disjoint and edge-touching boxes have an IoU of zero. */
TEST(Small_DetectSuppressionCriterion, IoUIsZeroForDisjointAndTouchingBoxes)
{
    const std::vector<float> far{20.0f, 0.0f, 30.0f, 10.0f};
    const std::vector<float> touching{10.0f, 0.0f, 20.0f, 10.0f};
    EXPECT_EQ(Detect::measure_suppression(K_SQUARE, far, Detect::SuppressionCriterion::IoU, 0.6f),
              0.0f);
    EXPECT_EQ(
        Detect::measure_suppression(K_SQUARE, touching, Detect::SuppressionCriterion::IoU, 0.6f),
        0.0f);
}

/** @brief Tests that two empty boxes have an IoU of zero, not NaN. */
TEST(Small_DetectSuppressionCriterion, IoUOfTwoEmptyBoxesIsZeroNotNaN)
{
    const std::vector<float> point{0.0f, 0.0f, 0.0f, 0.0f};
    const float measure =
        Detect::measure_suppression(point, point, Detect::SuppressionCriterion::IoU, 0.6f);
    EXPECT_FALSE(std::isnan(measure));
    EXPECT_EQ(measure, 0.0f);
}

/** @brief Tests that DistanceIoU subtracts the normalised squared centre distance. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUSubtractsTheNormalisedCentreDistance)
{
    // IoU 40 / 160 = 0.25. Centres (5, 5) and (5, 11): d = 36. Enclosing box 10 by 16: c = 356.
    const float measure = Detect::measure_suppression(
        K_SQUARE, K_SHIFTED_DOWN, Detect::SuppressionCriterion::DistanceIoU, 0.6f);
    EXPECT_NEAR(measure, 0.25 - std::pow(36.0 / 356.0, 0.6), K_MEASURE_TOLERANCE);
}

/** @brief Tests that DistanceIoU of disjoint boxes is negative, the distance term alone. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUOfDisjointBoxesIsNegative)
{
    // Centres (5, 5) and (25, 5): d = 400. Enclosing box 30 by 10: c = 1000.
    const std::vector<float> far{20.0f, 0.0f, 30.0f, 10.0f};
    const float measure =
        Detect::measure_suppression(K_SQUARE, far, Detect::SuppressionCriterion::DistanceIoU, 0.6f);
    EXPECT_NEAR(measure, -std::pow(0.4, 0.6), K_MEASURE_TOLERANCE);
}

/** @brief Tests that an exponent of zero makes the distance term exactly one. */
TEST(Small_DetectSuppressionCriterion, ZeroExponentSubtractsOne)
{
    const float measure = Detect::measure_suppression(
        K_SQUARE, K_SHIFTED_DOWN, Detect::SuppressionCriterion::DistanceIoU, 0.0f);
    EXPECT_NEAR(measure, 0.25 - 1.0, K_MEASURE_TOLERANCE);
}

/** @brief Tests that identical boxes have a DistanceIoU of one, with no centre distance. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUOfIdenticalBoxesIsOne)
{
    const float measure =
        Detect::measure_suppression(K_SQUARE, K_SQUARE, Detect::SuppressionCriterion::DistanceIoU,
                                    0.6f);
    EXPECT_NEAR(measure, 1.0, K_MEASURE_TOLERANCE);
}

/** @brief Tests that two empty boxes at one point fall back to their IoU, which is zero. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUOfCoincidentPointsIsTheIoU)
{
    const std::vector<float> point{0.0f, 0.0f, 0.0f, 0.0f};
    const float measure =
        Detect::measure_suppression(point, point, Detect::SuppressionCriterion::DistanceIoU, 0.6f);
    EXPECT_FALSE(std::isnan(measure));
    EXPECT_EQ(measure, 0.0f);
}

/** @brief Tests that the DistanceIoU of two boxes does not depend on their order. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUIsSymmetric)
{
    const float forward = Detect::measure_suppression(
        K_SQUARE, K_SHIFTED_DIAGONAL, Detect::SuppressionCriterion::DistanceIoU, 0.6f);
    const float backward = Detect::measure_suppression(
        K_SHIFTED_DIAGONAL, K_SQUARE, Detect::SuppressionCriterion::DistanceIoU, 0.6f);
    EXPECT_NEAR(forward, backward, K_MEASURE_TOLERANCE);
}

/** @brief Tests that the closer of two boxes with equal IoU is suppressed under DistanceIoU. */
TEST(Small_DetectSuppressionCriterion, DistanceIoUKeepsTheFartherOfTwoEqualIoUBoxes)
{
    // Both have IoU 0.25 with the square; their measures, -0.0029 and 0.0326, straddle 0.
    const std::vector<float> boxes{0.0f, 0.0f, 10.0f, 10.0f, 0.0f, 6.0f,  10.0f, 16.0f,
                                   2.0f, 5.0f, 12.0f, 15.0f};
    const std::vector<float> scores{0.9f, 0.8f, 0.7f};
    std::vector<std::int64_t> kept;
    Detect::suppress_overlaps(boxes, scores, 0.0f, Detect::SuppressionCriterion::DistanceIoU,
                              0.6f, kept);
    EXPECT_EQ(kept, (std::vector<std::int64_t>{0, 1}));

    std::vector<std::int64_t> iou_kept;
    Detect::suppress_overlaps(boxes, scores, 0.0f, Detect::SuppressionCriterion::IoU, 0.6f,
                              iou_kept);
    EXPECT_EQ(iou_kept, (std::vector<std::int64_t>{0}));
}

/** @brief Tests that the IoU criterion keeps a box whose IoU equals the threshold. */
TEST(Small_DetectSuppressionCriterion, IoUCriterionKeepsABoxAtTheThreshold)
{
    const std::vector<float> boxes{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f};
    const std::vector<float> scores{0.9f, 0.8f};
    std::vector<std::int64_t> at_threshold;
    std::vector<std::int64_t> below_iou;
    Detect::suppress_overlaps(boxes, scores, 0.5f, Detect::SuppressionCriterion::IoU, 0.6f,
                              at_threshold);
    Detect::suppress_overlaps(boxes, scores, 0.25f, Detect::SuppressionCriterion::IoU, 0.6f,
                              below_iou);

    // The boxes share 2 of 4 + 2 - 2 = 4 units: an IoU of exactly 0.5.
    EXPECT_EQ(at_threshold, (std::vector<std::int64_t>{0, 1}));
    EXPECT_EQ(below_iou, (std::vector<std::int64_t>{0}));
}

/** @brief Tests that a zero-score row is never kept, even when it overlaps nothing. */
TEST(Small_DetectSuppressionCriterion, ZeroScoreRowsAreNeverKept)
{
    const std::vector<float> boxes{0.0f,  0.0f,  10.0f, 10.0f, 50.0f, 50.0f, 60.0f, 60.0f,
                                   20.0f, 20.0f, 30.0f, 30.0f};
    const std::vector<float> scores{0.9f, 0.0f, 0.5f};
    std::vector<std::int64_t> kept;
    Detect::suppress_overlaps(boxes, scores, 0.5f, Detect::SuppressionCriterion::IoU, 0.6f, kept);
    EXPECT_EQ(kept, (std::vector<std::int64_t>{0, 2}));
}

/** @brief Tests that an empty input yields no survivors. */
TEST(Small_DetectSuppressionCriterion, EmptyInputKeepsNothing)
{
    std::vector<std::int64_t> kept{7};
    Detect::suppress_overlaps({}, {}, 0.5f, Detect::SuppressionCriterion::DistanceIoU, 0.6f,
                              kept);
    EXPECT_TRUE(kept.empty());
}

/** @brief Tests that the four-argument form and the IoU form of the new overload agree. */
TEST(Small_DetectSuppressionCriterion, FourArgumentFormEqualsTheIoUCriterion)
{
    const std::vector<float> boxes{0.0f,  0.0f,  1.0f,  1.0f,  10.0f, 10.0f, 11.0f, 11.0f,
                                   20.0f, 20.0f, 21.0f, 21.0f, 30.0f, 30.0f, 31.0f, 31.0f};
    const std::vector<float> scores{0.1f, 0.9f, 0.5f, 0.5f};
    std::vector<std::int64_t> four_argument;
    std::vector<std::int64_t> six_argument;
    Detect::suppress_overlaps(boxes, scores, 0.5f, four_argument);
    Detect::suppress_overlaps(boxes, scores, 0.5f, Detect::SuppressionCriterion::IoU, 0.6f,
                              six_argument);
    EXPECT_EQ(four_argument, (std::vector<std::int64_t>{1, 2, 3, 0}));
    EXPECT_EQ(six_argument, four_argument);
}

/** @brief Tests that the selection passes the criterion and the exponent on to suppression. */
TEST(Small_DetectSuppressionCriterion, SelectionAppliesTheConfiguredCriterion)
{
    // The three boxes of DistanceIoUKeepsTheFartherOfTwoEqualIoUBoxes, as centre and size.
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 5.0f, 11.0f, 10.0f, 10.0f,
                                   7.0f, 10.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.9f, 0.8f, 0.7f};
    const Detect::PredictionShape shape{1, 3, 1};
    Detect::NMSConfig config;
    config.confidence_threshold = 0.1f;
    config.iou_threshold = 0.0f;
    config.criterion = Detect::SuppressionCriterion::DistanceIoU;

    std::vector<std::vector<float>> detections;
    SA_EXPECT_OK(Detect::select_nms_detections(boxes, scores, shape, config, detections));
    const std::vector<float> expected{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f,
                                      0.0f, 6.0f, 10.0f, 16.0f, 0.8f, 0.0f};
    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0], expected);
}

/** @brief Tests that the default configuration suppresses by IoU with exponent 0.6. */
TEST(Small_DetectSuppressionCriterion, NMSConfigDefaultsToTheIoUCriterion)
{
    const Detect::NMSConfig config;
    EXPECT_EQ(config.criterion, Detect::SuppressionCriterion::IoU);
    EXPECT_FLOAT_EQ(config.distance_exponent, 0.6f);
}

/** @brief Tests that boxes whose size is not four per score stop the process, with both counts. */
TEST(Small_DetectSuppressionCriterionDeathTest, RejectsBoxesOfTheWrongSizeFatally)
{
    const std::vector<float> boxes(5, 0.0f);
    const std::vector<float> scores(2, 0.5f);
    std::vector<std::int64_t> kept;
    SA_EXPECT_FATAL((void)Detect::suppress_overlaps(boxes, scores, 0.5f,
                                                    Detect::SuppressionCriterion::IoU, 0.6f, kept),
                    "a suppression needs", "got 5 and 2 values");
}

/** @brief Tests that a criterion value outside the enumeration stops the process. */
TEST(Small_DetectSuppressionCriterionDeathTest, RejectsAnUnknownCriterionFatally)
{
    const std::vector<float> boxes(4, 0.0f);
    const std::vector<float> scores(1, 0.5f);
    std::vector<std::int64_t> kept;
    SA_EXPECT_FATAL((void)Detect::suppress_overlaps(
                        boxes, scores, 0.5f, static_cast<Detect::SuppressionCriterion>(7), 0.6f,
                        kept),
                    "must be IoU or DistanceIoU", "got 7");
}

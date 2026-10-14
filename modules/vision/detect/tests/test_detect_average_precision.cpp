/**************************************************************************/
/* test_detect_average_precision.cpp                                      */
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
 * @file test_detect_average_precision.cpp
 * @brief Tests detection matching and the average-precision table against hand cases.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    using SushiAI::Vision::Detect::AveragePrecision;
    using SushiAI::Vision::Detect::K_IOU_THRESHOLD_COUNT;
    using SushiAI::Vision::Detect::MetricsSummary;

    /** @brief Holds the absolute tolerance of a metric: 4500 float64 steps at 1. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Holds the area under an envelope that is 1 up to recall 1: 0.99 + 0.01 / 2. */
    constexpr double K_FULL_AP = 0.995;

    /** @brief Holds an 8 by 8 ground-truth box. */
    const std::vector<float> K_BOX_A{0.0f, 0.0f, 8.0f, 8.0f};

    /** @brief Holds an 8 by 8 box that does not touch K_BOX_A. */
    const std::vector<float> K_BOX_B{20.0f, 20.0f, 28.0f, 28.0f};

    /** @brief Holds an 8 by 8 box that touches neither K_BOX_A nor K_BOX_B. */
    const std::vector<float> K_BOX_FAR{100.0f, 100.0f, 108.0f, 108.0f};

    /** @brief Holds the 8 by 5 box whose IoU with K_BOX_A is 40 / 64 = 0.625. */
    const std::vector<float> K_BOX_LOOSE{0.0f, 0.0f, 8.0f, 5.0f};

    /** @brief Holds the 8 by 7 box whose IoU with K_BOX_A is 56 / 64 = 0.875. */
    const std::vector<float> K_BOX_NEAR{0.0f, 0.0f, 8.0f, 7.0f};

    /** @brief Holds no detections and no boxes. */
    const std::vector<float> K_NO_VALUES;

    /** @brief Holds no classes. */
    const std::vector<std::int64_t> K_NO_CLASSES;

    /** @brief Returns one detection row: the box, then the score and the class. */
    std::vector<float> make_row(const std::vector<float>& box, float score, float label)
    {
        std::vector<float> row(box);
        row.push_back(score);
        row.push_back(label);
        return row;
    }

    /** @brief Returns the arrays of @p parts laid end to end. */
    std::vector<float> join(std::initializer_list<std::vector<float>> parts)
    {
        std::vector<float> joined;
        for (const std::vector<float>& part : parts)
            joined.insert(joined.end(), part.begin(), part.end());
        return joined;
    }

    /** @brief Returns the match flags of one image's detections. */
    std::vector<std::uint8_t> match(const std::vector<float>& detections,
                                    const std::vector<float>& boxes,
                                    const std::vector<std::int64_t>& classes)
    {
        const std::size_t rows = detections.size() / SushiAI::Vision::Detect::K_DETECTION_VALUES;
        std::vector<std::uint8_t> correct(rows * K_IOU_THRESHOLD_COUNT, 7);
        SushiAI::Vision::Detect::match_detections(detections, boxes, classes, correct);
        return correct;
    }

    /** @brief Adds one image whose arrays are vectors. */
    void add_image(AveragePrecision& metric,
                   const std::vector<float>& detections,
                   const std::vector<float>& boxes,
                   const std::vector<std::int64_t>& classes)
    {
        metric.add_image(detections, boxes, classes);
    }

    /** @brief Returns the table of a single image. */
    MetricsSummary summarize_image(const std::vector<float>& detections,
                                   const std::vector<float>& boxes,
                                   const std::vector<std::int64_t>& classes)
    {
        AveragePrecision metric;
        add_image(metric, detections, boxes, classes);
        return metric.summarize();
    }
}

/** @brief Tests that the ten thresholds carry the float32 bits of torch.linspace(0.5, 0.95, 10). */
TEST(Unit_AveragePrecision, HoldsTheTenReferenceThresholds)
{
    const std::vector<float> expected{0x1.0p-1f,      0x1.19999ap-1f, 0x1.333334p-1f,
                                      0x1.4cccccp-1f, 0x1.666666p-1f, 0x1.8p-1f,
                                      0x1.99999ap-1f, 0x1.b33334p-1f, 0x1.ccccccp-1f,
                                      0x1.e66666p-1f};
    ASSERT_EQ(SushiAI::Vision::Detect::K_IOU_THRESHOLDS.size(), expected.size());
    for (std::size_t level = 0; level < expected.size(); ++level)
        EXPECT_EQ(SushiAI::Vision::Detect::K_IOU_THRESHOLDS[level], expected[level]) << level;
}

/** @brief Tests that an empty curve integrates to the area under the line from 1 to 0. */
TEST(Unit_AveragePrecision, IntegratesAnEmptyCurveToOneHalf)
{
    const std::vector<double> none;

    // Recall 0 1 1 with envelope 1 0 0 gives 1 - x, whose 100 trapezoids sum to 0.5.
    EXPECT_NEAR(SushiAI::Vision::Detect::integrate_precision(none, none), 0.5, K_TOLERANCE);
}

/** @brief Tests that the envelope lifts a precision dip to the larger value at higher recall. */
TEST(Unit_AveragePrecision, IntegratesUnderThePrecisionEnvelope)
{
    const std::vector<double> recall{0.5, 0.5, 1.0};
    const std::vector<double> precision{1.0, 0.5, 2.0 / 3.0};

    // 50 points at 1, 50 at 2/3, the last at 0: 0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3.
    EXPECT_NEAR(SushiAI::Vision::Detect::integrate_precision(recall, precision),
                0.49 + 0.05 / 6.0 + 0.98 / 3.0 + 0.01 / 3.0, K_TOLERANCE);
}

/** @brief Tests that a curve with one length for recall and another for precision is refused. */
TEST(Unit_AveragePrecision, RejectsCurvesOfUnequalLength)
{
    const std::vector<double> recall{0.5, 1.0};
    const std::vector<double> precision{1.0};
    SA_EXPECT_ERROR((void)SushiAI::Vision::Detect::integrate_precision(recall, precision),
                    "1 and 2");
}

/** @brief Tests that a detector that finds every box exactly scores 0.995 and not 1. */
TEST(Unit_AveragePrecision, PerfectDetectorScoresTheInterpolationCeiling)
{
    AveragePrecision metric;
    add_image(metric, make_row(K_BOX_A, 0.9f, 0.0f), K_BOX_A, {0});
    add_image(metric, make_row(K_BOX_B, 0.8f, 1.0f), K_BOX_B, {1});
    const MetricsSummary summary = metric.summarize();

    // Recall 0 1 1 1, envelope 1 1 0 0: 100 points at 1 and the point at recall 1 at 0.
    ASSERT_EQ(summary.classes.size(), 2u);
    for (const SushiAI::Vision::Detect::ClassMetrics& row : summary.classes)
    {
        EXPECT_NEAR(row.precision, 1.0, K_TOLERANCE);
        EXPECT_NEAR(row.recall, 1.0, K_TOLERANCE);
        EXPECT_NEAR(row.ap50, K_FULL_AP, K_TOLERANCE);
        EXPECT_NEAR(row.ap50_95, K_FULL_AP, K_TOLERANCE);
    }
    EXPECT_NEAR(summary.precision, 1.0, K_TOLERANCE);
    EXPECT_NEAR(summary.recall, 1.0, K_TOLERANCE);
    EXPECT_NEAR(summary.map50, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.map50_95, K_FULL_AP, K_TOLERANCE);
}

/** @brief Tests that a class with boxes and no detections scores zero in every column. */
TEST(Unit_AveragePrecision, NoDetectionsScoreZero)
{
    const MetricsSummary summary = summarize_image(K_NO_VALUES, K_BOX_A, {0});

    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_EQ(summary.classes[0].label, 0);
    EXPECT_EQ(summary.classes[0].instances, 1);
    EXPECT_EQ(summary.classes[0].precision, 0.0);
    EXPECT_EQ(summary.classes[0].recall, 0.0);
    EXPECT_EQ(summary.classes[0].ap50, 0.0);
    EXPECT_EQ(summary.classes[0].ap50_95, 0.0);
    EXPECT_EQ(summary.precision, 0.0);
    EXPECT_EQ(summary.recall, 0.0);
    EXPECT_EQ(summary.map50, 0.0);
    EXPECT_EQ(summary.map50_95, 0.0);
}

/** @brief Tests that a false positive scored above the only true positive gives AP 0.4975. */
TEST(Unit_AveragePrecision, FalsePositiveAboveTheTruePositiveHalvesPrecision)
{
    const std::vector<float> detections =
        join({make_row(K_BOX_FAR, 0.9f, 0.0f), make_row(K_BOX_A, 0.6f, 0.0f)});
    const MetricsSummary summary = summarize_image(detections, K_BOX_A, {0});

    // Recall 0 1, precision 0 0.5; the envelope is 0.5 on [0, 1): 0.99 * 0.5 + 0.01 * 0.25.
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_NEAR(summary.classes[0].ap50, 0.4975, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].ap50_95, 0.4975, K_TOLERANCE);
    EXPECT_NEAR(summary.map50, 0.4975, K_TOLERANCE);

    // Mean F1 is flat at 2/3 below confidence 0.6, so the first point is read: P 1/2, R 1.
    EXPECT_NEAR(summary.classes[0].precision, 0.5, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].recall, 1.0, K_TOLERANCE);
}

/** @brief Tests that precision and recall are read where the smoothed mean F1 is largest. */
TEST(Unit_AveragePrecision, ReadsPrecisionAndRecallAtTheBestF1Confidence)
{
    const std::vector<float> detections =
        join({make_row(K_BOX_A, 0.9f, 0.0f), make_row(K_BOX_FAR, 0.3f, 0.0f)});
    const MetricsSummary summary = summarize_image(detections, K_BOX_A, {0});

    // F1 climbs from 2/3 at 0.3 to 1 at 0.9 and is 0 above: the window ending at 899 wins.
    const double high = static_cast<double>(0.9f);
    const double low = static_cast<double>(0.3f);
    const double confidence = 849.0 / 999.0;

    // Precision falls from 1 at 0.9 to 1/2 at 0.3, so point 849 reads 0.958208...; point 0, 1/2.
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_NEAR(summary.classes[0].precision, 1.0 - 0.5 * (high - confidence) / (high - low),
                K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].precision, 0.9582082255854392, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].recall, 1.0, K_TOLERANCE);
    EXPECT_NEAR(summary.precision, 0.9582082255854392, K_TOLERANCE);
    EXPECT_NEAR(summary.recall, 1.0, K_TOLERANCE);
}

/** @brief Tests that the second of two detections on one box is a false positive. */
TEST(Unit_AveragePrecision, TwoDetectionsOnOneBoxCountOneMatch)
{
    const std::vector<float> detections = join({make_row(K_BOX_A, 0.9f, 0.0f),
                                                make_row(K_BOX_A, 0.8f, 0.0f),
                                                make_row(K_BOX_B, 0.7f, 0.0f)});
    const std::vector<float> boxes = join({K_BOX_A, K_BOX_B});
    const std::vector<std::uint8_t> correct = match(detections, boxes, {0, 0});
    for (std::size_t level = 0; level < K_IOU_THRESHOLD_COUNT; ++level)
    {
        EXPECT_EQ(correct[level], 1) << level;
        EXPECT_EQ(correct[K_IOU_THRESHOLD_COUNT + level], 0) << level;
        EXPECT_EQ(correct[2 * K_IOU_THRESHOLD_COUNT + level], 1) << level;
    }

    // Recall 1/2 1/2 1, precision 1 1/2 2/3; three matches would give 0.995 instead.
    const MetricsSummary summary = summarize_image(detections, boxes, {0, 0});
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_NEAR(summary.classes[0].ap50, 0.49 + 0.05 / 6.0 + 0.98 / 3.0 + 0.01 / 3.0,
                K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].precision, 2.0 / 3.0, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].recall, 1.0, K_TOLERANCE);
}

/** @brief Tests that a detected class without ground truth gets no row and no share of a mean. */
TEST(Unit_AveragePrecision, ClassWithoutGroundTruthIsLeftOutOfTheMean)
{
    const std::vector<float> detections =
        join({make_row(K_BOX_A, 0.9f, 0.0f), make_row(K_BOX_A, 0.8f, 3.0f)});
    const MetricsSummary summary = summarize_image(detections, K_BOX_A, {0});

    // A row for class 3 would halve every mean to 0.4975 and 0.5.
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_EQ(summary.classes[0].label, 0);
    EXPECT_NEAR(summary.map50, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.map50_95, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.precision, 1.0, K_TOLERANCE);
    EXPECT_NEAR(summary.recall, 1.0, K_TOLERANCE);
}

/** @brief Tests that a class with ground truth and no detections enters the means as zero. */
TEST(Unit_AveragePrecision, ClassWithoutDetectionsCountsAsZeroInTheMean)
{
    const MetricsSummary summary =
        summarize_image(make_row(K_BOX_A, 0.9f, 0.0f), join({K_BOX_A, K_BOX_B}), {0, 1});

    // Rows 0.995 and 0: the means are 0.995 / 2 for AP and 1 / 2 for precision and recall.
    ASSERT_EQ(summary.classes.size(), 2u);
    EXPECT_EQ(summary.classes[1].label, 1);
    EXPECT_EQ(summary.classes[1].ap50, 0.0);
    EXPECT_NEAR(summary.map50, 0.4975, K_TOLERANCE);
    EXPECT_NEAR(summary.map50_95, 0.4975, K_TOLERANCE);
    EXPECT_NEAR(summary.precision, 0.5, K_TOLERANCE);
    EXPECT_NEAR(summary.recall, 0.5, K_TOLERANCE);
}

/** @brief Tests that detections of equal score are ranked in the order they were added. */
TEST(Unit_AveragePrecision, EqualScoresKeepTheOrderTheyWereAddedIn)
{
    const std::vector<float> miss = make_row(K_BOX_FAR, 0.5f, 0.0f);
    const std::vector<float> hit = make_row(K_BOX_A, 0.5f, 0.0f);

    // Miss first: recall 0 1, precision 0 1/2, as with distinct scores, so 0.4975.
    const MetricsSummary miss_first = summarize_image(join({miss, hit}), K_BOX_A, {0});
    ASSERT_EQ(miss_first.classes.size(), 1u);
    EXPECT_NEAR(miss_first.classes[0].ap50, 0.4975, K_TOLERANCE);

    // Hit first: recall 1 1, precision 1 1/2; the envelope is 1 up to recall 1, so 0.995.
    const MetricsSummary hit_first = summarize_image(join({hit, miss}), K_BOX_A, {0});
    ASSERT_EQ(hit_first.classes.size(), 1u);
    EXPECT_NEAR(hit_first.classes[0].ap50, K_FULL_AP, K_TOLERANCE);
}

/** @brief Tests that a detection at IoU 0.625 counts at 0.5, 0.55 and 0.6 and at no other. */
TEST(Unit_AveragePrecision, LooseDetectionCountsAtTheLowThresholdsOnly)
{
    const std::vector<float> detections = make_row(K_BOX_LOOSE, 0.9f, 0.0f);
    const std::vector<std::uint8_t> correct = match(detections, K_BOX_A, {0});
    EXPECT_EQ(correct, (std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0, 0, 0, 0, 0}));

    // AP is 0.995 at three thresholds and 0 at seven: 3 * 0.995 / 10 = 0.2985.
    const MetricsSummary summary = summarize_image(detections, K_BOX_A, {0});
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_NEAR(summary.classes[0].ap50, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].ap50_95, 0.2985, K_TOLERANCE);
    EXPECT_NEAR(summary.map50, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.map50_95, 0.2985, K_TOLERANCE);
}

/** @brief Tests that a detection in an image without ground truth is a false positive. */
TEST(Unit_AveragePrecision, ImageWithoutGroundTruthAddsFalsePositives)
{
    AveragePrecision metric;
    add_image(metric, make_row(K_BOX_A, 0.6f, 0.0f), K_BOX_A, {0});
    add_image(metric, make_row(K_BOX_A, 0.9f, 0.0f), K_NO_VALUES, K_NO_CLASSES);
    const MetricsSummary summary = metric.summarize();

    // The unlabelled image's detection ranks first: recall 0 1, precision 0 1/2, so 0.4975.
    EXPECT_EQ(summary.images, 2);
    EXPECT_EQ(summary.instances, 1);
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_EQ(summary.classes[0].images, 1);
    EXPECT_NEAR(summary.classes[0].ap50, 0.4975, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].precision, 0.5, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].recall, 1.0, K_TOLERANCE);
}

/** @brief Tests that the boxes of an image without detections lower recall and AP. */
TEST(Unit_AveragePrecision, ImageWithoutDetectionsLowersRecall)
{
    AveragePrecision metric;
    add_image(metric, make_row(K_BOX_A, 0.6f, 0.0f), K_BOX_A, {0});
    add_image(metric, K_NO_VALUES, K_BOX_A, {0});
    const MetricsSummary summary = metric.summarize();

    // Recall 1/2, precision 1: 50 points at 1, then 0 from recall 1/2 on: 0.49 + 0.005.
    EXPECT_EQ(metric.image_count(), 2);
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_EQ(summary.classes[0].images, 2);
    EXPECT_EQ(summary.classes[0].instances, 2);
    EXPECT_NEAR(summary.classes[0].ap50, 0.495, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].ap50_95, 0.495, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].precision, 1.0, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[0].recall, 0.5, K_TOLERANCE);
}

/** @brief Tests that a detection whose best box is taken is not given its second-best box. */
TEST(Unit_AveragePrecision, DetectionWhoseBestBoxIsTakenStaysUnmatched)
{
    const std::vector<float> detections =
        join({make_row(K_BOX_A, 0.9f, 0.0f), make_row(K_BOX_NEAR, 0.8f, 0.0f)});

    // The second detection overlaps K_BOX_A by 0.875 and K_BOX_LOOSE by 40 / 56 = 0.714.
    const std::vector<std::uint8_t> correct =
        match(detections, join({K_BOX_A, K_BOX_LOOSE}), {0, 0});
    EXPECT_EQ(correct, (std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

/** @brief Tests that a detection over two boxes matches the one of larger IoU, listed second. */
TEST(Unit_AveragePrecision, DetectionTakesTheBoxOfLargestIoU)
{
    // IoU is 40 / 56 = 0.714 with K_BOX_LOOSE and 56 / 64 = 0.875 with K_BOX_A: 0.85 passes.
    const std::vector<std::uint8_t> correct =
        match(make_row(K_BOX_NEAR, 0.9f, 0.0f), join({K_BOX_LOOSE, K_BOX_A}), {0, 0});
    EXPECT_EQ(correct, (std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 0, 0}));
}

/** @brief Tests that a detection whose IoU is not a number matches at no threshold. */
TEST(Unit_AveragePrecision, DetectionWithNaNBoxDoesNotMatch)
{
    const std::vector<float> detection{std::numeric_limits<float>::quiet_NaN(),
                                       0.0f, 8.0f, 8.0f, 0.9f, 0.0f};
    const std::vector<std::uint8_t> correct = match(detection, K_BOX_A, {0});
    EXPECT_EQ(correct, std::vector<std::uint8_t>(K_IOU_THRESHOLD_COUNT, 0));
}

/** @brief Tests that a box goes to its first detection and not to the one of larger IoU. */
TEST(Unit_AveragePrecision, BoxGoesToItsFirstDetectionAtEachThreshold)
{
    const std::vector<float> detections =
        join({make_row(K_BOX_LOOSE, 0.9f, 0.0f), make_row(K_BOX_A, 0.8f, 0.0f)});

    // Row 0 has IoU 0.625 and holds the box up to 0.6; row 1, with IoU 1, gets it from 0.65.
    const std::vector<std::uint8_t> correct = match(detections, K_BOX_A, {0});
    EXPECT_EQ(correct, (std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1}));
}

/** @brief Tests that a detection of another class does not match a box it covers exactly. */
TEST(Unit_AveragePrecision, DetectionOfAnotherClassDoesNotMatch)
{
    const std::vector<std::uint8_t> correct = match(make_row(K_BOX_A, 0.9f, 1.0f), K_BOX_A, {0});
    EXPECT_EQ(correct, std::vector<std::uint8_t>(K_IOU_THRESHOLD_COUNT, 0));
}

/** @brief Tests that an accumulator without ground truth reports no rows and zero means. */
TEST(Unit_AveragePrecision, NoGroundTruthGivesNoRows)
{
    AveragePrecision metric;
    EXPECT_TRUE(metric.summarize().classes.empty());

    add_image(metric, make_row(K_BOX_A, 0.9f, 0.0f), K_NO_VALUES, K_NO_CLASSES);
    const MetricsSummary summary = metric.summarize();
    EXPECT_EQ(summary.images, 1);
    EXPECT_TRUE(summary.classes.empty());
    EXPECT_EQ(summary.precision, 0.0);
    EXPECT_EQ(summary.recall, 0.0);
    EXPECT_EQ(summary.map50, 0.0);
    EXPECT_EQ(summary.map50_95, 0.0);
}

/** @brief Tests that clear() leaves nothing of the images added before it. */
TEST(Unit_AveragePrecision, ClearDropsEveryImage)
{
    AveragePrecision metric;
    add_image(metric, make_row(K_BOX_FAR, 0.9f, 0.0f), K_BOX_A, {0});
    metric.clear();
    EXPECT_EQ(metric.image_count(), 0);

    add_image(metric, make_row(K_BOX_A, 0.9f, 0.0f), K_BOX_A, {0});
    const MetricsSummary summary = metric.summarize();
    EXPECT_EQ(summary.images, 1);
    EXPECT_EQ(summary.instances, 1);
    ASSERT_EQ(summary.classes.size(), 1u);
    EXPECT_NEAR(summary.classes[0].ap50, K_FULL_AP, K_TOLERANCE);
}

/** @brief Tests that malformed detections and ground truth are refused and leave no image. */
TEST(Unit_AveragePrecision, RejectsMalformedImages)
{
    AveragePrecision metric;
    const std::vector<float> row = make_row(K_BOX_A, 0.9f, 0.0f);
    const std::vector<float> short_row(5, 0.0f);
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_ERROR(add_image(metric, short_row, K_BOX_A, {0}), "[N, 6]", "got 5");
    SA_EXPECT_ERROR(add_image(metric, row, K_BOX_A, {0, 1}), "4 values", "2 classes");
    SA_EXPECT_ERROR(add_image(metric, row, K_BOX_A, {-1}), "box 0", "negative class -1");
    SA_EXPECT_ERROR(add_image(metric, make_row(K_BOX_A, 0.9f, 0.5f), K_BOX_A, {0}),
                    "detection 0", "not a class index");
    SA_EXPECT_ERROR(add_image(metric, make_row(K_BOX_A, not_a_number, 0.0f), K_BOX_A, {0}),
                    "detection 0", "score");
    EXPECT_EQ(metric.image_count(), 0);
    EXPECT_TRUE(metric.summarize().classes.empty());

    std::vector<std::uint8_t> correct(9);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::match_detections(row, K_BOX_A, K_NO_CLASSES, correct),
                    "0 classes");
    const std::vector<std::int64_t> classes{0};
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::match_detections(row, K_BOX_A, classes, correct),
                    "need 10 values", "got 9");
}

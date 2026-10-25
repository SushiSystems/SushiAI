/**************************************************************************/
/* test_batch_average_precision.cpp                                       */
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
 * @file test_batch_average_precision.cpp
 * @brief Tests the feed of a batch into AveragePrecision against a mean computed by hand.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/batch_average_precision.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace
{
    using SushiAI::Vision::Detect::add_batch;
    using SushiAI::Vision::Detect::AveragePrecision;
    using SushiAI::Vision::Detect::GroundTruthBatch;
    using SushiAI::Vision::Detect::MetricsSummary;

    /** @brief Holds the absolute tolerance of a metric: 4500 float64 steps at 1. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Holds the area under an envelope that is 1 up to recall 1: 0.99 + 0.01 / 2. */
    constexpr double K_FULL_AP = 0.995;

    /** @brief Holds the AP of class 0 below: 49 trapezoids at 1 and one from 1 down to 0. */
    constexpr double K_CLASS0_AP = 0.495;

    /**
     * @brief Returns the two-image batch every test reads.
     *
     * Image 0 holds a class 0 box and a class 1 box; image 1 holds one class 0 box and a
     * padding slot. The three boxes are 8 by 8 and far apart.
     */
    GroundTruthBatch make_truth()
    {
        GroundTruthBatch truth;
        truth.batch = 2;
        truth.max_boxes = 2;
        truth.labels = {0, 1, 0, 0};
        truth.boxes = {0.0f,  0.0f,  8.0f,  8.0f,  // image 0, class 0
                       20.0f, 20.0f, 28.0f, 28.0f, // image 0, class 1
                       0.0f,  0.0f,  8.0f,  8.0f,  // image 1, class 0
                       0.0f,  0.0f,  0.0f,  0.0f}; // padding
        truth.valid = {1, 1, 1, 0};
        return truth;
    }
}

/**
 * @brief Tests the mean over a class with a miss and a false positive and a class found whole.
 *
 * Class 0 has two boxes. Its detections by score are 0.9, a true positive in image 0, and 0.8, a
 * false positive in image 1; the box of image 1 is missed. Recall is 0.5 and 0.5, precision 1 and
 * 1/2. The envelope is 1 up to recall 0.5 and 0 from there, so the 101 points give 49 trapezoids
 * at 1 and one from 1 to 0: 0.49 + 0.005. Class 1 has one box, found: 0.995. The mean is 0.745.
 */
TEST(Small_BatchAveragePrecision, MeansAClassWithAMissAndAFalsePositiveWithAFoundOne)
{
    const std::vector<std::vector<float>> detections{
        {0.0f, 0.0f, 8.0f, 8.0f, 0.9f, 0.0f, 20.0f, 20.0f, 28.0f, 28.0f, 0.7f, 1.0f},
        {100.0f, 100.0f, 108.0f, 108.0f, 0.8f, 0.0f}};
    AveragePrecision metric;
    add_batch(metric, detections, make_truth());
    const MetricsSummary summary = metric.summarize();

    // The padding slot would add a fourth box, of class 0, if it were not dropped.
    EXPECT_EQ(summary.images, 2);
    EXPECT_EQ(summary.instances, 3);
    ASSERT_EQ(summary.classes.size(), 2u);
    EXPECT_EQ(summary.classes[0].instances, 2);
    EXPECT_NEAR(summary.classes[0].ap50, K_CLASS0_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.classes[1].ap50, K_FULL_AP, K_TOLERANCE);
    EXPECT_NEAR(summary.map50, (K_CLASS0_AP + K_FULL_AP) / 2.0, K_TOLERANCE);
    EXPECT_NEAR(summary.map50, 0.745, K_TOLERANCE);
}

/** @brief Tests that two batches add up in one metric as two images would. */
TEST(Small_BatchAveragePrecision, BatchesAccumulateIntoOneMetric)
{
    GroundTruthBatch first;
    first.batch = 1;
    first.max_boxes = 1;
    first.labels = {0};
    first.boxes = {0.0f, 0.0f, 8.0f, 8.0f};
    first.valid = {1};

    const std::vector<std::vector<float>> found{{0.0f, 0.0f, 8.0f, 8.0f, 0.9f, 0.0f}};
    const std::vector<std::vector<float>> missed(1);
    AveragePrecision metric;
    add_batch(metric, found, first);
    add_batch(metric, missed, first);

    // Two boxes of class 0, one found: recall 0.5 and precision 1, the AP of class 0 above.
    EXPECT_EQ(metric.image_count(), 2);
    EXPECT_NEAR(metric.summarize().map50, K_CLASS0_AP, K_TOLERANCE);
}

/** @brief Tests that a batch whose detection arrays do not match its images is refused. */
TEST(Small_BatchAveragePrecision, RejectsAnArrayCountThatIsNotTheBatch)
{
    AveragePrecision metric;
    const std::vector<std::vector<float>> one_array(1);

    SA_EXPECT_ERROR(add_batch(metric, one_array, make_truth()), "2 images", "got 1 arrays");
    EXPECT_EQ(metric.image_count(), 0);
}

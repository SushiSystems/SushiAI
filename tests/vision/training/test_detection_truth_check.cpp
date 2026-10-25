/**************************************************************************/
/* test_detection_truth_check.cpp                                         */
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

/**
 * @file test_detection_truth_check.cpp
 * @brief Tests the checks of a ground truth that both detection training steps run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/detection_truth_check.hpp>

using namespace SushiAI;
using Vision::Detect::GroundTruthBatch;
using Vision::Training::check_truth_labels;
using Vision::Training::check_truth_layout;

namespace
{
    /** @brief Holds the images of the truth below. */
    constexpr std::int64_t K_IMAGES = 2;

    /** @brief Holds the rows of one image of the truth below. */
    constexpr std::int64_t K_ROWS = 2;

    /** @brief Holds the classes the labels are checked against. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Returns a truth of K_IMAGES images and K_ROWS rows, the last row padding. */
    GroundTruthBatch make_truth()
    {
        GroundTruthBatch truth;
        truth.batch = K_IMAGES;
        truth.max_boxes = K_ROWS;
        truth.labels = {0, 2, 1, 0};
        truth.valid = {1, 1, 1, 0};
        truth.boxes = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                       0.0f, 0.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        return truth;
    }
} // namespace

/** @brief Tests that a truth of the declared batch and matching arrays passes. */
TEST(Small_DetectionTruthCheck, AcceptsATruthOfTheDeclaredBatch)
{
    const GroundTruthBatch truth = make_truth();

    SA_EXPECT_OK(check_truth_layout(truth, K_IMAGES, "heads were declared for"));
    SA_EXPECT_OK(check_truth_labels(truth, K_CLASSES, "head.0"));
}

/** @brief Tests that another batch is Incompatible and the message names both and the origin. */
TEST(Small_DetectionTruthCheck, ATruthOfAnotherBatchNamesBothBatchesAndTheOrigin)
{
    GroundTruthBatch truth = make_truth();
    truth.batch = 3;

    SA_EXPECT_FAILURE(check_truth_layout(truth, K_IMAGES, "ports were traced for"),
                      StatusCode::Incompatible, "3 images", "ports were traced for a batch of 2");
}

/** @brief Tests that a negative row count is refused with its value. */
TEST(Small_DetectionTruthCheck, ANegativeRowCountIsRefused)
{
    GroundTruthBatch truth = make_truth();
    truth.max_boxes = -1;

    SA_EXPECT_FAILURE(check_truth_layout(truth, K_IMAGES, "heads were declared for"),
                      StatusCode::Incompatible, "-1 rows", "cannot be negative");
}

/** @brief Tests that each array of the wrong size is refused with the sizes it needs. */
TEST(Small_DetectionTruthCheck, AnArrayOfAnotherSizeIsRefusedWithTheSizesItNeeds)
{
    GroundTruthBatch short_labels = make_truth();
    short_labels.labels.pop_back();
    SA_EXPECT_FAILURE(check_truth_layout(short_labels, K_IMAGES, "heads were declared for"),
                      StatusCode::Incompatible, "needs 4 labels, 4 flags and 16 box values",
                      "holds 3, 4 and 16");

    GroundTruthBatch short_valid = make_truth();
    short_valid.valid.pop_back();
    SA_EXPECT_FAILURE(check_truth_layout(short_valid, K_IMAGES, "heads were declared for"),
                      StatusCode::Incompatible, "holds 4, 3 and 16");

    GroundTruthBatch short_boxes = make_truth();
    short_boxes.boxes.pop_back();
    SA_EXPECT_FAILURE(check_truth_layout(short_boxes, K_IMAGES, "heads were declared for"),
                      StatusCode::Incompatible, "holds 4, 4 and 15");
}

/** @brief Tests that a valid row with a label at or above the classes is refused by its row. */
TEST(Small_DetectionTruthCheck, AValidLabelOutsideTheClassesNamesTheRowAndTheHolder)
{
    GroundTruthBatch truth = make_truth();
    truth.labels[2] = K_CLASSES;

    SA_EXPECT_FAILURE(check_truth_labels(truth, K_CLASSES, "head.1"),
                      StatusCode::InvalidDescription, "row 2", "label 3", "head.1 has 3 classes");
}

/** @brief Tests that a negative label of a valid row is refused. */
TEST(Small_DetectionTruthCheck, ANegativeValidLabelIsRefused)
{
    GroundTruthBatch truth = make_truth();
    truth.labels[0] = -1;

    SA_EXPECT_FAILURE(check_truth_labels(truth, K_CLASSES, "the ports"),
                      StatusCode::InvalidDescription, "row 0", "label -1");
}

/** @brief Tests that a padding row may hold any label, since no step reads it. */
TEST(Small_DetectionTruthCheck, APaddingRowMayHoldAnyLabel)
{
    GroundTruthBatch truth = make_truth();
    truth.labels[3] = 99;

    SA_EXPECT_OK(check_truth_labels(truth, K_CLASSES, "the ports"));
}

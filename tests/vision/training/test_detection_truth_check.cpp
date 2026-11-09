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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */

/**
 * @file test_detection_truth_check.cpp
 * @brief Tests the checks of a ground truth that both detection training steps run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
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

    /** @brief Names an array to shorten and the sizes the refusal states. */
    struct ShortArray
    {
        void (*shorten)(GroundTruthBatch&);  /**< Removes the last element of one array. */
        const char* sizes;                   /**< What the message says the truth holds. */
    };

    /** @brief Provides the arrays of another size than the layout needs. */
    class Small_DetectionTruthCheckSizes : public ::testing::TestWithParam<ShortArray>
    {

    };

    /** @brief Names a label to set on a row and what the refusal names. */
    struct StrayLabel
    {
        std::size_t row;        /**< The row that receives the label. */
        std::int32_t label;     /**< The label set. */
        const char* holder;     /**< Names who holds the classes. */
        const char* needles[3]; /**< What the message names; "" matches any. */
    };

    /** @brief Provides the labels of valid rows outside the classes. */
    class Small_DetectionTruthCheckLabels : public ::testing::TestWithParam<StrayLabel>
    {

    };
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
TEST_P(Small_DetectionTruthCheckSizes, AnArrayOfAnotherSizeIsRefusedWithTheSizesItNeeds)
{
    GroundTruthBatch truth = make_truth();
    GetParam().shorten(truth);

    SA_EXPECT_FAILURE(check_truth_layout(truth, K_IMAGES, "heads were declared for"),
                      StatusCode::Incompatible, "needs 4 labels, 4 flags and 16 box values",
                      GetParam().sizes);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectionTruthCheckSizes,
    ::testing::Values(
        ShortArray{[](GroundTruthBatch& truth) { truth.labels.pop_back(); }, "holds 3, 4 and 16"},
        ShortArray{[](GroundTruthBatch& truth) { truth.valid.pop_back(); }, "holds 4, 3 and 16"},
        ShortArray{[](GroundTruthBatch& truth) { truth.boxes.pop_back(); }, "holds 4, 4 and 15"}));

/** @brief Tests that a valid row with a label outside the classes is refused by its row. */
TEST_P(Small_DetectionTruthCheckLabels, AValidLabelOutsideTheClassesNamesTheRowAndTheHolder)
{
    const StrayLabel& row = GetParam();
    GroundTruthBatch truth = make_truth();
    truth.labels[row.row] = row.label;

    SA_EXPECT_FAILURE(check_truth_labels(truth, K_CLASSES, row.holder),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1],
                      row.needles[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectionTruthCheckLabels,
    ::testing::Values(
        StrayLabel{2, K_CLASSES, "head.1", {"row 2", "label 3", "head.1 has 3 classes"}},
        StrayLabel{0, -1, "the ports", {"row 0", "label -1", ""}}));

/** @brief Tests that a padding row may hold any label, since no step reads it. */
TEST(Small_DetectionTruthCheck, APaddingRowMayHoldAnyLabel)
{
    GroundTruthBatch truth = make_truth();
    truth.labels[3] = 99;

    SA_EXPECT_OK(check_truth_labels(truth, K_CLASSES, "the ports"));
}

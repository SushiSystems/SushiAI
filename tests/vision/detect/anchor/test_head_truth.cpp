/**************************************************************************/
/* test_head_truth.cpp                                                    */
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
/**************************************************************************/


/**
 * @file test_head_truth.cpp
 * @brief Tests which truth rows an anchor head accepts and how it names the ones it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_truth.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::compute_head_delta;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::validate_head_truth;
    using SushiAI::Vision::Detect::GroundTruthBatch;

    /** @brief Holds the classes of the head every case is checked against. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the square network input of 64 pixels, on which every corner is exact. */
    constexpr NetworkSize K_NETWORK{64, 64};

    /**
     * @brief Returns a one-image truth of two rows; row 0 is a valid box and row 1 the case's.
     * @param corners Row 1 as `x1 y1 x2 y2` in pixels.
     */
    GroundTruthBatch make_truth(const std::vector<float>& corners, std::int64_t label)
    {
        GroundTruthBatch truth;
        truth.batch = 1;
        truth.max_boxes = 2;
        truth.labels = {0, label};
        truth.valid = {1, 1};
        truth.boxes = {16.0f, 16.0f, 48.0f, 48.0f};
        truth.boxes.insert(truth.boxes.end(), corners.begin(), corners.end());
        return truth;
    }

    /** @brief Holds a truth row the check refuses and the words its message names. */
    struct RefusedRow
    {
        std::string name;            /**< The row's name. */
        std::vector<float> corners;  /**< Row 1 as `x1 y1 x2 y2` in pixels. */
        std::int64_t label;          /**< The class of row 1. */
        std::string first;           /**< A fragment the message names. */
        std::string second;          /**< A second fragment, empty when the row has none. */
    };

    /** @brief Runs one truth row the check refuses. */
    class Small_DetectAnchorHeadTruthRefusal : public ::testing::TestWithParam<RefusedRow>
    {

    };

    /** @brief Names a row by its name field. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusedRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that rows inside the input pass, a centre of exactly 0 included. */
TEST(Small_DetectAnchorHeadTruth, AcceptsRowsWhoseCentreLiesInsideTheInput)
{
    // Corners of -8 and 8 give a centre of 0, the closed end of [0, 1).
    const GroundTruthBatch truth = make_truth({-8.0f, -8.0f, 8.0f, 8.0f}, 2);

    SA_EXPECT_OK(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK));
}

/** @brief Tests that a row refused for its centre, size or class is named with its value. */
TEST_P(Small_DetectAnchorHeadTruthRefusal, ARefusedRowIsNamedWithItsValue)
{
    const RefusedRow& row = GetParam();
    const GroundTruthBatch truth = make_truth(row.corners, row.label);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    CentreOutside, Small_DetectAnchorHeadTruthRefusal,
    ::testing::Values(
        RefusedRow{"AtTheRightEdge", {56.0f, 16.0f, 72.0f, 48.0f}, 0, "centre", "(1, 0.5)"},
        RefusedRow{"LeftOfTheInput", {-48.0f, 16.0f, 16.0f, 48.0f}, 0, "(-0.25, 0.5)", ""},
        RefusedRow{"BelowTheInput", {16.0f, 64.0f, 48.0f, 96.0f}, 0, "(0.5, 1.25)", ""},
        RefusedRow{"AboveTheInput", {16.0f, -48.0f, 48.0f, 16.0f}, 0, "(0.5, -0.25)", ""}),
    refusal_name);

/** @brief Tests that a centre that is not a number is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreThatIsNotANumber)
{
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    const GroundTruthBatch truth = make_truth({not_a_number, 16.0f, 48.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "centre");
}

INSTANTIATE_TEST_SUITE_P(
    SideNotPositive, Small_DetectAnchorHeadTruthRefusal,
    ::testing::Values(
        RefusedRow{"AWidthOfZero", {32.0f, 16.0f, 32.0f, 48.0f}, 0, "size 0 by 0.5", ""},
        RefusedRow{"ANegativeHeight", {16.0f, 40.0f, 48.0f, 24.0f}, 0, "size 0.5 by -0.25", ""}),
    refusal_name);

/** @brief Tests that a width that overflows to infinity around a centre of 0 is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesASideThatOverflowsToInfinity)
{
    const GroundTruthBatch truth = make_truth({-3.0e38f, 16.0f, 3.0e38f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "finite");
}

INSTANTIATE_TEST_SUITE_P(
    ClassOutOfRange, Small_DetectAnchorHeadTruthRefusal,
    ::testing::Values(
        RefusedRow{"AtTheClassCount", {16.0f, 16.0f, 48.0f, 48.0f}, K_CLASSES, "class 3",
                   "3 classes"},
        RefusedRow{"ANegativeClass", {16.0f, 16.0f, 48.0f, 48.0f}, -1, "class -1", ""}),
    refusal_name);

/** @brief Tests that a row the truth marks as padding is not read. */
TEST(Small_DetectAnchorHeadTruth, LeavesAPaddingRowUnread)
{
    GroundTruthBatch truth = make_truth({56.0f, 16.0f, 72.0f, 48.0f}, K_CLASSES);
    truth.valid[1] = 0;

    SA_EXPECT_OK(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK));
}

/** @brief Tests that a truth of another batch, or one short of box values, is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesArraysThatDoNotHoldTheirShape)
{
    GroundTruthBatch truth = make_truth({16.0f, 16.0f, 48.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 2, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "2 images", "batch of 1");

    truth.boxes.pop_back();
    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "8 box values", "7");
}

/** @brief Tests that a network side below 1 is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesANetworkSideBelowOne)
{
    const GroundTruthBatch truth = make_truth({16.0f, 16.0f, 48.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, NetworkSize{64, 0}),
                      StatusCode::InvalidDescription, "network size", "64 by 0");
}

/** @brief Tests that the delta refuses a truth outside the grid by its row and writes nothing. */
TEST(Small_DetectAnchorHeadTruth, TheDeltaRefusesATruthOutsideTheGrid)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    HeadOptions options;
    options.num_classes = K_CLASSES;
    const HeadShape shape{1, 2, 2};
    const std::vector<float> head(static_cast<std::size_t>(2 * 2 * (5 + K_CLASSES)), 0.0f);
    const GroundTruthBatch truth = make_truth({56.0f, 16.0f, 72.0f, 48.0f}, 0);
    HeadDelta result;

    SA_EXPECT_FAILURE(compute_head_delta(head, shape, options, layout, 0, HeadTrainingOptions{},
                                         truth, K_NETWORK, result),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "centre",
                      "(1, 0.5)");
    EXPECT_TRUE(result.delta.empty());
    EXPECT_TRUE(result.assigned_row.empty());
}

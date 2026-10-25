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
 * @file test_head_truth.cpp
 * @brief Tests which truth rows an anchor head accepts and how it names the ones it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
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
} // namespace

/** @brief Tests that rows inside the input pass, a centre of exactly 0 included. */
TEST(Small_DetectAnchorHeadTruth, AcceptsRowsWhoseCentreLiesInsideTheInput)
{
    // Corners of -8 and 8 give a centre of 0, the closed end of [0, 1).
    const GroundTruthBatch truth = make_truth({-8.0f, -8.0f, 8.0f, 8.0f}, 2);

    SA_EXPECT_OK(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK));
}

/** @brief Tests that a centre of exactly 1 along the columns is refused with its row. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreAtTheRightEdge)
{
    const GroundTruthBatch truth = make_truth({56.0f, 16.0f, 72.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "centre",
                      "(1, 0.5)");
}

/** @brief Tests that a centre left of the input is refused with its value. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreLeftOfTheInput)
{
    const GroundTruthBatch truth = make_truth({-48.0f, 16.0f, 16.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "(-0.25, 0.5)");
}

/** @brief Tests that a centre below the input is refused with its value. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreBelowTheInput)
{
    const GroundTruthBatch truth = make_truth({16.0f, 64.0f, 48.0f, 96.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "(0.5, 1.25)");
}

/** @brief Tests that a centre above the input is refused with its value. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreAboveTheInput)
{
    const GroundTruthBatch truth = make_truth({16.0f, -48.0f, 48.0f, 16.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "(0.5, -0.25)");
}

/** @brief Tests that a centre that is not a number is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesACentreThatIsNotANumber)
{
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    const GroundTruthBatch truth = make_truth({not_a_number, 16.0f, 48.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "centre");
}

/** @brief Tests that a width of 0 is refused with the size it has. */
TEST(Small_DetectAnchorHeadTruth, RefusesAWidthThatIsNotPositive)
{
    const GroundTruthBatch truth = make_truth({32.0f, 16.0f, 32.0f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "size 0 by 0.5");
}

/** @brief Tests that a negative height is refused with the size it has. */
TEST(Small_DetectAnchorHeadTruth, RefusesAHeightThatIsNotPositive)
{
    const GroundTruthBatch truth = make_truth({16.0f, 40.0f, 48.0f, 24.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0",
                      "size 0.5 by -0.25");
}

/** @brief Tests that a width that overflows to infinity around a centre of 0 is refused. */
TEST(Small_DetectAnchorHeadTruth, RefusesASideThatOverflowsToInfinity)
{
    const GroundTruthBatch truth = make_truth({-3.0e38f, 16.0f, 3.0e38f, 48.0f}, 0);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "finite");
}

/** @brief Tests that a class at the head's class count is refused with its id. */
TEST(Small_DetectAnchorHeadTruth, RefusesAClassAtTheClassCount)
{
    const GroundTruthBatch truth = make_truth({16.0f, 16.0f, 48.0f, 48.0f}, K_CLASSES);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "class 3",
                      "3 classes");
}

/** @brief Tests that a negative class is refused with its id. */
TEST(Small_DetectAnchorHeadTruth, RefusesANegativeClass)
{
    const GroundTruthBatch truth = make_truth({16.0f, 16.0f, 48.0f, 48.0f}, -1);

    SA_EXPECT_FAILURE(validate_head_truth(truth, 1, K_CLASSES, K_NETWORK),
                      StatusCode::InvalidDescription, "truth row 1 of image 0", "class -1");
}

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

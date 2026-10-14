/**************************************************************************/
/* test_detect_anchor_delta.cpp                                           */
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
 * @file test_detect_anchor_delta.cpp
 * @brief Tests the training options and the squared-error delta of an anchor head.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::Status;
    using SushiAI::StatusCode;
    using SushiAI::Test::status_is_ok;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::BoxGradient;
    using SushiAI::Vision::Detect::Anchor::compute_head_delta;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::validate_head_training;
    using SushiAI::Vision::Detect::GroundTruthBatch;

    /** @brief Holds the absolute tolerance of every delta, from the unit's acceptance. */
    constexpr float K_TOLERANCE = 1e-6f;

    /** @brief Holds a value no enumeration of the training options names. */
    constexpr std::uint8_t K_UNKNOWN_ENUMERATOR = 200;

    /** @brief Holds a logit whose float32 logistic is exactly 1, and exactly 0 when negated. */
    constexpr float K_SATURATED = 100.0f;

    /** @brief Holds the natural logarithm of 2, the width target of a truth twice its anchor. */
    constexpr double K_LOG_OF_TWO = 0.69314718055994530942;

    /** @brief Holds the side of the square network input, which makes every corner exact. */
    constexpr std::int64_t K_NETWORK_SIDE = 64;

    /** @brief Holds the seed of the logits the empty-truth test draws. */
    constexpr std::uint32_t K_SEED = 20261009u;

    /** @brief Holds the entry of the centre offset along the columns within an anchor slot. */
    constexpr std::size_t K_X = 0;

    /** @brief Holds the entry of the centre offset along the rows within an anchor slot. */
    constexpr std::size_t K_Y = 1;

    /** @brief Holds the entry of the box width within an anchor slot. */
    constexpr std::size_t K_WIDTH = 2;

    /** @brief Holds the entry of the box height within an anchor slot. */
    constexpr std::size_t K_HEIGHT = 3;

    /** @brief Holds the entry of the objectness within an anchor slot. */
    constexpr std::size_t K_OBJECTNESS = 4;

    /** @brief Holds the entry of class 0; class c follows at K_FIRST_CLASS + c. */
    constexpr std::size_t K_FIRST_CLASS = 5;

    /** @brief Holds a head, its truth and what compute_head_delta writes from them. */
    struct DeltaCase
    {
        HeadShape shape;              /**< Extents of the head. */
        HeadOptions options;          /**< Classes, coding and scale. */
        AnchorLayout layout;          /**< Anchors of every head. */
        std::int64_t head_index = 0;  /**< The head the case computes. */
        HeadTrainingOptions training; /**< Weights, thresholds and gates. */
        GroundTruthBatch truth;       /**< Truth rows in pixel corners. */
        std::vector<float> head;      /**< `[B, H, W, A * (5 + C)]`, zero until set. */
        HeadDelta result;             /**< What the call wrote. */

        /** @brief Returns the anchor slots of the computed head. */
        std::size_t slots() const
        {
            return layout.head_anchors[static_cast<std::size_t>(head_index)].size();
        }

        /** @brief Returns the entries of one slot, 5 + C. */
        std::size_t entries() const
        {
            return K_FIRST_CLASS + static_cast<std::size_t>(options.num_classes);
        }

        /** @brief Returns the index of slot @p slot of cell (@p row, @p col) in `[B, H, W, A]`. */
        std::size_t position(std::size_t image, std::size_t row, std::size_t col,
                             std::size_t slot) const
        {
            const std::size_t width = static_cast<std::size_t>(shape.width);
            const std::size_t height = static_cast<std::size_t>(shape.height);
            return ((image * height + row) * width + col) * slots() + slot;
        }

        /** @brief Returns the head entry @p entry of the slot at @p position. */
        float& at(std::size_t position, std::size_t entry)
        {
            return head[position * entries() + entry];
        }

        /** @brief Returns the delta entry @p entry of the slot at @p position. */
        float delta(std::size_t position, std::size_t entry) const
        {
            return result.delta[position * entries() + entry];
        }

        /** @brief Sets every class logit of every slot to @p value. */
        void fill_classes(float value)
        {
            for (std::size_t index = 0; index < head.size(); ++index)
            {
                if (index % entries() >= K_FIRST_CLASS)
                    head[index] = value;
            }
        }

        /** @brief Appends a truth row of pixel corners to every image; rows stay padded. */
        void add_truth(float x1, float y1, float x2, float y2, std::int64_t label)
        {
            const std::size_t images = static_cast<std::size_t>(shape.batch);
            const std::size_t rows = static_cast<std::size_t>(truth.max_boxes);
            GroundTruthBatch grown;
            grown.batch = shape.batch;
            grown.max_boxes = truth.max_boxes + 1;
            for (std::size_t image = 0; image < images; ++image)
            {
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const std::size_t index = image * rows + row;
                    grown.labels.push_back(truth.labels[index]);
                    grown.valid.push_back(truth.valid[index]);
                    for (std::size_t value = 0; value < 4; ++value)
                        grown.boxes.push_back(truth.boxes[index * 4 + value]);
                }
                grown.labels.push_back(label);
                grown.valid.push_back(1);
                grown.boxes.insert(grown.boxes.end(), {x1, y1, x2, y2});
            }
            truth = grown;
        }

        /** @brief Runs compute_head_delta on the case's head and returns its status. */
        [[nodiscard]] Status compute()
        {
            const NetworkSize network{K_NETWORK_SIDE, K_NETWORK_SIDE};
            return compute_head_delta(head, shape, options, layout, head_index, training, truth,
                               network, result);
        }
    };

    /** @brief Returns a case with a zero head, no truth and the default training options. */
    DeltaCase make_case(const HeadShape& shape, std::int64_t classes, const AnchorLayout& layout,
                        std::int64_t head_index)
    {
        DeltaCase made;
        made.shape = shape;
        made.options.num_classes = classes;
        made.layout = layout;
        made.head_index = head_index;
        made.truth.batch = shape.batch;

        const std::size_t positions =
            static_cast<std::size_t>(shape.batch * shape.height * shape.width) * made.slots();
        made.head.assign(positions * made.entries(), 0.0f);
        return made;
    }

    /** @brief Expects every entry of @p actual within K_TOLERANCE of @p expected. */
    void expect_deltas(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], K_TOLERANCE) << "entry " << index;
    }
} // namespace

/** @brief Tests that the default training options pass their check. */
TEST(Unit_DetectAnchorDelta, DefaultTrainingOptionsValidate)
{
    EXPECT_TRUE(validate_head_training(HeadTrainingOptions{}).is_ok());
}

/** @brief Tests that each option outside its range is an invalid description. */
TEST(Unit_DetectAnchorDelta, TrainingOptionsOutsideTheirRangeAreRefused)
{
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    std::vector<HeadTrainingOptions> refused(8);
    refused[0].thresholds.match_threshold = 0.0f;
    refused[1].thresholds.match_threshold = 1.5f;
    refused[2].thresholds.ignore_threshold = -0.1f;
    refused[3].class_gate = 1.5f;
    refused[4].box_weight = -1.0f;
    refused[5].object_weight = not_a_number;
    refused[6].box_weight = std::numeric_limits<float>::infinity();
    refused[7].max_delta = -2.0f;
    refused.emplace_back();
    refused.back().box_gradient = static_cast<BoxGradient>(K_UNKNOWN_ENUMERATOR);

    for (std::size_t index = 0; index < refused.size(); ++index)
    {
        EXPECT_EQ(validate_head_training(refused[index]).code(),
                  StatusCode::InvalidDescription)
            << "options " << index;
    }
}

/** @brief Tests that one truth writes its cell and slot with k = (2 - tw * th) * 0.75. */
TEST(Unit_DetectAnchorDelta, OneTruthWritesTheCellAndSlotOfItsBestAnchor)
{
    AnchorLayout layout;
    layout.anchors = {{0.125f, 0.125f}, {0.25f, 0.5f}, {1.0f, 1.0f}};
    layout.head_anchors = {{0, 1, 2}};
    DeltaCase head = make_case({1, 2, 2}, 2, layout, 0);

    // A class probability of 0 keeps every slot out of the ignore rule.
    head.fill_classes(-K_SATURATED);
    // Centre (0.75, 0.25) and size (0.5, 0.5): cell row 0, column 1, offsets 0.5 and 0.5.
    head.add_truth(32.0f, 0.0f, 64.0f, 32.0f, 1);
    // Shape IoUs with (0.5, 0.5) are 0.0625, 0.5 and 0.25, so slot 1 is assigned.
    const std::size_t assigned = head.position(0, 0, 1, 1);
    head.at(assigned, K_X) = K_SATURATED;
    head.at(assigned, K_Y) = -K_SATURATED;
    head.at(assigned, K_HEIGHT) = 0.5f;
    head.at(assigned, K_FIRST_CLASS) = 0.0f;
    head.at(assigned, K_FIRST_CLASS + 1) = 0.0f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double k = (2.0 - 0.5 * 0.5) * 0.75;
    std::vector<float> expected(head.head.size(), 0.0f);
    for (std::size_t position = 0; position < 12; ++position)
        expected[position * head.entries() + K_OBJECTNESS] = -0.5f;
    float* slot = expected.data() + assigned * head.entries();
    slot[K_X] = static_cast<float>(k * (0.5 - 1.0));
    slot[K_Y] = static_cast<float>(k * (0.5 - 0.0));
    slot[K_WIDTH] = static_cast<float>(k * (K_LOG_OF_TWO - 0.0));
    slot[K_HEIGHT] = static_cast<float>(k * (0.0 - 0.5));
    slot[K_OBJECTNESS] = 0.5f;
    slot[K_FIRST_CLASS] = -0.5f;
    slot[K_FIRST_CLASS + 1] = 0.5f;
    expect_deltas(head.result.delta, expected);

    EXPECT_EQ(head.result.assignments, 1);
    for (std::size_t position = 0; position < 12; ++position)
    {
        EXPECT_EQ(head.result.assigned_row[position], position == assigned ? 0 : -1);
        EXPECT_EQ(head.result.assigned_class[position], position == assigned ? 1 : -1);
    }
}

/** @brief Tests that the IoU sum takes the IoU of the assigned slot's box with its truth. */
TEST(Unit_DetectAnchorDelta, AnAssignedSlotThatCoversItsTruthAddsOneToTheIoUSum)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    DeltaCase head = make_case({1, 1, 1}, 1, layout, 0);
    // Zero logits decode to the cell centre and the anchor, which is the truth's box.
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.result.assignments, 1);
    EXPECT_NEAR(head.result.iou_sum, 1.0, 1e-6);
}

/** @brief Tests that a truth whose best anchor another head owns assigns nothing. */
TEST(Unit_DetectAnchorDelta, ATruthWhoseBestAnchorAnotherHeadOwnsAssignsNothing)
{
    AnchorLayout layout;
    layout.anchors = {{0.125f, 0.125f}, {0.5f, 0.5f}};
    layout.head_anchors = {{0}, {1}};
    DeltaCase head = make_case({1, 2, 2}, 2, layout, 0);
    head.fill_classes(-K_SATURATED);
    // The truth has the size of anchor 1, which head 1 owns.
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 1);

    ASSERT_TRUE(status_is_ok(head.compute()));

    std::vector<float> expected(head.head.size(), 0.0f);
    for (std::size_t position = 0; position < 4; ++position)
        expected[position * head.entries() + K_OBJECTNESS] = -0.5f;
    expect_deltas(head.result.delta, expected);

    EXPECT_EQ(head.result.assignments, 0);
    EXPECT_EQ(head.result.iou_sum, 0.0);
    for (std::size_t position = 0; position < 4; ++position)
    {
        EXPECT_EQ(head.result.assigned_row[position], -1);
        EXPECT_EQ(head.result.assigned_class[position], -1);
    }
}

/** @brief Tests that the ignore rule needs a class probability above 0.25. */
TEST(Unit_DetectAnchorDelta, AnOverlappingSlotIsIgnoredOnlyWithAClassAboveTheGate)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}, {0.45f, 0.45f}};
    layout.head_anchors = {{1}, {0}};
    DeltaCase head = make_case({2, 1, 1}, 2, layout, 0);
    // Head 1 owns the truth's best anchor, so head 0 assigns nothing.
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    // The slot decodes to a 0.45 square on the truth's centre: IoU 0.2025 / 0.25 = 0.81.
    head.fill_classes(-K_SATURATED);
    // logistic(-1.0) is 0.2689, above the gate; logistic(-1.2) is 0.2315, below it.
    head.at(head.position(0, 0, 0, 0), K_FIRST_CLASS + 1) = -1.0f;
    head.at(head.position(1, 0, 0, 0), K_FIRST_CLASS + 1) = -1.2f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.result.assignments, 0);
    EXPECT_EQ(head.delta(head.position(0, 0, 0, 0), K_OBJECTNESS), 0.0f);
    EXPECT_NEAR(head.delta(head.position(1, 0, 0, 0), K_OBJECTNESS), -0.5f, K_TOLERANCE);
}

/** @brief Tests that an overlap at or below the ignore threshold keeps its objectness delta. */
TEST(Unit_DetectAnchorDelta, AnOverlapBelowTheIgnoreThresholdIsNotIgnored)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}, {0.45f, 0.45f}};
    layout.head_anchors = {{1}, {0}};
    DeltaCase head = make_case({1, 1, 1}, 2, layout, 0);
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    // Class probabilities of 0.5 pass the gate; the IoU of 0.81 stays under 0.9.
    head.training.thresholds.ignore_threshold = 0.9f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_NEAR(head.delta(0, K_OBJECTNESS), -0.5f, K_TOLERANCE);
}

/** @brief Tests that a second truth in a slot adds its box delta and updates one class. */
TEST(Unit_DetectAnchorDelta, TwoTruthsInOneSlotUpdateOneClassOnly)
{
    AnchorLayout layout;
    layout.anchors = {{0.25f, 0.25f}};
    layout.head_anchors = {{0}};
    DeltaCase head = make_case({1, 1, 1}, 3, layout, 0);
    head.at(0, K_X) = -K_SATURATED;
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 2);

    ASSERT_TRUE(status_is_ok(head.compute()));

    // The second truth rewrites class 2 alone; a full update would turn class 0 to -0.5.
    const double k = (2.0 - 0.5 * 0.5) * 0.75;
    std::vector<float> expected(head.head.size(), 0.0f);
    expected[K_X] = static_cast<float>(2.0 * k * (0.5 - 0.0));
    expected[K_Y] = 0.0f;
    expected[K_WIDTH] = static_cast<float>(2.0 * k * K_LOG_OF_TWO);
    expected[K_HEIGHT] = static_cast<float>(2.0 * k * K_LOG_OF_TWO);
    expected[K_OBJECTNESS] = 0.5f;
    expected[K_FIRST_CLASS] = 0.5f;
    expected[K_FIRST_CLASS + 1] = -0.5f;
    expected[K_FIRST_CLASS + 2] = 0.5f;
    expect_deltas(head.result.delta, expected);
    EXPECT_EQ(head.result.assignments, 2);
}

/** @brief Tests that without truths only the objectness entries hold a delta. */
TEST(Unit_DetectAnchorDelta, AnEmptyTruthListGivesObjectnessDeltasAlone)
{
    AnchorLayout layout;
    layout.anchors = {{0.125f, 0.125f}, {0.25f, 0.5f}, {1.0f, 1.0f}};
    layout.head_anchors = {{0, 1, 2}};
    DeltaCase head = make_case({2, 2, 2}, 2, layout, 0);
    head.training.object_weight = 2.0f;

    std::mt19937 generator(K_SEED);
    std::uniform_real_distribution<float> logits(-3.0f, 3.0f);
    for (float& value : head.head)
        value = logits(generator);

    ASSERT_TRUE(status_is_ok(head.compute()));

    std::vector<float> expected(head.head.size(), 0.0f);
    for (std::size_t index = K_OBJECTNESS; index < expected.size(); index += head.entries())
    {
        const double objectness = 1.0 / (1.0 + std::exp(-static_cast<double>(head.head[index])));
        expected[index] = static_cast<float>(2.0 * (0.0 - objectness));
    }
    expect_deltas(head.result.delta, expected);
    EXPECT_EQ(head.result.assignments, 0);
    for (const std::int64_t row : head.result.assigned_row)
        EXPECT_EQ(row, -1);
}

/** @brief Tests that a padding row of the truth batch is read as no truth. */
TEST(Unit_DetectAnchorDelta, APaddingRowAssignsNothing)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    DeltaCase head = make_case({1, 1, 1}, 1, layout, 0);
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    head.truth.valid[0] = 0;

    ASSERT_TRUE(status_is_ok(head.compute()));

    // Zero logits give an objectness of 0.5; the slot would be ignored if the row counted.
    EXPECT_EQ(head.result.assignments, 0);
    EXPECT_NEAR(head.delta(0, K_OBJECTNESS), -0.5f, K_TOLERANCE);
    EXPECT_EQ(head.delta(0, K_FIRST_CLASS), 0.0f);
}

/**************************************************************************/
/* test_head_delta.cpp                                                    */
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
 * @file test_head_delta.cpp
 * @brief Tests the training delta of an anchor head: squared error, the Activated coding, the
 *        complete IoU and its gradient.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "anchor_delta_support.hpp"
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>

namespace
{
    using AnchorDeltaSupport::expect_box_delta;
    using AnchorDeltaSupport::K_BOX_ENTRIES;
    using AnchorDeltaSupport::Quad;
    using SushiAI::Status;
    using SushiAI::Test::status_is_ok;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::BoxLoss;
    using SushiAI::Vision::Detect::Anchor::compute_head_delta;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::GroundTruthBatch;

    /** @brief Holds the absolute tolerance of an objectness or class delta. */
    constexpr double K_TOLERANCE = 1e-6;

    /** @brief Holds the absolute tolerance of a box delta against a central difference. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-5;

    /** @brief Holds the tolerance of a box delta relative to a central difference. */
    constexpr double K_RELATIVE_TOLERANCE = 1e-5;

    /** @brief Holds a logit whose float32 logistic is exactly 1, and exactly 0 when negated. */
    constexpr float K_SATURATED = 100.0f;

    /** @brief Holds a size logit whose float32 exponential is infinite. */
    constexpr float K_OVERFLOWING = 200.0f;

    /** @brief Holds the side of the square network input, which makes every corner exact. */
    constexpr std::int64_t K_NETWORK_SIDE = 64;

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

    /** @brief Returns the logistic of @p value in double. */
    double logistic(double value)
    {
        return 1.0 / (1.0 + std::exp(-value));
    }

    /** @brief Returns the box of pixel corners as centre and size over the network side. */
    Quad box_of_corners(double x1, double y1, double x2, double y2)
    {
        const double side = static_cast<double>(K_NETWORK_SIDE);
        return {(x1 + x2) / 2.0 / side, (y1 + y2) / 2.0 / side, (x2 - x1) / side,
                (y2 - y1) / side};
    }

    /**
     * @brief Returns the box delta of the Logit coding before the bound and the division.
     * @param raw The slot's four box logits; the last two scale the size components.
     * @return The central difference of the complete IoU by @p box, chained and weighted.
     */
    Quad expect_logit_delta(const Quad& box, const Quad& target, const Quad& raw, double weight)
    {
        Quad delta = expect_box_delta(box, target, weight);
        delta[K_WIDTH] *= std::exp(raw[K_WIDTH]);
        delta[K_HEIGHT] *= std::exp(raw[K_HEIGHT]);
        return delta;
    }

    /** @brief Holds a one-image head, its truth and what compute_head_delta writes from them. */
    struct IoUCase
    {
        HeadShape shape;              /**< Extents of the head. */
        HeadOptions options;          /**< Classes, coding and scale. */
        AnchorLayout layout;          /**< Anchors of every head. */
        std::int64_t head_index = 0;  /**< The head the case computes. */
        HeadTrainingOptions training; /**< Weights, thresholds and gates. */
        GroundTruthBatch truth;       /**< Truth rows in pixel corners. */
        std::vector<float> head;      /**< `[1, H, W, A * (5 + C)]`, zero until set. */
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

        /** @brief Returns the index of slot @p slot of cell (@p row, @p col) in `[H, W, A]`. */
        std::size_t position(std::size_t row, std::size_t col, std::size_t slot) const
        {
            return (row * static_cast<std::size_t>(shape.width) + col) * slots() + slot;
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

        /** @brief Sets the four box logits of the slot at @p position. */
        void set_box(std::size_t position, const Quad& raw)
        {
            for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                at(position, entry) = static_cast<float>(raw[entry]);
        }

        /** @brief Returns the box logits of the slot at @p position as the head holds them. */
        Quad raw_box(std::size_t position) const
        {
            Quad raw{};
            for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                raw[entry] = static_cast<double>(head[position * entries() + entry]);
            return raw;
        }

        /** @brief Returns the Logit decode of slot @p slot of cell (@p row, @p col). */
        Quad decode(std::size_t row, std::size_t col, std::size_t slot) const
        {
            const Quad raw = raw_box(position(row, col, slot));
            const std::size_t anchor = static_cast<std::size_t>(
                layout.head_anchors[static_cast<std::size_t>(head_index)][slot]);
            const double scale = static_cast<double>(options.scale_xy);
            const double shift = (scale - 1.0) / 2.0;
            return {(static_cast<double>(col) + logistic(raw[K_X]) * scale - shift) /
                        static_cast<double>(shape.width),
                    (static_cast<double>(row) + logistic(raw[K_Y]) * scale - shift) /
                        static_cast<double>(shape.height),
                    std::exp(raw[K_WIDTH]) * static_cast<double>(layout.anchors[anchor].width),
                    std::exp(raw[K_HEIGHT]) * static_cast<double>(layout.anchors[anchor].height)};
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

        /** @brief Appends a truth row of pixel corners. */
        void add_truth(float x1, float y1, float x2, float y2, std::int64_t label)
        {
            truth.max_boxes += 1;
            truth.labels.push_back(label);
            truth.valid.push_back(1);
            truth.boxes.insert(truth.boxes.end(), {x1, y1, x2, y2});
        }

        /** @brief Runs compute_head_delta on the case's head and returns its status. */
        [[nodiscard]] Status compute()
        {
            const NetworkSize network{K_NETWORK_SIDE, K_NETWORK_SIDE};
            return compute_head_delta(head, shape, options, layout, head_index, training, truth,
                               network, result);
        }
    };

    /** @brief Returns a case of complete IoU boxes with a zero head and no truth. */
    IoUCase make_case(std::int64_t height, std::int64_t width, std::int64_t classes,
                      const AnchorLayout& layout, std::int64_t head_index)
    {
        IoUCase made;
        made.shape = {1, height, width};
        made.options.num_classes = classes;
        made.layout = layout;
        made.head_index = head_index;
        made.training.box_loss = BoxLoss::CompleteIoU;
        made.truth.batch = 1;

        const std::size_t positions = static_cast<std::size_t>(height * width) * made.slots();
        made.head.assign(positions * made.entries(), 0.0f);
        return made;
    }

    /** @brief Expects the four box deltas of the slot at @p position to equal @p expected. */
    void expect_box(const IoUCase& head, std::size_t position, const Quad& expected)
    {
        for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
        {
            const double bound =
                K_ABSOLUTE_TOLERANCE + K_RELATIVE_TOLERANCE * std::abs(expected[entry]);
            EXPECT_NEAR(static_cast<double>(head.delta(position, entry)), expected[entry], bound)
                << "slot " << position << ", entry " << entry;
        }
    }

    /** @brief Returns @p delta with every component divided by @p share. */
    Quad divide(const Quad& delta, double share)
    {
        return {delta[0] / share, delta[1] / share, delta[2] / share, delta[3] / share};
    }

    /** @brief Returns the sum of two deltas, component by component. */
    Quad add(const Quad& a, const Quad& b)
    {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
    }

    /** @brief Returns a 2x2 case of three anchors whose first two match a (0.5, 0.5) truth. */
    IoUCase make_two_match_case()
    {
        AnchorLayout layout;
        // Shape IoUs with (0.5, 0.5) are 1, 0.2 / 0.29 and 0.04.
        layout.anchors = {{0.5f, 0.5f}, {0.4f, 0.6f}, {0.1f, 0.1f}};
        layout.head_anchors = {{0, 1, 2}};
        IoUCase head = make_case(2, 2, 2, layout, 0);

        // A class probability of 0 keeps every slot out of the ignore rule.
        head.fill_classes(-K_SATURATED);
        // Centre (0.75, 0.25) and size (0.5, 0.5): cell row 0, column 1.
        head.add_truth(32.0f, 0.0f, 64.0f, 32.0f, 1);
        head.set_box(head.position(0, 1, 0), {0.3, -0.2, 0.1, -0.15});
        head.set_box(head.position(0, 1, 1), {-0.4, 0.25, -0.2, 0.1});
        return head;
    }

    /** @brief Returns a 1x1 case of one anchor with a truth of class 0 and one of class 1. */
    IoUCase make_two_truth_case()
    {
        AnchorLayout layout;
        layout.anchors = {{0.5f, 0.5f}};
        layout.head_anchors = {{0}};
        IoUCase head = make_case(1, 1, 2, layout, 0);
        head.add_truth(8.0f, 12.0f, 44.0f, 40.0f, 0);
        head.add_truth(20.0f, 4.0f, 60.0f, 30.0f, 1);
        head.set_box(0, {0.2, -0.3, 0.15, -0.1});
        return head;
    }
} // namespace

/** @brief Tests that two anchors above the match threshold are both assigned and written. */
TEST(Unit_DetectAnchorDeltaIou, TwoAnchorsAboveTheMatchThresholdAreBothAssigned)
{
    IoUCase head = make_two_match_case();
    head.training.thresholds.match_threshold = 0.5f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const Quad truth = box_of_corners(32.0, 0.0, 64.0, 32.0);
    const std::size_t best = head.position(0, 1, 0);
    const std::size_t further = head.position(0, 1, 1);
    // Each slot has one positive class delta, so the division of step 3 is by 1.
    expect_box(head, best,
               expect_logit_delta(head.decode(0, 1, 0), truth, head.raw_box(best), 0.75));
    expect_box(head, further,
               expect_logit_delta(head.decode(0, 1, 1), truth, head.raw_box(further), 0.75));

    EXPECT_EQ(head.result.assignments, 2);
    for (std::size_t position = 0; position < 12; ++position)
    {
        const bool assigned = position == best || position == further;
        EXPECT_NEAR(head.delta(position, K_OBJECTNESS), assigned ? 0.5 : -0.5, K_TOLERANCE);
        EXPECT_NEAR(head.delta(position, K_FIRST_CLASS), 0.0, K_TOLERANCE);
        EXPECT_NEAR(head.delta(position, K_FIRST_CLASS + 1), assigned ? 1.0 : 0.0, K_TOLERANCE);
        if (!assigned)
            expect_box(head, position, Quad{});

        // Only the slot of the best anchor records its truth.
        EXPECT_EQ(head.result.assigned_row[position], position == best ? 0 : -1);
        EXPECT_EQ(head.result.assigned_class[position], position == best ? 1 : -1);
    }
}

/** @brief Tests that the IoU sum takes the IoU of every assigned slot, further ones included. */
TEST(Unit_DetectAnchorDeltaIou, AFurtherAssignmentAddsItsIoUToTheSum)
{
    IoUCase head = make_two_match_case();
    head.training.thresholds.match_threshold = 0.5f;
    // Zero logits decode slot 0 to the truth's own box and slot 1 to its anchor there.
    head.set_box(head.position(0, 1, 0), Quad{});
    head.set_box(head.position(0, 1, 1), Quad{});

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.result.assignments, 2);
    EXPECT_NEAR(head.result.iou_sum, 1.0 + 0.2 / 0.29, 1e-6);
}

/** @brief Tests that a match threshold of 1 assigns the best anchor alone. */
TEST(Unit_DetectAnchorDeltaIou, AMatchThresholdOfOneAssignsTheBestAnchorAlone)
{
    IoUCase head = make_two_match_case();

    ASSERT_TRUE(status_is_ok(head.compute()));

    const Quad truth = box_of_corners(32.0, 0.0, 64.0, 32.0);
    const std::size_t best = head.position(0, 1, 0);
    const std::size_t further = head.position(0, 1, 1);
    expect_box(head, best,
               expect_logit_delta(head.decode(0, 1, 0), truth, head.raw_box(best), 0.75));
    expect_box(head, further, Quad{});
    EXPECT_NEAR(head.delta(further, K_OBJECTNESS), -0.5, K_TOLERANCE);
    EXPECT_EQ(head.result.assignments, 1);
}

/** @brief Tests that an anchor at or below the match threshold takes no truth. */
TEST(Unit_DetectAnchorDeltaIou, AnAnchorBelowTheMatchThresholdIsNotAssigned)
{
    IoUCase head = make_two_match_case();
    // The second anchor's shape IoU is 0.2 / 0.29, about 0.69.
    head.training.thresholds.match_threshold = 0.75f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.result.assignments, 1);
    expect_box(head, head.position(0, 1, 1), Quad{});
}

/** @brief Tests that a head takes a truth above the threshold whose best anchor is not its own. */
TEST(Unit_DetectAnchorDeltaIou, AHeadWithoutTheBestAnchorStillTakesAMatchingTruth)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}, {0.4f, 0.6f}};
    layout.head_anchors = {{0}, {1}};
    IoUCase head = make_case(1, 1, 1, layout, 1);
    head.training.thresholds.match_threshold = 0.5f;
    head.fill_classes(-K_SATURATED);
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    head.set_box(0, {0.3, -0.2, 0.1, -0.15});

    ASSERT_TRUE(status_is_ok(head.compute()));

    const Quad truth = box_of_corners(16.0, 16.0, 48.0, 48.0);
    expect_box(head, 0, expect_logit_delta(head.decode(0, 0, 0), truth, head.raw_box(0), 0.75));
    EXPECT_NEAR(head.delta(0, K_OBJECTNESS), 0.5, K_TOLERANCE);
    EXPECT_NEAR(head.delta(0, K_FIRST_CLASS), 1.0, K_TOLERANCE);
    EXPECT_EQ(head.result.assignments, 1);
    EXPECT_EQ(head.result.assigned_row[0], -1);
    EXPECT_EQ(head.result.assigned_class[0], -1);
}

/** @brief Tests that a slot with two positive class deltas halves its summed box deltas. */
TEST(Unit_DetectAnchorDeltaIou, TwoPositiveClassDeltasHalveTheBoxDeltasOfTheSlot)
{
    IoUCase head = make_two_truth_case();
    head.training.thresholds.match_threshold = 0.5f;
    head.training.box_weight = 0.07f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double weight = static_cast<double>(0.07f);
    const Quad box = head.decode(0, 0, 0);
    const Quad first =
        expect_logit_delta(box, box_of_corners(8.0, 12.0, 44.0, 40.0), head.raw_box(0), weight);
    const Quad second =
        expect_logit_delta(box, box_of_corners(20.0, 4.0, 60.0, 30.0), head.raw_box(0), weight);
    expect_box(head, 0, divide(add(first, second), 2.0));

    // The second truth rewrites its own class alone, which leaves both deltas positive.
    EXPECT_NEAR(head.delta(0, K_FIRST_CLASS), 0.5, K_TOLERANCE);
    EXPECT_NEAR(head.delta(0, K_FIRST_CLASS + 1), 0.5, K_TOLERANCE);
    EXPECT_EQ(head.result.assignments, 2);
}

/** @brief Tests that a match threshold of 1 leaves the summed box deltas undivided. */
TEST(Unit_DetectAnchorDeltaIou, AMatchThresholdOfOneDividesNothing)
{
    IoUCase head = make_two_truth_case();

    ASSERT_TRUE(status_is_ok(head.compute()));

    const Quad box = head.decode(0, 0, 0);
    const Quad first =
        expect_logit_delta(box, box_of_corners(8.0, 12.0, 44.0, 40.0), head.raw_box(0), 0.75);
    const Quad second =
        expect_logit_delta(box, box_of_corners(20.0, 4.0, 60.0, 30.0), head.raw_box(0), 0.75);
    expect_box(head, 0, add(first, second));
}

/** @brief Tests that the division below a match threshold of 1 covers squared-error boxes. */
TEST(Unit_DetectAnchorDeltaIou, TheDivisionCoversSquaredErrorBoxes)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    IoUCase head = make_case(1, 1, 2, layout, 0);
    head.training.box_loss = BoxLoss::SquaredError;
    head.training.thresholds.match_threshold = 0.5f;
    // Both truths are the anchor at the cell centre, so only the width entry is off target.
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 0);
    head.add_truth(16.0f, 16.0f, 48.0f, 48.0f, 1);
    head.at(0, K_WIDTH) = 0.5f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double k = (2.0 - 0.5 * 0.5) * 0.75;
    expect_box(head, 0, divide({0.0, 0.0, 2.0 * k * (0.0 - 0.5), 0.0}, 2.0));
}

/** @brief Tests that a box delta beyond the bound is the bound, with its sign. */
TEST(Unit_DetectAnchorDeltaIou, ABoxDeltaBeyondTheBoundIsClipped)
{
    IoUCase head = make_two_match_case();
    head.training.max_delta = 0.5f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double bound = static_cast<double>(0.5f);
    const std::size_t best = head.position(0, 1, 0);
    Quad expected = expect_logit_delta(head.decode(0, 1, 0), box_of_corners(32.0, 0.0, 64.0, 32.0),
                                     head.raw_box(best), 0.75);
    std::size_t beyond = 0;
    for (double& value : expected)
    {
        if (std::abs(value) > bound)
        {
            value = value > 0.0 ? bound : -bound;
            ++beyond;
        }
    }
    // The case has to hold a component on each side of the bound to show the clip.
    ASSERT_GT(beyond, 0u);
    ASSERT_LT(beyond, K_BOX_ENTRIES);
    expect_box(head, best, expected);
}

/** @brief Tests that a box component that is not finite adds nothing to its entry. */
TEST(Unit_DetectAnchorDeltaIou, ANonFiniteBoxComponentAddsZero)
{
    IoUCase head = make_two_match_case();
    const std::size_t best = head.position(0, 1, 0);
    // The width decodes to infinity, and any width gradient times it is not finite.
    head.at(best, K_WIDTH) = K_OVERFLOWING;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.delta(best, K_WIDTH), 0.0f);
    for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
        EXPECT_TRUE(std::isfinite(head.delta(best, entry))) << "entry " << entry;
    EXPECT_NEAR(head.delta(best, K_OBJECTNESS), 0.5, K_TOLERANCE);
}

namespace
{
    /** @brief Holds the absolute tolerance of every delta, from the unit's acceptance. */
    constexpr float K_DELTA_TOLERANCE = 1e-6f;

    /** @brief Holds the natural logarithm of 2, the width target of a truth twice its anchor. */
    constexpr double K_LOG_OF_TWO = 0.69314718055994530942;

    /** @brief Holds the seed of the logits the empty-truth test draws. */
    constexpr std::uint32_t K_SEED = 20261009u;

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

    /** @brief Expects every entry of @p actual within K_DELTA_TOLERANCE of @p expected. */
    void expect_deltas(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], K_DELTA_TOLERANCE) << "entry " << index;
    }

} // namespace

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
    EXPECT_NEAR(head.delta(head.position(1, 0, 0, 0), K_OBJECTNESS), -0.5f, K_DELTA_TOLERANCE);
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

    EXPECT_NEAR(head.delta(0, K_OBJECTNESS), -0.5f, K_DELTA_TOLERANCE);
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
    EXPECT_NEAR(head.delta(0, K_OBJECTNESS), -0.5f, K_DELTA_TOLERANCE);
    EXPECT_EQ(head.delta(0, K_FIRST_CLASS), 0.0f);
}

namespace
{
    using SushiAI::Vision::Detect::Anchor::HeadCoding;
    using SushiAI::Vision::Detect::Anchor::ObjectnessTarget;

    /** @brief Holds the absolute tolerance of a box delta against a central difference. */
    constexpr double K_BOX_ABSOLUTE_TOLERANCE = 1e-5;

    /** @brief Holds the tolerance of a box delta relative to a central difference. */
    constexpr double K_BOX_RELATIVE_TOLERANCE = 1e-5;

    /** @brief Holds the entries of a slot of the two-class cases. */
    constexpr std::size_t K_ENTRIES = K_FIRST_CLASS + 2;

    /** @brief Holds the slot of the anchor that is the truth's best. */
    constexpr std::size_t K_BEST_SLOT = 0;

    /** @brief Holds the slot whose box overlaps the truth with an IoU of 0.8. */
    constexpr std::size_t K_NEAR_SLOT = 1;

    /** @brief Holds the objectness both slots of a case publish. */
    constexpr float K_PUBLISHED_OBJECTNESS = 0.3f;

    /** @brief Holds the IoU of the near slot with the truth: 0.25 shared over 0.3125 joined. */
    constexpr double K_NEAR_IOU = 0.25 / 0.3125;

    /** @brief Holds a 1x1 head of two slots and two classes under the Activated coding. */
    struct ActivatedCase
    {
        HeadOptions options;          /**< Classes, coding and scale. */
        AnchorLayout layout;          /**< The two anchors of the one head. */
        HeadTrainingOptions training; /**< Weights, thresholds and gates. */
        GroundTruthBatch truth;       /**< Truth rows in pixel corners. */
        std::vector<float> head;      /**< `[1, 1, 1, 2 * 7]`, values in [0, 1]. */
        HeadDelta result;             /**< What the call wrote. */

        /** @brief Returns the head entry @p entry of slot @p slot. */
        float& at(std::size_t slot, std::size_t entry)
        {
            return head[slot * K_ENTRIES + entry];
        }

        /** @brief Returns the delta entry @p entry of slot @p slot. */
        double delta(std::size_t slot, std::size_t entry) const
        {
            return static_cast<double>(result.delta[slot * K_ENTRIES + entry]);
        }

        /** @brief Sets the four box entries of slot @p slot. */
        void set_box(std::size_t slot, const std::array<float, K_BOX_ENTRIES>& entries)
        {
            for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                at(slot, entry) = entries[entry];
        }

        /** @brief Sets both class entries of slot @p slot. */
        void set_classes(std::size_t slot, float first, float second)
        {
            at(slot, K_FIRST_CLASS) = first;
            at(slot, K_FIRST_CLASS + 1) = second;
        }

        /** @brief Returns the Activated decode of slot @p slot of the one cell, in double. */
        Quad decode(std::size_t slot) const
        {
            const float* entries = head.data() + slot * K_ENTRIES;
            const std::size_t anchor = static_cast<std::size_t>(layout.head_anchors[0][slot]);
            const double scale = static_cast<double>(options.scale_xy);
            const double shift = (scale - 1.0) / 2.0;
            const double side_x = static_cast<double>(entries[K_WIDTH]);
            const double side_y = static_cast<double>(entries[K_HEIGHT]);
            return {static_cast<double>(entries[0]) * scale - shift,
                    static_cast<double>(entries[1]) * scale - shift,
                    side_x * side_x * 4.0 * static_cast<double>(layout.anchors[anchor].width),
                    side_y * side_y * 4.0 * static_cast<double>(layout.anchors[anchor].height)};
        }

        /** @brief Runs compute_head_delta on the case's head and returns its status. */
        [[nodiscard]] Status compute()
        {
            const HeadShape shape{1, 1, 1};
            const NetworkSize network{K_NETWORK_SIDE, K_NETWORK_SIDE};
            return compute_head_delta(head, shape, options, layout, 0, training, truth, network,
                               result);
        }
    };

    /** @brief Returns the truth of every case as centre and size. */
    Quad truth_box()
    {
        return {0.5, 0.5, 0.5, 0.625};
    }

    /**
     * @brief Returns a case whose truth of class 0 takes slot 0 and overlaps slot 1 by 0.8.
     * @return Slot 0 decodes to a box of little overlap; slot 1 to (0.5, 0.5, 0.5, 0.5), whose
     *         IoU with the truth is K_NEAR_IOU. Both publish an objectness of 0.3 and a class
     *         probability of 0.6, above the class gate.
     */
    ActivatedCase make_case()
    {
        ActivatedCase made;
        made.options.num_classes = 2;
        made.options.coding = HeadCoding::Activated;
        made.layout.anchors = {{0.5f, 0.625f}, {0.5f, 0.5f}};
        made.layout.head_anchors = {{0, 1}};
        made.training.box_loss = BoxLoss::CompleteIoU;
        made.training.objectness_target = ObjectnessTarget::BestIoU;

        made.truth.batch = 1;
        made.truth.max_boxes = 1;
        made.truth.labels = {0};
        made.truth.valid = {1};
        made.truth.boxes = {16.0f, 12.0f, 48.0f, 52.0f};

        made.head.assign(2 * K_ENTRIES, 0.0f);
        made.set_box(K_BEST_SLOT, {0.5f, 0.5f, 0.1f, 0.1f});
        made.set_box(K_NEAR_SLOT, {0.5f, 0.5f, 0.5f, 0.5f});
        for (const std::size_t slot : {K_BEST_SLOT, K_NEAR_SLOT})
        {
            made.at(slot, K_OBJECTNESS) = K_PUBLISHED_OBJECTNESS;
            made.set_classes(slot, 0.6f, 0.2f);
        }
        return made;
    }
} // namespace

/** @brief Tests that an ignored slot of IoU 0.8 and objectness 0.3 gets a delta of 0.5. */
TEST(Unit_DetectAnchorDeltaActivated, AnIgnoredSlotIsPulledTowardItsBestIoU)
{
    ActivatedCase head = make_case();

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double objectness = static_cast<double>(K_PUBLISHED_OBJECTNESS);
    EXPECT_NEAR(head.delta(K_NEAR_SLOT, K_OBJECTNESS), 1.0 * (K_NEAR_IOU - objectness),
                K_TOLERANCE);
    EXPECT_EQ(head.result.assigned_row[K_NEAR_SLOT], -1);
}

/** @brief Tests that the object weight multiplies the delta of an ignored slot. */
TEST(Unit_DetectAnchorDeltaActivated, TheObjectWeightScalesAnIgnoredSlot)
{
    ActivatedCase head = make_case();
    head.training.object_weight = 0.5f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double objectness = static_cast<double>(K_PUBLISHED_OBJECTNESS);
    EXPECT_NEAR(head.delta(K_NEAR_SLOT, K_OBJECTNESS), 0.5 * (K_NEAR_IOU - objectness),
                K_TOLERANCE);
}

/** @brief Tests that the target One gives an ignored slot no objectness delta. */
TEST(Unit_DetectAnchorDeltaActivated, TheTargetOneLeavesAnIgnoredSlotAlone)
{
    ActivatedCase head = make_case();
    head.training.objectness_target = ObjectnessTarget::One;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_EQ(head.delta(K_NEAR_SLOT, K_OBJECTNESS), 0.0);
}

/** @brief Tests that a slot with no class above the gate is pulled toward 0, not its IoU. */
TEST(Unit_DetectAnchorDeltaActivated, ASlotWithoutAGatedClassIsNotIgnored)
{
    ActivatedCase head = make_case();
    head.set_classes(K_NEAR_SLOT, 0.2f, 0.2f);

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double objectness = static_cast<double>(K_PUBLISHED_OBJECTNESS);
    EXPECT_NEAR(head.delta(K_NEAR_SLOT, K_OBJECTNESS), 0.0 - objectness, K_TOLERANCE);
}

/** @brief Tests that under BestIoU an assigned slot keeps a delta that is not 0. */
TEST(Unit_DetectAnchorDeltaActivated, AnAssignedSlotKeepsADeltaThatIsNotZero)
{
    ActivatedCase head = make_case();

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double objectness = static_cast<double>(K_PUBLISHED_OBJECTNESS);
    EXPECT_EQ(head.result.assigned_row[K_BEST_SLOT], 0);
    EXPECT_NEAR(head.delta(K_BEST_SLOT, K_OBJECTNESS), 0.0 - objectness, K_TOLERANCE);
}

/** @brief Tests that under BestIoU an assigned slot whose delta is 0 is pulled toward 1. */
TEST(Unit_DetectAnchorDeltaActivated, AnAssignedSlotWithAZeroDeltaIsPulledTowardOne)
{
    ActivatedCase head = make_case();
    // An objectness of exactly 0 leaves the delta toward 0 at 0.
    head.at(K_BEST_SLOT, K_OBJECTNESS) = 0.0f;

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_NEAR(head.delta(K_BEST_SLOT, K_OBJECTNESS), 1.0, K_TOLERANCE);
}

/** @brief Tests that the target One pulls an assigned slot toward 1 whatever it held. */
TEST(Unit_DetectAnchorDeltaActivated, TheTargetOnePullsAnAssignedSlotTowardOne)
{
    ActivatedCase head = make_case();
    head.training.objectness_target = ObjectnessTarget::One;

    ASSERT_TRUE(status_is_ok(head.compute()));

    const double objectness = static_cast<double>(K_PUBLISHED_OBJECTNESS);
    EXPECT_NEAR(head.delta(K_BEST_SLOT, K_OBJECTNESS), 1.0 - objectness, K_TOLERANCE);
}

/** @brief Tests that the box delta is the weighted gradient at the decoded box, unchained. */
TEST(Unit_DetectAnchorDeltaActivated, TheBoxDeltaIsTheWeightedGradientWithNoSizeFactor)
{
    ActivatedCase head = make_case();
    head.options.scale_xy = 2.0f;
    head.set_box(K_BEST_SLOT, {0.4f, 0.55f, 0.6f, 0.45f});

    ASSERT_TRUE(status_is_ok(head.compute()));

    const Quad expected = expect_box_delta(head.decode(K_BEST_SLOT), truth_box(), 0.75);
    // A width gradient this large would show a factor on the width entry.
    ASSERT_GT(std::abs(expected[K_WIDTH]), 1e-2);
    for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
    {
        const double bound =
            K_BOX_ABSOLUTE_TOLERANCE + K_BOX_RELATIVE_TOLERANCE * std::abs(expected[entry]);
        EXPECT_NEAR(head.delta(K_BEST_SLOT, entry), expected[entry], bound)
            << "entry " << entry;
        EXPECT_EQ(head.delta(K_NEAR_SLOT, entry), 0.0) << "entry " << entry;
    }
}

/** @brief Tests that the class deltas read the published entries as probabilities. */
TEST(Unit_DetectAnchorDeltaActivated, TheClassDeltasReadTheEntriesAsProbabilities)
{
    ActivatedCase head = make_case();

    ASSERT_TRUE(status_is_ok(head.compute()));

    EXPECT_NEAR(head.delta(K_BEST_SLOT, K_FIRST_CLASS), 1.0 - static_cast<double>(0.6f),
                K_TOLERANCE);
    EXPECT_NEAR(head.delta(K_BEST_SLOT, K_FIRST_CLASS + 1), 0.0 - static_cast<double>(0.2f),
                K_TOLERANCE);
    EXPECT_EQ(head.delta(K_NEAR_SLOT, K_FIRST_CLASS), 0.0);
    EXPECT_EQ(head.delta(K_NEAR_SLOT, K_FIRST_CLASS + 1), 0.0);
}

namespace
{
    using AnchorDeltaSupport::K_PI;
    using SushiAI::Result;
    using SushiAI::Vision::Detect::BoxGradient;
    using SushiAI::Vision::Detect::K_ASPECT_TRADE_OFF_EPS;

    /** @brief Holds four float32 values: a box as centre and size, or pixel corners. */
    using Box = std::array<float, 4>;

    /** @brief Holds the tolerance of the hand cases, relative, with the same value as floor. */
    constexpr double K_HAND_TOLERANCE = 1e-5;

    /** @brief Holds the relative tolerance against the dumped deltas, as the report asks. */
    constexpr double K_DUMP_RELATIVE = 1e-4;

    /** @brief Holds the absolute tolerance of a dumped cell's IoU, printed to six digits. */
    constexpr double K_DUMP_IOU_TOLERANCE = 1e-6;

    /** @brief Holds the side of the network input; every truth corner of a case is exact on it. */
    constexpr std::int64_t K_GRADIENT_NETWORK_SIDE = 20;

    /** @brief Holds the Activated size entry whose side is four times its anchor. */
    constexpr float K_FULL_SIZE_ENTRY = 1.0f;

    /** @brief Holds the objectness and the class probability the one slot publishes. */
    constexpr float K_PUBLISHED_SCORE = 0.5f;

    /** @brief Holds one dumped cell: its prediction, its truth and the box deltas stored there. */
    struct DumpedCell
    {
        Box box;     /**< The decoded prediction of the slot, centre and size. */
        Box corners; /**< The truth row as `x1 y1 x2 y2` on a 20-pixel input. */
        Quad delta;  /**< The box entries of the dumped delta. */
        double iou;  /**< The IoU of the two boxes the table prints. */
    };

    /**
     * @brief Holds the three cells of the table in section 11 of REPORT.md, in its order.
     *
     * The deltas and the IoU are the table's dumped column. The boxes are the rows of those
     * cells in `layer_098_yolo_boxes` and `layer_094_yolo_boxes` of `references/yolov7_tiny`;
     * the corners divide by 20 into the float32 rows of its `truth`.
     */
    const std::array<DumpedCell, 3> K_DUMPED_CELLS{{
        {{0.4971627f, 0.516425252f, 0.814291656f, 0.940165877f},
         {6.0f, 4.0f, 14.0f, 16.0f},
         {0.000183408672, -0.00106176059, -0.0385336503, -0.0332950354},
         0.313492},
        {{0.760803103f, 0.716585577f, 0.483656645f, 0.553600848f},
         {11.0f, 13.0f, 17.0f, 17.0f},
         {-0.0112515502, 0.0061833025, -0.04532497, -0.0397488885},
         0.224087},
        {{0.259860605f, 0.294024527f, 0.236384213f, 0.275245309f},
         {4.0f, 4.5f, 6.0f, 7.5f},
         {-0.00749072898, 0.00453934958, -0.097358264, -0.0835391209},
         0.230543},
    }};

    /** @brief Holds a 1x1 Activated head of one slot and one class, with one truth. */
    struct GradientCase
    {
        HeadOptions options;          /**< One class, the Activated coding. */
        AnchorLayout layout;          /**< The one anchor of the one head. */
        HeadTrainingOptions training; /**< Complete IoU boxes and the form under test. */
        GroundTruthBatch truth;       /**< One row in pixel corners. */
        std::vector<float> head;      /**< `[1, 1, 1, 6]`. */
        HeadDelta result;             /**< What the call wrote. */

        /** @brief Runs compute_head_delta; returns the slot's four box deltas or the failure. */
        [[nodiscard]] Result<Quad> compute()
        {
            const HeadShape shape{1, 1, 1};
            const NetworkSize network{K_GRADIENT_NETWORK_SIDE, K_GRADIENT_NETWORK_SIDE};
            SA_TRY(compute_head_delta(head, shape, options, layout, 0, training, truth,
                                      network, result));

            Quad delta{};
            for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                delta[entry] = static_cast<double>(result.delta[entry]);
            return delta;
        }
    };

    /**
     * @brief Returns a case whose slot decodes to @p box bit for bit, with a box weight of 1.
     * @param box A centre inside the one cell and positive sides.
     * @param corners The truth as `x1 y1 x2 y2` on the 20-pixel input.
     */
    GradientCase make_case(const Box& box, const Box& corners, BoxGradient form)
    {
        GradientCase made;
        made.options.num_classes = 1;
        made.options.coding = HeadCoding::Activated;
        // An anchor of a quarter of the side decodes the full size entry to the side itself.
        made.layout.anchors = {{box[K_WIDTH] / 4.0f, box[K_HEIGHT] / 4.0f}};
        made.layout.head_anchors = {{0}};
        made.training.box_loss = BoxLoss::CompleteIoU;
        made.training.box_gradient = form;
        made.training.box_weight = 1.0f;

        made.truth.batch = 1;
        made.truth.max_boxes = 1;
        made.truth.labels = {0};
        made.truth.valid = {1};
        made.truth.boxes.assign(corners.begin(), corners.end());

        made.head = {box[0],            box[1],            K_FULL_SIZE_ENTRY,
                     K_FULL_SIZE_ENTRY, K_PUBLISHED_SCORE, K_PUBLISHED_SCORE};
        return made;
    }

    /** @brief Returns the box deltas of the slot of @p box under @p form. */
    [[nodiscard]] Result<Quad> measure_delta(const Box& box, const Box& corners,
                                             BoxGradient form)
    {
        return make_case(box, corners, form).compute();
    }

    /** @brief Expects @p value to equal @p expected within the hand tolerance. */
    void expect_hand_value(double value, double expected)
    {
        EXPECT_NEAR(value, expected, K_HAND_TOLERANCE + K_HAND_TOLERANCE * std::abs(expected));
    }
} // namespace

/** @brief Tests that options left at their default take the analytic form. */
TEST(Unit_DetectAnchorDeltaGradient, TheDefaultFormIsAnalytic)
{
    // A 0.2 square centred in a 0.4 square: IoU = w * h / 0.16, so its slope by w is 1.25.
    const Box box{0.5f, 0.5f, 0.2f, 0.2f};
    const Box corners{6.0f, 6.0f, 14.0f, 14.0f};
    GradientCase head = make_case(box, corners, BoxGradient::Analytic);
    head.training.box_gradient = HeadTrainingOptions{}.box_gradient;

    SA_ASSERT_OK_AND_ASSIGN(const Quad delta, head.compute());

    expect_hand_value(delta[K_WIDTH], 1.25);
    expect_hand_value(delta[K_HEIGHT], 1.25);
}

/** @brief Tests that the compatible form doubles the size deltas the analytic form writes. */
TEST(Unit_DetectAnchorDeltaGradient, TheCompatibleFormDoublesTheSizeSlopeOfTheOverlap)
{
    // A 0.2 square centred in a 0.4 square: IoU = w * h / 0.16, so its slope by w is 1.25.
    const Box box{0.5f, 0.5f, 0.2f, 0.2f};
    const Box corners{6.0f, 6.0f, 14.0f, 14.0f};

    SA_ASSERT_OK_AND_ASSIGN(const Quad analytic,
                            measure_delta(box, corners, BoxGradient::Analytic));
    SA_ASSERT_OK_AND_ASSIGN(const Quad compatible,
                            measure_delta(box, corners, BoxGradient::DarknetCompatible));

    expect_hand_value(analytic[K_WIDTH], 1.25);
    expect_hand_value(analytic[K_HEIGHT], 1.25);
    expect_hand_value(compatible[K_WIDTH], 2.5);
    expect_hand_value(compatible[K_HEIGHT], 2.5);
    expect_hand_value(analytic[0], 0.0);
    expect_hand_value(compatible[0], 0.0);
}

/** @brief Tests that the compatible form flips the enclosing-box term of the centre delta. */
TEST(Unit_DetectAnchorDeltaGradient, TheCompatibleFormFlipsTheEnclosingTermOfTheCentre)
{
    // Disjoint equal squares: S = 0.25, Cw = 0.7, C = 0.53; x is (0.53 -+ 0.35) / 0.2809.
    const Box box{0.2f, 0.5f, 0.2f, 0.2f};
    const Box corners{12.0f, 8.0f, 16.0f, 12.0f};

    SA_ASSERT_OK_AND_ASSIGN(const Quad analytic,
                            measure_delta(box, corners, BoxGradient::Analytic));
    SA_ASSERT_OK_AND_ASSIGN(const Quad compatible,
                            measure_delta(box, corners, BoxGradient::DarknetCompatible));

    expect_hand_value(analytic[0], 0.18 / 0.2809);
    expect_hand_value(compatible[0], 0.88 / 0.2809);
    expect_hand_value(analytic[K_WIDTH], 0.7 * 0.25 / 0.2809);
    expect_hand_value(compatible[K_WIDTH], 0.7 * 0.25 / 0.2809);
    expect_hand_value(compatible[1], 0.0);
    expect_hand_value(compatible[K_HEIGHT], 0.0);
}

/** @brief Tests that the compatible form holds the trade-off fixed and drops a divisor. */
TEST(Unit_DetectAnchorDeltaGradient, TheCompatibleFormSimplifiesTheAspectTerm)
{
    // A 0.2 by 0.1 box centred in a 0.4 square: IoU 0.125, w^2 + h^2 = 0.05, no distance part.
    const Box box{0.5f, 0.5f, 0.2f, 0.1f};
    const Box corners{6.0f, 6.0f, 14.0f, 14.0f};

    SA_ASSERT_OK_AND_ASSIGN(const Quad analytic,
                            measure_delta(box, corners, BoxGradient::Analytic));
    SA_ASSERT_OK_AND_ASSIGN(const Quad compatible,
                            measure_delta(box, corners, BoxGradient::DarknetCompatible));

    const double angle_gap = std::atan(1.0) - std::atan(2.0);
    const double aspect = 4.0 * angle_gap * angle_gap / (K_PI * K_PI);
    const double trade_off = aspect / (1.0 - 0.125 + aspect + K_ASPECT_TRADE_OFF_EPS);
    const double slope = 8.0 / (K_PI * K_PI) * angle_gap;
    const double kept = 1.0 - trade_off * trade_off;
    const double full = trade_off * (2.0 - trade_off) * slope / 0.05;

    expect_hand_value(compatible[K_WIDTH], 1.25 + trade_off * slope * 0.1);
    expect_hand_value(compatible[K_HEIGHT], 2.5 - trade_off * slope * 0.2);
    expect_hand_value(analytic[K_WIDTH], kept * 0.625 + full * 0.1);
    expect_hand_value(analytic[K_HEIGHT], kept * 1.25 - full * 0.2);
}

/** @brief Tests the compatible form against the twelve dumped deltas of the report. */
TEST(Unit_DetectAnchorDeltaGradient, TheCompatibleFormReproducesTheDumpedDeltas)
{
    for (std::size_t cell = 0; cell < K_DUMPED_CELLS.size(); ++cell)
    {
        const DumpedCell& dumped = K_DUMPED_CELLS[cell];
        GradientCase head =
            make_case(dumped.box, dumped.corners, BoxGradient::DarknetCompatible);
        // The head the cells were dumped from: yolov7-tiny, whose division here is by 1.
        head.training.box_weight = 0.05f;
        head.training.max_delta = 2.0f;
        head.training.thresholds.match_threshold = 0.2f;
        head.training.objectness_target = ObjectnessTarget::BestIoU;

        SA_ASSERT_OK_AND_ASSIGN(const Quad delta, head.compute());

        EXPECT_NEAR(head.result.iou_sum, dumped.iou, K_DUMP_IOU_TOLERANCE) << "cell " << cell;
        for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
        {
            EXPECT_NEAR(delta[entry], dumped.delta[entry],
                        K_DUMP_RELATIVE * std::abs(dumped.delta[entry]))
                << "cell " << cell << ", entry " << entry;
        }
    }
}

/** @brief Tests that the analytic form does not give the dumped size deltas. */
TEST(Unit_DetectAnchorDeltaGradient, TheAnalyticFormDepartsFromTheDumpedSizeDeltas)
{
    const DumpedCell& dumped = K_DUMPED_CELLS[0];
    GradientCase head = make_case(dumped.box, dumped.corners, BoxGradient::Analytic);
    head.training.box_weight = 0.05f;

    SA_ASSERT_OK_AND_ASSIGN(const Quad delta, head.compute());

    // The overlap part of the size is halved, which moves both entries by tens of percent.
    EXPECT_GT(std::abs(delta[K_WIDTH] - dumped.delta[K_WIDTH]),
              0.1 * std::abs(dumped.delta[K_WIDTH]));
    EXPECT_GT(std::abs(delta[K_HEIGHT] - dumped.delta[K_HEIGHT]),
              0.1 * std::abs(dumped.delta[K_HEIGHT]));
}

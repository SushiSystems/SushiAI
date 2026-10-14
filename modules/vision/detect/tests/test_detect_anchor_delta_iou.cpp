/**************************************************************************/
/* test_detect_anchor_delta_iou.cpp                                       */
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
 * @file test_detect_anchor_delta_iou.cpp
 * @brief Tests the complete IoU delta of an anchor head and its further assignments.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "anchor_delta_support.hpp"

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

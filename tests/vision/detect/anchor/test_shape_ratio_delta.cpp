/**************************************************************************/
/* test_shape_ratio_delta.cpp                                             */
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
 * @file test_shape_ratio_delta.cpp
 * @brief Tests the loss, the objectness targets, the factors and the delta of one anchor head
 *        under the shape-ratio assignment.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::build_objectness_targets;
    using SushiAI::Vision::Detect::Anchor::compute_shape_ratio_delta;
    using SushiAI::Vision::Detect::Anchor::derive_shape_ratio_factors;
    using SushiAI::Vision::Detect::Anchor::HeadCoding;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::measure_shape_ratio_loss;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::objectness_balance;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioDeltaInput;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioFactors;
    using SushiAI::Vision::Detect::Anchor::ShapeRatioLoss;
    using SushiAI::Vision::Detect::Anchor::SizeForm;
    using SushiAI::Vision::Detect::Anchor::TargetAssignment;

    /** @brief Holds the step of a logit in the central difference. */
    constexpr float K_STEP = 1e-3f;

    /**
     * @brief Holds the absolute tolerance of a delta against its central difference.
     *
     * The complete IoU is measured in float, about 1e-7 of a loss near 1, which a step of
     * 2e-3 turns into 5e-5 before the box factor of 0.075; the curvature adds about 1e-6.
     */
    constexpr double K_DIFFERENCE_TOLERANCE = 1e-4;

    /** @brief Holds the tolerance of a value worked by hand; each is near exact in float. */
    constexpr double K_HAND_TOLERANCE = 1e-5;

    /** @brief Holds the class count of the two-level fixture. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Returns the two-level layout: grid units 1 by 1, 2 by 1.5; 1.5 by 1.5, 2 by 1. */
    AnchorLayout make_two_level_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.25f, 0.25f}, {0.5f, 0.375f}, {0.75f, 0.75f}, {1.0f, 0.5f}};
        layout.head_anchors = {{0, 1}, {2, 3}};
        return layout;
    }

    /** @brief Returns @p count head values of a fixed sequence in (-0.8, 0.8). */
    std::vector<float> make_head(std::size_t count)
    {
        std::vector<float> head(count);
        for (std::size_t index = 0; index < count; ++index)
            head[index] = 0.8f * std::sin(0.37f * static_cast<float>(index) + 0.1f);
        return head;
    }

    /** @brief Returns an input over @p head for level @p level of a one-image batch. */
    ShapeRatioDeltaInput make_input(const AnchorLayout& layout, const HeadTrainingOptions& training,
                                    std::int64_t level, std::int64_t side,
                                    const std::vector<float>& head,
                                    const std::vector<TargetAssignment>& assignments)
    {
        ShapeRatioDeltaInput input;
        input.head = SushiRuntime::span<const float>(head.data(), head.size());
        input.shape = HeadShape{1, side, side};
        input.options.num_classes = K_CLASSES;
        input.options.coding = HeadCoding::Logit;
        input.options.size = SizeForm::SquaredSigmoid;
        input.options.scale_xy = 2.0f;
        input.layout = &layout;
        input.head_index = level;
        input.training = &training;
        input.network = NetworkSize{64, 64};
        input.assignments =
            SushiRuntime::span<const TargetAssignment>(assignments.data(), assignments.size());
        return input;
    }

    /** @brief Holds pi in double. */
    constexpr double K_PI = 3.141592653589793;

    /** @brief Holds the anchor slots of each level of the two-level fixture. */
    constexpr std::size_t K_SLOTS = 2;

    /** @brief Holds one box as centre and size in grid units, in double. */
    using Quad = std::array<double, 4>;

    /** @brief Returns the logistic of @p value in double. */
    double logistic(double value)
    {
        return 1.0 / (1.0 + std::exp(-value));
    }

    /** @brief Decodes the slot @p record names, as the shape-ratio delta decodes it. */
    Quad decode_slot(const ShapeRatioDeltaInput& input, const TargetAssignment& record)
    {
        const std::size_t side = static_cast<std::size_t>(input.shape.width);
        const std::size_t cell =
            static_cast<std::size_t>(record.row) * side + static_cast<std::size_t>(record.col);
        const std::size_t slot = cell * K_SLOTS + static_cast<std::size_t>(record.slot);
        const float* entries =
            input.head.data() + slot * static_cast<std::size_t>(5 + K_CLASSES);
        const std::int64_t anchor =
            input.layout->head_anchors[static_cast<std::size_t>(input.head_index)]
                                      [static_cast<std::size_t>(record.slot)];
        const SushiAI::Vision::Detect::Anchor::AnchorSize size =
            input.layout->anchors[static_cast<std::size_t>(anchor)];
        const double scale = static_cast<double>(input.options.scale_xy);
        const double shift = (scale - 1.0) / 2.0;
        const double width = 2.0 * logistic(entries[2]);
        const double height = 2.0 * logistic(entries[3]);
        return {logistic(entries[0]) * scale - shift, logistic(entries[1]) * scale - shift,
                width * width * static_cast<double>(size.width) * static_cast<double>(side),
                height * height * static_cast<double>(size.height) * static_cast<double>(side)};
    }

    /** @brief Returns the IoU, the distance term and the aspect term of a pair, in double. */
    std::array<double, 3> measure_terms(const Quad& box, const Quad& truth)
    {
        const double left = std::max(box[0] - box[2] / 2.0, truth[0] - truth[2] / 2.0);
        const double right = std::min(box[0] + box[2] / 2.0, truth[0] + truth[2] / 2.0);
        const double top = std::max(box[1] - box[3] / 2.0, truth[1] - truth[3] / 2.0);
        const double bottom = std::min(box[1] + box[3] / 2.0, truth[1] + truth[3] / 2.0);
        const double shared = std::max(right - left, 0.0) * std::max(bottom - top, 0.0);
        const double iou = shared / (box[2] * box[3] + truth[2] * truth[3] - shared);

        const double hull_width = std::max(box[0] + box[2] / 2.0, truth[0] + truth[2] / 2.0) -
                                  std::min(box[0] - box[2] / 2.0, truth[0] - truth[2] / 2.0);
        const double hull_height = std::max(box[1] + box[3] / 2.0, truth[1] + truth[3] / 2.0) -
                                   std::min(box[1] - box[3] / 2.0, truth[1] - truth[3] / 2.0);
        const double shift_x = box[0] - truth[0];
        const double shift_y = box[1] - truth[1];
        const double distance = (shift_x * shift_x + shift_y * shift_y) /
                                (hull_width * hull_width + hull_height * hull_height);

        const double gap = std::atan(truth[2] / truth[3]) - std::atan(box[2] / box[3]);
        return {iou, distance, 4.0 * gap * gap / (K_PI * K_PI)};
    }

    /** @brief Returns the trade-off of the aspect term of a pair at its current boxes. */
    double measure_trade_off(const Quad& box, const Quad& truth)
    {
        const std::array<double, 3> terms = measure_terms(box, truth);
        return terms[2] /
               (1.0 - terms[0] + terms[2] + SushiAI::Vision::Detect::K_ASPECT_TRADE_OFF_EPS);
    }

    /** @brief Returns the truth of @p record as centre and size, in double. */
    Quad truth_of(const TargetAssignment& record)
    {
        return {static_cast<double>(record.box_x), static_cast<double>(record.box_y),
                static_cast<double>(record.box_width), static_cast<double>(record.box_height)};
    }

    /**
     * @brief Returns the mean of 1 - complete IoU over the level's assignments, the trade-off
     *        of assignment k taken from @p frozen[k] instead of its own boxes.
     */
    double measure_frozen_box(const ShapeRatioDeltaInput& input,
                              const std::vector<double>& frozen)
    {
        if (input.assignments.empty())
            return 0.0;

        double sum = 0.0;
        for (std::size_t index = 0; index < input.assignments.size(); ++index)
        {
            const TargetAssignment& record = input.assignments[index];
            const std::array<double, 3> terms =
                measure_terms(decode_slot(input, record), truth_of(record));
            sum += 1.0 - (terms[0] - terms[1] - frozen[index] * terms[2]);
        }
        return sum / static_cast<double>(input.assignments.size());
    }

    /**
     * @brief Returns the level's total loss with its box mean replaced by measure_frozen_box.
     * @param measured What measure_shape_ratio_loss returned for @p input.
     */
    double replace_box_term(const ShapeRatioDeltaInput& input, const ShapeRatioLoss& measured,
                            const std::vector<double>& frozen)
    {
        const double box_factor =
            derive_shape_ratio_factors(input).box * static_cast<double>(input.shape.batch);
        return measured.total - box_factor * measured.box +
               box_factor * measure_frozen_box(input, frozen);
    }

    /**
     * @brief Expects every delta of @p input's level to equal -B times the central
     *        difference of the loss by that logit, with the objectness targets and the
     *        trade-off of each complete IoU held at their values on the unshifted head.
     */
    void expect_delta_matches_difference(const AnchorLayout& layout,
                                         const HeadTrainingOptions& training, std::int64_t level,
                                         std::int64_t side,
                                         const std::vector<TargetAssignment>& assignments)
    {
        const std::size_t values = static_cast<std::size_t>(side * side * 2 * (5 + K_CLASSES));
        std::vector<float> head = make_head(values);
        const ShapeRatioDeltaInput base =
            make_input(layout, training, level, side, head, assignments);

        HeadDelta delta;
        SA_EXPECT_OK(compute_shape_ratio_delta(base, delta));
        ASSERT_EQ(delta.delta.size(), values);
        std::vector<float> targets;
        SA_EXPECT_OK(build_objectness_targets(base, targets));
        std::vector<double> frozen;
        for (const TargetAssignment& record : assignments)
            frozen.push_back(measure_trade_off(decode_slot(base, record), truth_of(record)));

        for (std::size_t index = 0; index < values; ++index)
        {
            const float held = head[index];
            head[index] = held + K_STEP;
            const float raised = head[index];
            const ShapeRatioDeltaInput up =
                make_input(layout, training, level, side, head, assignments);
            SA_ASSERT_OK_AND_ASSIGN(
                const ShapeRatioLoss above,
                measure_shape_ratio_loss(
                    up, SushiRuntime::span<const float>(targets.data(), targets.size())));
            const double above_total = replace_box_term(up, above, frozen);
            head[index] = held - K_STEP;
            const float lowered = head[index];
            const ShapeRatioDeltaInput down =
                make_input(layout, training, level, side, head, assignments);
            SA_ASSERT_OK_AND_ASSIGN(
                const ShapeRatioLoss below,
                measure_shape_ratio_loss(
                    down, SushiRuntime::span<const float>(targets.data(), targets.size())));
            const double below_total = replace_box_term(down, below, frozen);
            head[index] = held;

            const double slope = (above_total - below_total) /
                                 (static_cast<double>(raised) - static_cast<double>(lowered));
            EXPECT_NEAR(static_cast<double>(delta.delta[index]), -slope, K_DIFFERENCE_TOLERANCE)
                << "level " << level << ", entry " << index;
        }
    }
} // namespace

/** @brief Tests that the delta of either level is -B times the central difference of its loss. */
TEST(Small_DetectAnchorShapeRatioDelta, TheDeltaIsTheCentralDifferenceOfTheLoss)
{
    const AnchorLayout layout = make_two_level_layout();
    const HeadTrainingOptions training;

    // Truth 0 takes two cells of slot 0; truth 1 slot 1 of level 0 and slot 0 of level 1.
    const std::vector<TargetAssignment> fine = {
        {0, 0, 1, 1, 0.3f, 0.6f, 1.2f, 0.9f, 0, 0},
        {0, 0, 1, 2, -0.7f, 0.6f, 1.2f, 0.9f, 0, 0},
        {0, 1, 2, 3, 0.4f, 0.2f, 2.1f, 1.4f, 2, 1},
    };
    const std::vector<TargetAssignment> coarse = {
        {0, 0, 0, 1, 0.6f, 0.8f, 1.3f, 1.5f, 2, 1},
    };

    expect_delta_matches_difference(layout, training, 0, 4, fine);
    expect_delta_matches_difference(layout, training, 1, 2, coarse);
}

/** @brief Tests the objectness target and the balance of the third of three levels by hand. */
TEST(Small_DetectAnchorShapeRatioDelta, ObjectnessTargetAndBalanceOnAHandWorkedCase)
{
    // Three levels of one anchor each; level 2 has 2 by 2 cells and an anchor of one cell.
    AnchorLayout layout;
    layout.anchors = {{0.125f, 0.125f}, {0.25f, 0.25f}, {0.5f, 0.5f}};
    layout.head_anchors = {{0}, {1}, {2}};
    const HeadTrainingOptions training;

    // Zero logits decode to offset 0.5 and side (2 * 0.5)^2 * 1 = 1, which the truth equals.
    const std::vector<float> head(static_cast<std::size_t>(2 * 2 * (5 + 1)), 0.0f);
    const std::vector<TargetAssignment> assignments = {{0, 0, 1, 0, 0.5f, 0.5f, 1.0f, 1.0f, 0, 0}};

    ShapeRatioDeltaInput input;
    input.head = SushiRuntime::span<const float>(head.data(), head.size());
    input.shape = HeadShape{1, 2, 2};
    input.options.num_classes = 1;
    input.options.scale_xy = 2.0f;
    input.options.size = SizeForm::SquaredSigmoid;
    input.layout = &layout;
    input.head_index = 2;
    input.training = &training;
    input.network = NetworkSize{640, 640};
    input.assignments =
        SushiRuntime::span<const TargetAssignment>(assignments.data(), assignments.size());

    std::vector<float> targets;
    SA_EXPECT_OK(build_objectness_targets(input, targets));
    ASSERT_EQ(targets.size(), 4u);
    EXPECT_NEAR(targets[2], 1.0, K_HAND_TOLERANCE);
    EXPECT_EQ(targets[0], 0.0f);
    EXPECT_EQ(targets[1], 0.0f);
    EXPECT_EQ(targets[3], 0.0f);

    // Factor 1 * 1 * 3 / 3 * 0.4 over 4 slots; the gap is 0.5 - 1 assigned, 0.5 elsewhere.
    HeadDelta delta;
    SA_EXPECT_OK(compute_shape_ratio_delta(input, delta));
    EXPECT_NEAR(delta.delta[2 * 6 + 4], 0.05, K_HAND_TOLERANCE);
    EXPECT_NEAR(delta.delta[0 * 6 + 4], -0.05, K_HAND_TOLERANCE);
    EXPECT_NEAR(delta.delta[3 * 6 + 4], -0.05, K_HAND_TOLERANCE);
    EXPECT_EQ(delta.assignments, 1);
    EXPECT_EQ(delta.assigned_row[2], 0);
}

/** @brief Tests the balance lists for three levels and for any other count. */
TEST(Small_DetectAnchorShapeRatioDelta, TheBalanceFollowsTheLevelCount)
{
    EXPECT_DOUBLE_EQ(objectness_balance(3, 0), 4.0);
    EXPECT_DOUBLE_EQ(objectness_balance(3, 1), 1.0);
    EXPECT_DOUBLE_EQ(objectness_balance(3, 2), 0.4);
    EXPECT_DOUBLE_EQ(objectness_balance(4, 2), 0.25);
    EXPECT_DOUBLE_EQ(objectness_balance(5, 3), 0.06);
    EXPECT_DOUBLE_EQ(objectness_balance(5, 4), 0.02);
    EXPECT_DOUBLE_EQ(objectness_balance(2, 1), 1.0);
}

/** @brief Tests that the factors follow the level count, the class count and the input size. */
TEST(Small_DetectAnchorShapeRatioDelta, TheFactorsDeriveFromLevelsClassesAndInputSize)
{
    const AnchorLayout layout = make_two_level_layout();
    const HeadTrainingOptions training;
    const std::vector<float> head;
    const std::vector<TargetAssignment> none;

    ShapeRatioDeltaInput input = make_input(layout, training, 1, 2, head, none);
    input.options.num_classes = 40;
    input.network = NetworkSize{320, 320};

    // Two levels give 3 / 2; 40 classes give 40 / 80; 320 pixels give (320 / 640)^2.
    const ShapeRatioFactors factors = derive_shape_ratio_factors(input);
    EXPECT_NEAR(factors.box, 0.05 * 1.5, K_HAND_TOLERANCE);
    EXPECT_NEAR(factors.classes, 0.5 * 0.5 * 1.5, K_HAND_TOLERANCE);
    EXPECT_NEAR(factors.objectness, 1.0 * 0.25 * 1.5 * 1.0, K_HAND_TOLERANCE);
}

/** @brief Tests that an activated head, exponential sides and an outside assignment are refused. */
TEST(Small_DetectAnchorShapeRatioDelta, RefusesAHeadOrAnAssignmentItCannotRead)
{
    const AnchorLayout layout = make_two_level_layout();
    const HeadTrainingOptions training;
    const std::vector<float> head = make_head(2 * 2 * 2 * (5 + K_CLASSES));
    const std::vector<TargetAssignment> outside = {{0, 0, 2, 0, 0.5f, 0.5f, 1.0f, 1.0f, 0, 0}};
    const std::vector<TargetAssignment> none;
    HeadDelta delta;

    ShapeRatioDeltaInput activated = make_input(layout, training, 1, 2, head, none);
    activated.options.coding = HeadCoding::Activated;
    activated.options.size = SizeForm::Exponential;
    SA_EXPECT_FAILURE(compute_shape_ratio_delta(activated, delta),
                      StatusCode::InvalidDescription, "logits");

    ShapeRatioDeltaInput exponential = make_input(layout, training, 1, 2, head, none);
    exponential.options.size = SizeForm::Exponential;
    SA_EXPECT_FAILURE(compute_shape_ratio_delta(exponential, delta),
                      StatusCode::InvalidDescription, "squared_sigmoid");

    SA_EXPECT_FAILURE(
        compute_shape_ratio_delta(make_input(layout, training, 1, 2, head, outside), delta),
        StatusCode::InvalidDescription, "outside head 1");
}

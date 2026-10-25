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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
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
        input.options.scale_xy = 2.0f;
        input.layout = &layout;
        input.head_index = level;
        input.training = &training;
        input.network = NetworkSize{64, 64};
        input.assignments =
            SushiRuntime::span<const TargetAssignment>(assignments.data(), assignments.size());
        return input;
    }

    /**
     * @brief Expects every delta of @p input's level to equal -B times the central
     *        difference of the loss by that logit, with the objectness targets held fixed.
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

        for (std::size_t index = 0; index < values; ++index)
        {
            const float held = head[index];
            head[index] = held + K_STEP;
            const float raised = head[index];
            SA_ASSERT_OK_AND_ASSIGN(
                const ShapeRatioLoss above,
                measure_shape_ratio_loss(
                    make_input(layout, training, level, side, head, assignments),
                    SushiRuntime::span<const float>(targets.data(), targets.size())));
            head[index] = held - K_STEP;
            const float lowered = head[index];
            SA_ASSERT_OK_AND_ASSIGN(
                const ShapeRatioLoss below,
                measure_shape_ratio_loss(
                    make_input(layout, training, level, side, head, assignments),
                    SushiRuntime::span<const float>(targets.data(), targets.size())));
            head[index] = held;

            const double slope = (above.total - below.total) /
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

/** @brief Tests that a head of activated values and an assignment outside the head are refused. */
TEST(Small_DetectAnchorShapeRatioDelta, RefusesAnActivatedHeadAndAnAssignmentOutsideIt)
{
    const AnchorLayout layout = make_two_level_layout();
    const HeadTrainingOptions training;
    const std::vector<float> head = make_head(2 * 2 * 2 * (5 + K_CLASSES));
    const std::vector<TargetAssignment> outside = {{0, 0, 2, 0, 0.5f, 0.5f, 1.0f, 1.0f, 0, 0}};
    const std::vector<TargetAssignment> none;
    HeadDelta delta;

    ShapeRatioDeltaInput activated = make_input(layout, training, 1, 2, head, none);
    activated.options.coding = HeadCoding::Activated;
    SA_EXPECT_FAILURE(compute_shape_ratio_delta(activated, delta),
                      StatusCode::InvalidDescription, "logits");

    SA_EXPECT_FAILURE(
        compute_shape_ratio_delta(make_input(layout, training, 1, 2, head, outside), delta),
        StatusCode::InvalidDescription, "outside head 1");
}

/**************************************************************************/
/* test_detect_anchor_delta_activated.cpp                                 */
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
 * @file test_detect_anchor_delta_activated.cpp
 * @brief Tests the delta of an anchor head under the Activated coding and the BestIoU target.
 * @author Mustafa Garip
 */

#include <array>
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
    using SushiAI::Vision::Detect::Anchor::HeadCoding;
    using SushiAI::Vision::Detect::Anchor::HeadDelta;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::NetworkSize;
    using SushiAI::Vision::Detect::Anchor::ObjectnessTarget;
    using SushiAI::Vision::Detect::GroundTruthBatch;

    /** @brief Holds the absolute tolerance of an objectness or class delta. */
    constexpr double K_TOLERANCE = 1e-6;

    /** @brief Holds the absolute tolerance of a box delta against a central difference. */
    constexpr double K_BOX_ABSOLUTE_TOLERANCE = 1e-5;

    /** @brief Holds the tolerance of a box delta relative to a central difference. */
    constexpr double K_BOX_RELATIVE_TOLERANCE = 1e-5;

    /** @brief Holds the side of the square network input, which makes every corner exact. */
    constexpr std::int64_t K_NETWORK_SIDE = 64;

    /** @brief Holds the entry of the box width within an anchor slot. */
    constexpr std::size_t K_WIDTH = 2;

    /** @brief Holds the entry of the box height within an anchor slot. */
    constexpr std::size_t K_HEIGHT = 3;

    /** @brief Holds the entry of the objectness within an anchor slot. */
    constexpr std::size_t K_OBJECTNESS = 4;

    /** @brief Holds the entry of class 0; class c follows at K_FIRST_CLASS + c. */
    constexpr std::size_t K_FIRST_CLASS = 5;

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

/**************************************************************************/
/* test_detect_anchor_delta_gradient.cpp                                  */
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
 * @file test_detect_anchor_delta_gradient.cpp
 * @brief Tests both forms of the complete IoU gradient through the delta of an anchor head.
 * @author Mustafa Garip
 * @see docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md
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
#include <SushiAI/vision/detect/iou_gradient.hpp>
#include <common/test_status.hpp>

#include "anchor_delta_support.hpp"

namespace
{
    using AnchorDeltaSupport::K_BOX_ENTRIES;
    using AnchorDeltaSupport::K_PI;
    using AnchorDeltaSupport::Quad;
    using SushiAI::Result;
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
    using SushiAI::Vision::Detect::BoxGradient;
    using SushiAI::Vision::Detect::GroundTruthBatch;
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
    constexpr std::int64_t K_NETWORK_SIDE = 20;

    /** @brief Holds the Activated size entry whose side is four times its anchor. */
    constexpr float K_FULL_SIZE_ENTRY = 1.0f;

    /** @brief Holds the objectness and the class probability the one slot publishes. */
    constexpr float K_PUBLISHED_SCORE = 0.5f;

    /** @brief Holds the entry of the box width within an anchor slot. */
    constexpr std::size_t K_WIDTH = 2;

    /** @brief Holds the entry of the box height within an anchor slot. */
    constexpr std::size_t K_HEIGHT = 3;

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
            const NetworkSize network{K_NETWORK_SIDE, K_NETWORK_SIDE};
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

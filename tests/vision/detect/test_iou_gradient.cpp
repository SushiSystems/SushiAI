/**************************************************************************/
/* test_iou_gradient.cpp                                                  */
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
 * @file test_iou_gradient.cpp
 * @brief Tests both gradients of complete IoU: a finite difference and the dumped deltas.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/iou_gradient.hpp>

namespace Detect = SushiAI::Vision::Detect;

namespace
{
    /** @brief Holds one box as `x_centre y_centre width height`. */
    using Box = std::array<float, 4>;

    /** @brief Holds one box, or one gradient, in double. */
    using Quad = std::array<double, 4>;

    /** @brief Holds pi in double. */
    constexpr double K_PI = 3.141592653589793;

    /** @brief Holds the half step of the central difference. */
    constexpr double K_DIFFERENCE_STEP = 1e-6;

    /** @brief Holds the relative tolerance of the analytic form against the difference. */
    constexpr double K_ANALYTIC_RELATIVE = 1e-6;

    /** @brief Holds the absolute floor of that tolerance, for components near zero. */
    constexpr double K_ANALYTIC_ABSOLUTE = 1e-7;

    /** @brief Holds the relative tolerance against the dumped deltas, as the report asks. */
    constexpr double K_DUMP_RELATIVE = 1e-4;

    /** @brief Holds the tolerance of the hand cases, relative, with the same value as floor. */
    constexpr double K_HAND_TOLERANCE = 1e-5;

    /** @brief Holds the least gap between edges at which a difference crosses no kink. */
    constexpr double K_KINK_GAP = 1e-3;

    /** @brief Holds the box weight of the model the deltas were dumped from. */
    constexpr double K_DUMP_BOX_WEIGHT = 0.05;

    /** @brief Holds the seed of the pair sequence. */
    constexpr std::uint64_t K_PAIR_SEED = 20261009;

    /** @brief Holds the count of seeded pairs the difference is taken on. */
    constexpr int K_PAIR_COUNT = 100;

    /** @brief Holds one dumped cell: its two boxes and the four box deltas stored there. */
    struct DumpedCell
    {
        Box box;    /**< The decoded prediction of the slot. */
        Box target; /**< The truth row assigned to it. */
        Quad delta; /**< The box entries of the delta, gradient times the box weight. */
    };

    /**
     * @brief Holds the three cells of the table in section 11 of REPORT.md, in its order.
     *
     * The deltas are the table's dumped column. The boxes are the rows of those cells in
     * `layer_098_yolo_boxes`, `layer_094_yolo_boxes` and `truth` of `references/yolov7_tiny`.
     */
    const std::array<DumpedCell, 3> K_DUMPED_CELLS{{
        {{0.4971627f, 0.516425252f, 0.814291656f, 0.940165877f},
         {0.5f, 0.5f, 0.400000006f, 0.600000024f},
         {0.000183408672, -0.00106176059, -0.0385336503, -0.0332950354}},
        {{0.760803103f, 0.716585577f, 0.483656645f, 0.553600848f},
         {0.699999988f, 0.75f, 0.300000012f, 0.200000003f},
         {-0.0112515502, 0.0061833025, -0.04532497, -0.0397488885}},
        {{0.259860605f, 0.294024527f, 0.236384213f, 0.275245309f},
         {0.25f, 0.300000012f, 0.100000001f, 0.150000006f},
         {-0.00749072898, 0.00453934958, -0.097358264, -0.0835391209}},
    }};

    /** @brief Holds a 64-bit linear congruential sequence of numbers in [0, 1). */
    struct UnitSequence
    {
        std::uint64_t state = K_PAIR_SEED; /**< The last state. */

        /** @brief Advances the sequence and returns its next number. */
        double next() noexcept
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<double>(state >> 11) / 9007199254740992.0;
        }
    };

    /** @brief Returns @p box in double. */
    Quad widen(const Box& box) noexcept
    {
        return {static_cast<double>(box[0]), static_cast<double>(box[1]),
                static_cast<double>(box[2]), static_cast<double>(box[3])};
    }

    /** @brief Returns the left, right, top and bottom edges of a centre-and-size box. */
    Quad measure_edges(const Quad& box) noexcept
    {
        return {box[0] - box[2] / 2.0, box[0] + box[2] / 2.0, box[1] - box[3] / 2.0,
                box[1] + box[3] / 2.0};
    }

    /** @brief Returns the complete IoU of section 4.6 of SPEC.md, in double. */
    double measure_complete_iou(const Quad& box, const Quad& target) noexcept
    {
        const Quad own = measure_edges(box);
        const Quad truth = measure_edges(target);
        const double across =
            std::max(std::min(own[1], truth[1]) - std::max(own[0], truth[0]), 0.0);
        const double down =
            std::max(std::min(own[3], truth[3]) - std::max(own[2], truth[2]), 0.0);
        const double intersection = across * down;
        const double iou =
            intersection / (box[2] * box[3] + target[2] * target[3] - intersection);

        const double shift_x = box[0] - target[0];
        const double shift_y = box[1] - target[1];
        const double hull_width = std::max(own[1], truth[1]) - std::min(own[0], truth[0]);
        const double hull_height = std::max(own[3], truth[3]) - std::min(own[2], truth[2]);
        const double distance = (shift_x * shift_x + shift_y * shift_y) /
                                (hull_width * hull_width + hull_height * hull_height);

        const double angle_gap = std::atan(target[2] / target[3]) - std::atan(box[2] / box[3]);
        const double aspect = 4.0 * angle_gap * angle_gap / (K_PI * K_PI);
        const double trade_off = aspect / (1.0 - iou + aspect + Detect::K_ASPECT_TRADE_OFF_EPS);
        return iou - distance - trade_off * aspect;
    }

    /** @brief Returns the central difference of the complete IoU by each value of @p box. */
    Quad measure_difference(const Box& box, const Box& target) noexcept
    {
        const Quad truth = widen(target);
        Quad slope{};
        for (std::size_t entry = 0; entry < slope.size(); ++entry)
        {
            Quad ahead = widen(box);
            Quad behind = ahead;
            ahead[entry] += K_DIFFERENCE_STEP;
            behind[entry] -= K_DIFFERENCE_STEP;
            const double rise =
                measure_complete_iou(ahead, truth) - measure_complete_iou(behind, truth);
            slope[entry] = rise / (2.0 * K_DIFFERENCE_STEP);
        }
        return slope;
    }

    /** @brief Returns whether no edge and no overlap extent of the pair is within the gap. */
    bool is_away_from_kinks(const Box& box, const Box& target) noexcept
    {
        const Quad own = measure_edges(widen(box));
        const Quad truth = measure_edges(widen(target));
        for (std::size_t edge = 0; edge < own.size(); ++edge)
        {
            if (std::abs(own[edge] - truth[edge]) < K_KINK_GAP)
                return false;
        }
        const double across = std::min(own[1], truth[1]) - std::max(own[0], truth[0]);
        const double down = std::min(own[3], truth[3]) - std::max(own[2], truth[2]);
        return std::abs(across) >= K_KINK_GAP && std::abs(down) >= K_KINK_GAP;
    }

    /** @brief Returns the gradient of the pair under @p form. */
    Box measure_gradient(const Box& box, const Box& target, Detect::BoxGradient form)
    {
        Box gradient{};
        Detect::complete_iou_gradient(box, target, form, gradient);
        return gradient;
    }

    /** @brief Expects the analytic gradient of the pair to equal its central difference. */
    void expect_analytic_matches_difference(const Box& box, const Box& target)
    {
        ASSERT_TRUE(is_away_from_kinks(box, target));
        const Box gradient = measure_gradient(box, target, Detect::BoxGradient::Analytic);
        const Quad slope = measure_difference(box, target);
        for (std::size_t entry = 0; entry < slope.size(); ++entry)
        {
            EXPECT_NEAR(static_cast<double>(gradient[entry]), slope[entry],
                        K_ANALYTIC_ABSOLUTE + K_ANALYTIC_RELATIVE * std::abs(slope[entry]))
                << "entry " << entry;
        }
    }

    /** @brief Expects @p value to equal @p expected within the hand tolerance. */
    void expect_hand_value(float value, double expected)
    {
        EXPECT_NEAR(static_cast<double>(value), expected,
                    K_HAND_TOLERANCE + K_HAND_TOLERANCE * std::abs(expected));
    }
} // namespace

/** @brief Tests the analytic form on boxes that overlap without one holding the other. */
TEST(Unit_DetectIouGradient, AnalyticMatchesTheDifferenceOnOverlappingBoxes)
{
    expect_analytic_matches_difference({0.45f, 0.55f, 0.3f, 0.2f}, {0.55f, 0.52f, 0.42f, 0.6f});
    expect_analytic_matches_difference({0.6f, 0.4f, 0.25f, 0.35f}, {0.5f, 0.5f, 0.3f, 0.3f});
}

/** @brief Tests the analytic form on boxes apart along one axis and along both. */
TEST(Unit_DetectIouGradient, AnalyticMatchesTheDifferenceOnDisjointBoxes)
{
    expect_analytic_matches_difference({0.2f, 0.45f, 0.2f, 0.3f}, {0.7f, 0.5f, 0.25f, 0.22f});
    expect_analytic_matches_difference({0.2f, 0.2f, 0.15f, 0.25f}, {0.7f, 0.75f, 0.3f, 0.2f});
}

/** @brief Tests the analytic form on a box inside its truth and on one around it. */
TEST(Unit_DetectIouGradient, AnalyticMatchesTheDifferenceOnNestedBoxes)
{
    expect_analytic_matches_difference({0.52f, 0.47f, 0.2f, 0.1f}, {0.5f, 0.5f, 0.4f, 0.45f});
    expect_analytic_matches_difference({0.5f, 0.5f, 0.6f, 0.5f}, {0.52f, 0.47f, 0.2f, 0.3f});
}

/** @brief Tests the analytic form on 100 seeded pairs that sit on no kink of the measure. */
TEST(Unit_DetectIouGradient, AnalyticMatchesTheDifferenceOnSeededPairs)
{
    UnitSequence sequence;
    int checked = 0;
    while (checked < K_PAIR_COUNT)
    {
        Box pair[2];
        for (Box& box : pair)
        {
            box[0] = static_cast<float>(0.2 + 0.6 * sequence.next());
            box[1] = static_cast<float>(0.2 + 0.6 * sequence.next());
            box[2] = static_cast<float>(0.1 + 0.4 * sequence.next());
            box[3] = static_cast<float>(0.1 + 0.4 * sequence.next());
        }
        if (!is_away_from_kinks(pair[0], pair[1]))
            continue;
        expect_analytic_matches_difference(pair[0], pair[1]);
        ++checked;
    }
}

/** @brief Tests the compatible form against the twelve dumped deltas of the report. */
TEST(Unit_DetectIouGradient, DarknetCompatibleMatchesTheDumpedDeltas)
{
    for (const DumpedCell& cell : K_DUMPED_CELLS)
    {
        const Box gradient =
            measure_gradient(cell.box, cell.target, Detect::BoxGradient::DarknetCompatible);
        for (std::size_t entry = 0; entry < cell.delta.size(); ++entry)
        {
            EXPECT_NEAR(static_cast<double>(gradient[entry]) * K_DUMP_BOX_WEIGHT, cell.delta[entry],
                        K_DUMP_RELATIVE * std::abs(cell.delta[entry]))
                << "entry " << entry;
        }
    }
}

/** @brief Tests that the compatible form doubles the overlap part of width and height. */
TEST(Unit_DetectIouGradient, DarknetCompatibleDoublesTheSizeSlopeOfTheOverlap)
{
    // A 0.2 square centred in a 0.4 square: IoU = w * h / 0.16, so its slope by w is 1.25.
    const Box box{0.5f, 0.5f, 0.2f, 0.2f};
    const Box target{0.5f, 0.5f, 0.4f, 0.4f};
    const Box analytic = measure_gradient(box, target, Detect::BoxGradient::Analytic);
    const Box compatible = measure_gradient(box, target, Detect::BoxGradient::DarknetCompatible);

    expect_hand_value(analytic[2], 1.25);
    expect_hand_value(analytic[3], 1.25);
    expect_hand_value(compatible[2], 2.5);
    expect_hand_value(compatible[3], 2.5);
    expect_hand_value(analytic[0], 0.0);
    expect_hand_value(compatible[0], 0.0);
}

/** @brief Tests that the compatible form flips the enclosing-box term of the centre. */
TEST(Unit_DetectIouGradient, DarknetCompatibleFlipsTheEnclosingTermOfTheCentre)
{
    // Disjoint equal squares: S = 0.25, Cw = 0.7, C = 0.53; x is (0.53 -+ 0.35) / 0.2809.
    const Box box{0.2f, 0.5f, 0.2f, 0.2f};
    const Box target{0.7f, 0.5f, 0.2f, 0.2f};
    const Box analytic = measure_gradient(box, target, Detect::BoxGradient::Analytic);
    const Box compatible = measure_gradient(box, target, Detect::BoxGradient::DarknetCompatible);

    expect_hand_value(analytic[0], 0.18 / 0.2809);
    expect_hand_value(compatible[0], 0.88 / 0.2809);
    expect_hand_value(analytic[2], 0.7 * 0.25 / 0.2809);
    expect_hand_value(compatible[2], 0.7 * 0.25 / 0.2809);
    expect_hand_value(compatible[1], 0.0);
    expect_hand_value(compatible[3], 0.0);
}

/** @brief Tests that the compatible form holds the trade-off fixed and drops a divisor. */
TEST(Unit_DetectIouGradient, DarknetCompatibleSimplifiesTheAspectTerm)
{
    // A 0.2 by 0.1 box centred in a 0.4 square: IoU 0.125, w^2 + h^2 = 0.05, no distance part.
    const Box box{0.5f, 0.5f, 0.2f, 0.1f};
    const Box target{0.5f, 0.5f, 0.4f, 0.4f};
    const Box analytic = measure_gradient(box, target, Detect::BoxGradient::Analytic);
    const Box compatible = measure_gradient(box, target, Detect::BoxGradient::DarknetCompatible);

    const double angle_gap = std::atan(1.0) - std::atan(2.0);
    const double aspect = 4.0 * angle_gap * angle_gap / (K_PI * K_PI);
    const double trade_off = aspect / (1.0 - 0.125 + aspect + Detect::K_ASPECT_TRADE_OFF_EPS);
    const double slope = 8.0 / (K_PI * K_PI) * angle_gap;
    const double kept = 1.0 - trade_off * trade_off;
    const double full = trade_off * (2.0 - trade_off) * slope / 0.05;

    expect_hand_value(compatible[2], 1.25 + trade_off * slope * 0.1);
    expect_hand_value(compatible[3], 2.5 - trade_off * slope * 0.2);
    expect_hand_value(analytic[2], kept * 0.625 + full * 0.1);
    expect_hand_value(analytic[3], kept * 1.25 - full * 0.2);
}

/** @brief Tests that both forms write zeros for two boxes that are one point. */
TEST(Unit_DetectIouGradient, BothFormsAreZeroForTwoBoxesAtOnePoint)
{
    const Box point{0.5f, 0.5f, 0.0f, 0.0f};
    for (const Detect::BoxGradient form :
         {Detect::BoxGradient::Analytic, Detect::BoxGradient::DarknetCompatible})
    {
        const Box gradient = measure_gradient(point, point, form);
        for (const float component : gradient)
            EXPECT_EQ(component, 0.0f);
    }
}

/**************************************************************************/
/* anchor_delta_support.hpp                                               */
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
 * @file anchor_delta_support.hpp
 * @brief Defines the complete IoU and its central difference the anchor delta tests expect.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <SushiAI/vision/detect/iou_gradient.hpp>

/** @brief Holds what the tests of an anchor head's box delta compute their expectations with. */
namespace AnchorDeltaSupport
{
    /** @brief Holds a box or its four deltas in double: x, y, width, height. */
    using Quad = std::array<double, 4>;

    /** @brief Holds the number of box entries of a slot. */
    constexpr std::size_t K_BOX_ENTRIES = 4;

    /** @brief Holds the ratio of a circle's circumference to its diameter. */
    constexpr double K_PI = 3.14159265358979323846;

    /** @brief Holds the step of the central difference the expected gradient is taken with. */
    constexpr double K_DIFFERENCE_STEP = 1e-6;

    /** @brief Returns the complete IoU of two centre-and-size boxes, by SPEC 4.6. */
    inline double measure_complete_iou(const Quad& box, const Quad& target)
    {
        const double left = std::max(box[0] - box[2] / 2.0, target[0] - target[2] / 2.0);
        const double right = std::min(box[0] + box[2] / 2.0, target[0] + target[2] / 2.0);
        const double top = std::max(box[1] - box[3] / 2.0, target[1] - target[3] / 2.0);
        const double bottom = std::min(box[1] + box[3] / 2.0, target[1] + target[3] / 2.0);
        const double shared =
            right > left && bottom > top ? (right - left) * (bottom - top) : 0.0;
        const double iou = shared / (box[2] * box[3] + target[2] * target[3] - shared);

        const double hull_width = std::max(box[0] + box[2] / 2.0, target[0] + target[2] / 2.0) -
                                  std::min(box[0] - box[2] / 2.0, target[0] - target[2] / 2.0);
        const double hull_height = std::max(box[1] + box[3] / 2.0, target[1] + target[3] / 2.0) -
                                   std::min(box[1] - box[3] / 2.0, target[1] - target[3] / 2.0);
        const double shift_x = target[0] - box[0];
        const double shift_y = target[1] - box[1];
        const double distance = (shift_x * shift_x + shift_y * shift_y) /
                                (hull_width * hull_width + hull_height * hull_height);

        const double angle_gap = std::atan(target[2] / target[3]) - std::atan(box[2] / box[3]);
        const double aspect = 4.0 * angle_gap * angle_gap / (K_PI * K_PI);
        const double trade_off =
            aspect / (1.0 - iou + aspect + SushiAI::Vision::Detect::K_ASPECT_TRADE_OFF_EPS);
        return iou - distance - trade_off * aspect;
    }

    /** @brief Returns @p weight times the central difference of the complete IoU by @p box. */
    inline Quad expect_box_delta(const Quad& box, const Quad& target, double weight)
    {
        Quad delta{};
        for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
        {
            Quad ahead = box;
            Quad behind = box;
            ahead[entry] += K_DIFFERENCE_STEP;
            behind[entry] -= K_DIFFERENCE_STEP;
            delta[entry] = weight *
                           (measure_complete_iou(ahead, target) -
                            measure_complete_iou(behind, target)) /
                           (2.0 * K_DIFFERENCE_STEP);
        }
        return delta;
    }

} // namespace AnchorDeltaSupport

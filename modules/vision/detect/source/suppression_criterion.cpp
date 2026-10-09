/**************************************************************************/
/* suppression_criterion.cpp                                              */
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
 * @file suppression_criterion.cpp
 * @brief Implements the suppression measures of the one-to-many postprocess.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/suppression_criterion.hpp>

#include <algorithm>
#include <cmath>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Returns the IoU of two x1 y1 x2 y2 boxes, with no epsilon term. */
            float plain_iou(const float* a, const float* b) noexcept
            {
                const float across = std::min(a[2], b[2]) - std::max(a[0], b[0]);
                const float down = std::min(a[3], b[3]) - std::max(a[1], b[1]);
                const float intersection =
                    (across < 0.0f ? 0.0f : across) * (down < 0.0f ? 0.0f : down);
                const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
                const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
                const float joined = area_a + area_b - intersection;
                if (intersection == 0.0f || joined == 0.0f)
                    return 0.0f;
                return intersection / joined;
            }

            /** @brief Returns `(d / c)^exponent` for two x1 y1 x2 y2 boxes, 0 when `c` is 0. */
            float distance_term(const float* a, const float* b, float exponent) noexcept
            {
                const float shift_x = (a[0] + a[2]) - (b[0] + b[2]);
                const float shift_y = (a[1] + a[3]) - (b[1] + b[3]);
                const float centre_distance = (shift_x * shift_x + shift_y * shift_y) * 0.25f;

                const float hull_width = std::max(a[2], b[2]) - std::min(a[0], b[0]);
                const float hull_height = std::max(a[3], b[3]) - std::min(a[1], b[1]);
                const float diagonal = hull_width * hull_width + hull_height * hull_height;
                if (diagonal == 0.0f)
                    return 0.0f;
                return std::pow(centre_distance / diagonal, exponent);
            }
        } // namespace

        float measure_suppression(SushiRuntime::span<const float> box1,
                                  SushiRuntime::span<const float> box2,
                                  SuppressionCriterion criterion,
                                  float distance_exponent) noexcept
        {
            const float* a = box1.data();
            const float* b = box2.data();
            const float overlap = plain_iou(a, b);
            if (criterion == SuppressionCriterion::Iou)
                return overlap;
            return overlap - distance_term(a, b, distance_exponent);
        }

    } // namespace Vision::Detect
} // namespace SushiAI

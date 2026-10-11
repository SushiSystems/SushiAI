/**************************************************************************/
/* suppression_criterion.hpp                                              */
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
 * @file suppression_criterion.hpp
 * @brief Declares the suppression criteria of the one-to-many postprocess and their measure.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Holds the exponent of the distance term when a config does not set one. */
        inline constexpr float K_DEFAULT_DISTANCE_EXPONENT = 0.6f;

        /**
         * @brief Selects the measure that decides whether a box suppresses a lower-scoring one.
         * @see modules/vision/detect/README.md
         */
        enum class SuppressionCriterion : std::uint8_t
        {
            IoU,         /**< The intersection over union of the two boxes. */
            DistanceIoU, /**< The IoU minus the distance term, see measure_suppression. */
        };

        /**
         * @brief Returns the suppression measure of two x1 y1 x2 y2 boxes under @p criterion.
         * @param distance_exponent Read by DistanceIoU only.
         * @return The IoU, 0 when the intersection or the union is 0. DistanceIoU subtracts
         *         `(d / c)^distance_exponent`: `d` the squared distance of the centres, `c` the
         *         squared diagonal of the smallest enclosing box; nothing when `c` is 0.
         */
        [[nodiscard]] float measure_suppression(SushiRuntime::span<const float> box1,
                                                SushiRuntime::span<const float> box2,
                                                SuppressionCriterion criterion,
                                                float distance_exponent) noexcept;

    } // namespace Vision::Detect
} // namespace SushiAI

/**************************************************************************/
/* iou_gradient.hpp                                                       */
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
 * @file iou_gradient.hpp
 * @brief Declares the gradient of complete IoU with respect to a centre-and-size box.
 * @author Mustafa Garip
 * @see docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Holds the term that keeps the trade-off of the aspect penalty finite. */
        inline constexpr double K_ASPECT_TRADE_OFF_EPS = 1e-6;

        /**
         * @brief Selects which gradient of complete IoU a box loss descends along.
         * @see docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md
         */
        enum class BoxGradient : std::uint8_t
        {
            Analytic,          /**< The derivative of the measure. */
            DarknetCompatible, /**< The gradient section 11 of the report states. */
            FixedTradeOff,     /**< The derivative with the aspect trade-off held constant. */
        };

        /**
         * @brief Writes the gradient of the complete IoU of @p box and @p target at @p box.
         * @param box The predicted box, `x_centre y_centre width height`, as is @p target.
         * @param gradient Receives four components in the order of @p box; the measure rises
         *        along them. All are 0 when both boxes are one point.
         * @pre Each span holds four values.
         */
        void complete_iou_gradient(SushiRuntime::span<const float> box,
                                   SushiRuntime::span<const float> target, BoxGradient form,
                                   SushiRuntime::span<float> gradient) noexcept;

    } // namespace Vision::Detect
} // namespace SushiAI

/**************************************************************************/
/* box_coding.hpp                                                         */
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
 * @file box_coding.hpp
 * @brief Declares the coding of boxes as left, top, right, bottom distances about anchor points.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /**
         * @brief Decodes `[B, A, 4]` distances about `[A, 2]` anchor points into x1 y1 x2 y2.
         * @param boxes Receives `[B, A, 4]` in the units of @p anchor_points; may be @p distances.
         * @pre The spans hold the [B, A, 4] and [A, 2] sizes their names give, and the output
         *      holds as many values as the boxes.
         * @see modules/vision/detect/README.md
         */
        void decode_xyxy(SushiRuntime::span<const float> distances,
                         SushiRuntime::span<const float> anchor_points,
                         SushiRuntime::span<float> boxes) noexcept;

        /**
         * @brief Decodes `[B, A, 4]` distances about `[A, 2]` anchor points into centre and size.
         * @param boxes Receives `[B, A, 4]` in the units of @p anchor_points; may be @p distances.
         * @pre The spans hold the [B, A, 4] and [A, 2] sizes their names give, and the output
         *      holds as many values as the boxes.
         * @see modules/vision/detect/README.md
         */
        void decode_xywh(SushiRuntime::span<const float> distances,
                         SushiRuntime::span<const float> anchor_points,
                         SushiRuntime::span<float> boxes) noexcept;

        /**
         * @brief Encodes `[B, A, 4]` x1 y1 x2 y2 boxes as distances about `[A, 2]` anchor points.
         * @param distances Receives `[B, A, 4]`, not clamped; may be @p boxes.
         * @pre The spans hold the [B, A, 4] and [A, 2] sizes their names give, and the output
         *      holds as many values as the boxes.
         * @see modules/vision/detect/README.md
         */
        void encode_distances(SushiRuntime::span<const float> anchor_points,
                              SushiRuntime::span<const float> boxes,
                              SushiRuntime::span<float> distances) noexcept;

        /**
         * @brief Multiplies `[B, A, 4]` boxes by the `[A]` stride of their anchor.
         * @param scaled Receives `[B, A, 4]`; may be @p boxes.
         * @pre The spans hold the [B, A, 4] and [A, 2] sizes their names give, and the output
         *      holds as many values as the boxes.
         * @see modules/vision/detect/README.md
         */
        void scale_boxes(SushiRuntime::span<const float> boxes,
                         SushiRuntime::span<const float> strides,
                         SushiRuntime::span<float> scaled) noexcept;

    } // namespace Vision::Detect::Free
} // namespace SushiAI

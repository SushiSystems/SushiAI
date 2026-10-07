/**************************************************************************/
/* iou.hpp                                                                */
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
 * @file iou.hpp
 * @brief Declares intersection over union and complete IoU of x1 y1 x2 y2 boxes in float32.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Detect
    {
        /** @brief Holds the term that keeps the IoU divisions away from zero. */
        inline constexpr float K_IOU_EPS = 1e-7f;

        /**
         * @brief Returns the IoU of two boxes.
         * @pre Each span holds the four values of one box.
         * @see include/SushiAI/detect/README.md
         */
        [[nodiscard]] float intersection_over_union(SushiRuntime::span<const float> box1,
                                                    SushiRuntime::span<const float> box2) noexcept;

        /**
         * @brief Returns the complete IoU of two boxes, which is at most their IoU.
         * @pre Each span holds the four values of one box.
         * @see include/SushiAI/detect/README.md
         */
        [[nodiscard]] float complete_iou(SushiRuntime::span<const float> box1,
                                         SushiRuntime::span<const float> box2) noexcept;

        /**
         * @brief Writes the IoU of row i of @p boxes1 with row i of @p boxes2.
         * @param ious Receives `[N]` for two `[N, 4]` inputs.
         */
        void elementwise_iou(SushiRuntime::span<const float> boxes1,
                             SushiRuntime::span<const float> boxes2,
                             SushiRuntime::span<float> ious);

        /**
         * @brief Writes the complete IoU of row i of @p boxes1 with row i of @p boxes2.
         * @param ious Receives `[N]` for two `[N, 4]` inputs.
         */
        void elementwise_complete_iou(SushiRuntime::span<const float> boxes1,
                                      SushiRuntime::span<const float> boxes2,
                                      SushiRuntime::span<float> ious);

        /**
         * @brief Writes the IoU of every row of @p boxes1 with every row of @p boxes2.
         * @param ious Receives `[N, M]` for `[N, 4]` and `[M, 4]` inputs.
         * @see include/SushiAI/detect/README.md
         */
        void pairwise_iou(SushiRuntime::span<const float> boxes1,
                          SushiRuntime::span<const float> boxes2,
                          SushiRuntime::span<float> ious);

    } // namespace Detect
} // namespace SushiAI

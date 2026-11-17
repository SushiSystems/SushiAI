/**************************************************************************/
/* yolox_decode.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file yolox_decode.hpp
 * @brief Declares the decoding of a YOLOX head's outputs into boxes, scores and detections.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Yolox
    {
        /** @brief Views the three outputs of one head level, each laid out NHWC. */
        struct YoloxLevel
        {
            std::int64_t height = 0; /**< Rows of the level's grid, H. */
            std::int64_t width = 0;  /**< Columns of the level's grid, W. */
            float stride = 0.0f;     /**< Input pixels per grid cell; positive. */

            /** @brief Holds the `[N, H, W, C]` class logits. */
            SushiRuntime::span<const float> classes;

            /** @brief Holds the `[N, H, W, 4]` offsets x and y, then log width and height. */
            SushiRuntime::span<const float> boxes;

            /** @brief Holds the `[N, H, W, 1]` objectness logits. */
            SushiRuntime::span<const float> objectness;
        };

        /** @brief Holds the boxes and scores of every anchor of every level. */
        struct YoloxDecoded
        {
            /** @brief Holds the extents of the two arrays below. */
            PredictionShape shape;

            /** @brief Holds `[B, A, 4]` centre x, centre y, width, height, in input pixels. */
            std::vector<float> boxes;

            /** @brief Holds `[B, A, C]`, sigmoid(objectness) times sigmoid(class logit). */
            std::vector<float> scores;
        };

        /**
         * @brief Decodes the head outputs of a batch, level by level, row by row: the cell
         *        (gx, gy) of stride s with raw box (px, py, pw, ph) gives centre ((gx + px) * s,
         *        (gy + py) * s) and size (exp(pw) * s, exp(ph) * s).
         * @param levels The levels, finest first; @p batch and @p classes are positive.
         * @return InvalidDescription for no level, a non-positive extent or stride, or a span
         *         that does not fit its level; @p decoded is unchanged then.
         */
        [[nodiscard]] Status decode_yolox(SushiRuntime::span<const YoloxLevel> levels,
                                          std::int64_t batch, std::int64_t classes,
                                          YoloxDecoded& decoded) noexcept;

        /**
         * @brief Returns the suppression settings of @p task: the best class of each anchor, and
         *        boxes of different classes keep apart.
         */
        [[nodiscard]] NMSConfig make_yolox_nms_config(const YoloxTask& task) noexcept;

        /**
         * @brief Decodes the head outputs and keeps what thresholding and suppression leave.
         * @param detections Receives one `[n, 6]` array per image, x1 y1 x2 y2 score class,
         *        highest score first.
         * @return The failure of decode_yolox() or of select_nms_detections().
         */
        [[nodiscard]] Status select_yolox_detections(
            SushiRuntime::span<const YoloxLevel> levels, std::int64_t batch, std::int64_t classes,
            const YoloxTask& task, std::vector<std::vector<float>>& detections) noexcept;

    } // namespace Vision::Detect::Yolox
} // namespace SushiAI

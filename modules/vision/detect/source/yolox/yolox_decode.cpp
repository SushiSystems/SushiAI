/**************************************************************************/
/* yolox_decode.cpp                                                       */
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
 * @file yolox_decode.cpp
 * @brief Implements the decoding of a YOLOX head's outputs into boxes, scores and detections.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/yolox/yolox_decode.hpp>

#include <cmath>
#include <cstddef>
#include <utility>

#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Yolox
    {
        namespace
        {
            /** @brief Returns the logistic function of @p logit. */
            [[nodiscard]] float sigmoid(float logit) noexcept
            {
                return 1.0f / (1.0f + std::exp(-logit));
            }

            /** @brief Returns the number of cells of @p level in one image. */
            [[nodiscard]] std::int64_t cells_of(const YoloxLevel& level) noexcept
            {
                return level.height * level.width;
            }

            /** @brief Checks that @p level and the three spans of its batch agree. */
            Status check_level(const YoloxLevel& level, std::size_t index, std::int64_t batch,
                               std::int64_t classes)
            {
                SA_FAIL_IF(level.height < 1 || level.width < 1, StatusCode::InvalidDescription,
                           "yolox level {} has the grid {} by {}; both must be positive", index,
                           level.height, level.width);
                SA_FAIL_IF(!(level.stride > 0.0f) || !std::isfinite(level.stride),
                           StatusCode::InvalidDescription,
                           "yolox level {} has the stride {}; it must be positive and finite",
                           index, level.stride);

                const std::int64_t cells = batch * cells_of(level);
                SA_FAIL_IF(static_cast<std::int64_t>(level.classes.size()) != cells * classes,
                           StatusCode::InvalidDescription,
                           "yolox level {} holds {} class values; {} cells of {} classes need {}",
                           index, level.classes.size(), cells, classes, cells * classes);
                SA_FAIL_IF(static_cast<std::int64_t>(level.boxes.size())
                               != cells * static_cast<std::int64_t>(K_YOLOX_BOX_CHANNELS),
                           StatusCode::InvalidDescription,
                           "yolox level {} holds {} box values; {} cells need {}", index,
                           level.boxes.size(), cells,
                           cells * static_cast<std::int64_t>(K_YOLOX_BOX_CHANNELS));
                SA_FAIL_IF(static_cast<std::int64_t>(level.objectness.size()) != cells,
                           StatusCode::InvalidDescription,
                           "yolox level {} holds {} objectness values; {} cells need {}", index,
                           level.objectness.size(), cells, cells);
                return Status();
            }

            /** @brief Writes the box and the scores of the cell at @p cell of @p level. */
            void decode_cell(const YoloxLevel& level, std::int64_t cell, std::int64_t classes,
                             float* box, float* scores) noexcept
            {
                const std::int64_t row = cell % cells_of(level);
                const float grid_x = static_cast<float>(row % level.width);
                const float grid_y = static_cast<float>(row / level.width);
                const float* raw = level.boxes.data() + cell * K_YOLOX_BOX_CHANNELS;

                box[0] = (grid_x + raw[0]) * level.stride;
                box[1] = (grid_y + raw[1]) * level.stride;
                box[2] = std::exp(raw[2]) * level.stride;
                box[3] = std::exp(raw[3]) * level.stride;

                const float objectness = sigmoid(level.objectness[static_cast<std::size_t>(cell)]);
                const float* logits = level.classes.data() + cell * classes;
                for (std::int64_t at = 0; at < classes; ++at)
                    scores[at] = objectness * sigmoid(logits[at]);
            }
        } // namespace

        Status decode_yolox(SushiRuntime::span<const YoloxLevel> levels, std::int64_t batch,
                            std::int64_t classes, YoloxDecoded& decoded) noexcept
        {
            SA_FAIL_IF(levels.empty(), StatusCode::InvalidDescription,
                       "yolox decoding needs at least one level and was given none");
            SA_FAIL_IF(batch < 1 || classes < 1, StatusCode::InvalidDescription,
                       "yolox decoding needs a positive batch and class count; it was given {} "
                       "and {}",
                       batch, classes);

            std::int64_t anchors = 0;
            for (std::size_t index = 0; index < levels.size(); ++index)
            {
                SA_TRY(check_level(levels[index], index, batch, classes));
                anchors += cells_of(levels[index]);
            }

            YoloxDecoded result;
            result.shape = PredictionShape{batch, anchors, classes};
            result.boxes.resize(static_cast<std::size_t>(batch * anchors * K_YOLOX_BOX_CHANNELS));
            result.scores.resize(static_cast<std::size_t>(batch * anchors * classes));

            for (std::int64_t image = 0; image < batch; ++image)
            {
                std::int64_t anchor = image * anchors;
                for (const YoloxLevel& level : levels)
                {
                    const std::int64_t cells = cells_of(level);
                    for (std::int64_t row = 0; row < cells; ++row, ++anchor)
                        decode_cell(level, image * cells + row, classes,
                                    result.boxes.data() + anchor * K_YOLOX_BOX_CHANNELS,
                                    result.scores.data() + anchor * classes);
                }
            }

            decoded = std::move(result);
            return Status();
        }

        NMSConfig make_yolox_nms_config(const YoloxTask& task) noexcept
        {
            NMSConfig config;
            config.confidence_threshold = task.score_threshold;
            config.iou_threshold = task.suppression.threshold;
            config.label_mode = LabelMode::BestClass;
            config.class_mode = ClassMode::ClassAware;
            config.criterion = task.suppression.criterion;
            config.distance_exponent = task.suppression.distance_exponent;

            return config;
        }

        Status select_yolox_detections(SushiRuntime::span<const YoloxLevel> levels,
                                       std::int64_t batch, std::int64_t classes,
                                       const YoloxTask& task,
                                       std::vector<std::vector<float>>& detections) noexcept
        {
            YoloxDecoded decoded;
            SA_TRY(decode_yolox(levels, batch, classes, decoded));

            return select_nms_detections(
                SushiRuntime::span<const float>(decoded.boxes.data(), decoded.boxes.size()),
                SushiRuntime::span<const float>(decoded.scores.data(), decoded.scores.size()),
                decoded.shape, make_yolox_nms_config(task), detections);
        }

    } // namespace Vision::Detect::Yolox
} // namespace SushiAI

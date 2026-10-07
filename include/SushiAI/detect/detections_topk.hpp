/**************************************************************************/
/* detections_topk.hpp                                                    */
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
 * @file detections_topk.hpp
 * @brief Declares the top-k postprocess of the one-to-one head.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/detect/detection_row.hpp>
#include <SushiAI/detect/prediction_shape.hpp>

namespace SushiAI
{
    namespace Detect
    {
        /**
         * @brief Returns the rows per image the top-k postprocess writes: min(limit, A).
         * @param max_detections Limit on rows per image; K_MAX_DETECTIONS in the reference.
         */
        [[nodiscard]] std::int64_t count_top_detections(const PredictionShape& shape,
                                                        std::int64_t max_detections);

        /**
         * @brief Writes the highest-scoring anchor and class pairs of each image as detections.
         * @param boxes `[B, A, 4]` x1 y1 x2 y2.
         * @param scores `[B, A, C]` class probabilities; no NaN.
         * @param max_detections Limit on rows per image; K_MAX_DETECTIONS in the reference.
         * @param detections Receives `[B, k, 6]`, highest score first, k from count_top_detections.
         * @see include/SushiAI/detect/README.md
         */
        void select_top_detections(SushiRuntime::span<const float> boxes,
                                   SushiRuntime::span<const float> scores,
                                   const PredictionShape& shape,
                                   std::int64_t max_detections,
                                   SushiRuntime::span<float> detections);

        /**
         * @brief Writes each image's rows of @p detections whose score exceeds the threshold.
         * @param detections `[B, k, 6]` as select_top_detections writes them.
         * @pre k is at most the validator's row limit, K_MAX_DETECTIONS in the reference.
         * @param batch Images, B; must be positive and divide the rows of @p detections.
         * @param kept Receives one `[n, 6]` array per image, rows in their order in @p detections.
         * @see include/SushiAI/detect/README.md
         */
        void keep_confident_detections(SushiRuntime::span<const float> detections,
                                       std::int64_t batch,
                                       float confidence_threshold,
                                       std::vector<std::vector<float>>& kept);

    } // namespace Detect
} // namespace SushiAI

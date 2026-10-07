/**************************************************************************/
/* detections_nms.hpp                                                     */
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
 * @file detections_nms.hpp
 * @brief Declares the non-maximum-suppression postprocess of the one-to-many head.
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
         * @brief Selects how many classes one anchor may contribute.
         * @see include/SushiAI/detect/README.md
         */
        enum class LabelMode : std::uint8_t
        {
            BestClass,  /**< The anchor's highest-scoring class only. */
            MultiLabel, /**< Every class whose score passes the confidence threshold. */
        };

        /**
         * @brief Selects whether boxes of different classes suppress each other.
         * @see include/SushiAI/detect/README.md
         */
        enum class ClassMode : std::uint8_t
        {
            ClassAware,    /**< A box is suppressed only by a box of its own class. */
            ClassAgnostic, /**< A box is suppressed by any overlapping box. */
        };

        /**
         * @brief Holds the settings of the suppression, at the values the validator passes.
         * @see include/SushiAI/detect/README.md
         */
        struct NmsConfig
        {
            float confidence_threshold = K_VALIDATION_CONFIDENCE; /**< A score must exceed it. */
            float iou_threshold = 0.7f;                           /**< A larger IoU suppresses. */
            LabelMode label_mode = LabelMode::MultiLabel;         /**< Classes per anchor. */
            ClassMode class_mode = ClassMode::ClassAware;         /**< Whether classes interact. */
            std::int64_t max_detections = K_MAX_DETECTIONS;       /**< Limit on rows per image. */
            std::int64_t max_candidates = 30000;                  /**< Limit on boxes in NMS. */
            float class_offset = 7680.0f;                         /**< Shift between classes. */
        };

        /**
         * @brief Writes the indices of the boxes that survive suppression, highest score first.
         * @param boxes `[N, 4]` x1 y1 x2 y2.
         * @param scores `[N]`; equal scores are visited by ascending index.
         * @see include/SushiAI/detect/README.md
         */
        void suppress_overlaps(SushiRuntime::span<const float> boxes,
                               SushiRuntime::span<const float> scores,
                               float iou_threshold,
                               std::vector<std::int64_t>& kept);

        /**
         * @brief Writes each image's detections after thresholding and suppression.
         * @param boxes `[B, A, 4]` centre x, centre y, width, height.
         * @param scores `[B, A, C]` class probabilities.
         * @param detections Receives one `[n, 6]` array per image, highest score first.
         * @see include/SushiAI/detect/README.md
         */
        void select_nms_detections(SushiRuntime::span<const float> boxes,
                                   SushiRuntime::span<const float> scores,
                                   const PredictionShape& shape,
                                   const NmsConfig& config,
                                   std::vector<std::vector<float>>& detections);

    } // namespace Detect
} // namespace SushiAI

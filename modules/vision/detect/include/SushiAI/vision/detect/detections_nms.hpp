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
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Selects how many classes one anchor may contribute.
         * @see modules/vision/detect/README.md
         */
        enum class LabelMode : std::uint8_t
        {
            BestClass,  /**< The anchor's highest-scoring class only. */
            MultiLabel, /**< Every class whose score passes the confidence threshold. */
        };

        /**
         * @brief Selects whether boxes of different classes suppress each other.
         * @see modules/vision/detect/README.md
         */
        enum class ClassMode : std::uint8_t
        {
            ClassAware,    /**< A box is suppressed only by a box of its own class. */
            ClassAgnostic, /**< A box is suppressed by any overlapping box. */
        };

        /**
         * @brief Holds the settings of the suppression, at the values the validator passes.
         * @see modules/vision/detect/README.md
         */
        struct NMSConfig
        {
            float confidence_threshold = K_VALIDATION_CONFIDENCE; /**< A score must exceed it. */
            float iou_threshold = 0.7f;                           /**< Larger measures suppress. */
            LabelMode label_mode = LabelMode::MultiLabel;         /**< Classes per anchor. */
            ClassMode class_mode = ClassMode::ClassAware;         /**< Whether classes interact. */
            std::int64_t max_detections = K_MAX_DETECTIONS;       /**< Limit on rows per image. */
            std::int64_t max_candidates = 30000;                  /**< Limit on boxes in NMS. */
            float class_offset = 7680.0f;                         /**< Shift between classes. */
            SuppressionCriterion criterion = SuppressionCriterion::IoU; /**< The measure. */
            float distance_exponent = K_DEFAULT_DISTANCE_EXPONENT;      /**< DistanceIoU only. */
        };

        /**
         * @brief Writes the indices of the boxes that survive suppression, highest score first.
         * @param boxes `[N, 4]` x1 y1 x2 y2.
         * @param scores `[N]`; equal scores are visited by ascending index, and a score of 0
         *        is never kept.
         * @pre @p boxes holds four values per score.
         * @see modules/vision/detect/README.md
         */
        void suppress_overlaps(SushiRuntime::span<const float> boxes,
                               SushiRuntime::span<const float> scores,
                               float iou_threshold,
                               std::vector<std::int64_t>& kept) noexcept;

        /**
         * @brief Writes the indices of the boxes that survive suppression under @p criterion.
         * @param boxes `[N, 4]` x1 y1 x2 y2.
         * @param scores `[N]`; a score of 0 is never kept and suppresses nothing.
         * @param threshold A measure above it suppresses the lower-scoring box.
         * @param kept Receives the survivors, highest score first; equal scores by index.
         * @pre @p boxes holds four values per score and @p criterion is IoU or DistanceIoU.
         */
        void suppress_overlaps(SushiRuntime::span<const float> boxes,
                               SushiRuntime::span<const float> scores,
                               float threshold,
                               SuppressionCriterion criterion,
                               float distance_exponent,
                               std::vector<std::int64_t>& kept) noexcept;

        /**
         * @brief Writes each image's detections after thresholding and suppression.
         * @param boxes `[B, A, 4]` centre x, centre y, width, height.
         * @param scores `[B, A, C]` class probabilities.
         * @param detections Receives one `[n, 6]` array per image, highest score first.
         * @return InvalidDescription for an extent that is not positive, arrays that do not match
         *         the shape, a threshold outside [0, 1] or a limit below 1.
         */
        [[nodiscard]] Status select_nms_detections(SushiRuntime::span<const float> boxes,
                                                   SushiRuntime::span<const float> scores,
                                                   const PredictionShape& shape,
                                                   const NMSConfig& config,
                                                   std::vector<std::vector<float>>& detections) noexcept;

    } // namespace Vision::Detect
} // namespace SushiAI

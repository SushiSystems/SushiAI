/**************************************************************************/
/* detections_nms.cpp                                                     */
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
 * @file detections_nms.cpp
 * @brief Implements the non-maximum-suppression postprocess of the one-to-many head.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/detections_nms.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/float_order.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds the boxes that passed the confidence threshold, one row per label. */
            struct Candidates
            {
                std::vector<float> boxes;   /**< `[n, 4]` x1 y1 x2 y2. */
                std::vector<float> scores;  /**< `[n]` score of the row's class. */
                std::vector<float> classes; /**< `[n]` class index as a float. */

                /** @brief Appends one row. */
                void add(const float* box, float score, std::size_t class_index)
                {
                    boxes.insert(boxes.end(), box, box + K_BOX_VALUES);
                    scores.push_back(score);
                    classes.push_back(static_cast<float>(class_index));
                }
            };

            /** @brief Returns the rows of @p source named by @p order, in that order. */
            Candidates reorder(const Candidates& source, const std::vector<std::int64_t>& order)
            {
                Candidates sorted;
                sorted.boxes.reserve(order.size() * K_BOX_VALUES);
                sorted.scores.reserve(order.size());
                sorted.classes.reserve(order.size());
                for (const std::int64_t index : order)
                {
                    const std::size_t row = static_cast<std::size_t>(index);
                    sorted.add(source.boxes.data() + row * K_BOX_VALUES, source.scores[row],
                               static_cast<std::size_t>(source.classes[row]));
                }
                return sorted;
            }

            /** @brief Returns the rows of one image whose score exceeds the threshold. */
            Candidates collect_candidates(SushiRuntime::span<const float> boxes,
                                          SushiRuntime::span<const float> scores,
                                          std::size_t classes,
                                          const NMSConfig& config)
            {
                const bool every_class = config.label_mode == LabelMode::MultiLabel && classes > 1;
                const std::size_t anchors = boxes.size() / K_BOX_VALUES;
                Candidates candidates;
                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                {
                    const float* row = scores.data() + anchor * classes;
                    const float* best = row;
                    for (const float* at = row + 1; at < row + classes; ++at)
                    {
                        if (ranks_above(*at, *best))
                            best = at;
                    }
                    if (!(*best > config.confidence_threshold))
                        continue;

                    float corners[K_BOX_VALUES];
                    xywh_to_xyxy(boxes.subspan(anchor * K_BOX_VALUES, K_BOX_VALUES),
                                 SushiRuntime::span<float>(corners, K_BOX_VALUES));
                    if (!every_class)
                    {
                        candidates.add(corners, *best, static_cast<std::size_t>(best - row));
                        continue;
                    }
                    for (std::size_t column = 0; column < classes; ++column)
                    {
                        if (row[column] > config.confidence_threshold)
                            candidates.add(corners, row[column], column);
                    }
                }
                return candidates;
            }

            /** @brief Returns the boxes of @p candidates, each moved by class times @p offset. */
            std::vector<float> shift_by_class(const Candidates& candidates, float offset)
            {
                std::vector<float> shifted(candidates.boxes.size());
                for (std::size_t row = 0; row < candidates.scores.size(); ++row)
                {
                    const float shift = candidates.classes[row] * offset;
                    for (std::size_t column = 0; column < K_BOX_VALUES; ++column)
                    {
                        shifted[row * K_BOX_VALUES + column] =
                            candidates.boxes[row * K_BOX_VALUES + column] + shift;
                    }
                }
                return shifted;
            }

            /** @brief Writes the detections of one image's `[A, 4]` boxes and `[A, C]` scores. */
            void select_image_detections(SushiRuntime::span<const float> boxes,
                                         SushiRuntime::span<const float> scores,
                                         std::size_t classes,
                                         const NMSConfig& config,
                                         std::vector<float>& detections)
            {
                detections.clear();
                Candidates candidates = collect_candidates(boxes, scores, classes, config);
                if (candidates.scores.empty())
                    return;

                std::vector<std::int64_t> order;
                if (candidates.scores.size() > static_cast<std::size_t>(config.max_candidates))
                {
                    select_top_k(candidates.scores, config.max_candidates, order);
                    candidates = reorder(candidates, order);
                }

                const float offset =
                    config.class_mode == ClassMode::ClassAgnostic ? 0.0f : config.class_offset;
                suppress_overlaps(shift_by_class(candidates, offset), candidates.scores,
                                  config.iou_threshold, config.criterion,
                                  config.distance_exponent, order);
                if (order.size() > static_cast<std::size_t>(config.max_detections))
                    order.resize(static_cast<std::size_t>(config.max_detections));

                detections.reserve(order.size() * K_DETECTION_VALUES);
                for (const std::int64_t index : order)
                {
                    const std::size_t row = static_cast<std::size_t>(index);
                    const float* box = candidates.boxes.data() + row * K_BOX_VALUES;
                    detections.insert(detections.end(), box, box + K_BOX_VALUES);
                    detections.push_back(candidates.scores[row]);
                    detections.push_back(candidates.classes[row]);
                }
            }
        } // namespace

        void suppress_overlaps(SushiRuntime::span<const float> boxes,
                               SushiRuntime::span<const float> scores,
                               float iou_threshold,
                               std::vector<std::int64_t>& kept) noexcept
        {
            const std::size_t count = scores.size();
            SA_FATAL_IF(boxes.size() != count * K_BOX_VALUES,
                        "a suppression needs [N, 4] boxes and [N] scores; got {} and {} values",
                        boxes.size(), count);
            suppress_overlaps(boxes, scores, iou_threshold, SuppressionCriterion::IoU,
                              K_DEFAULT_DISTANCE_EXPONENT, kept);
        }

        void suppress_overlaps(SushiRuntime::span<const float> boxes,
                               SushiRuntime::span<const float> scores,
                               float threshold,
                               SuppressionCriterion criterion,
                               float distance_exponent,
                               std::vector<std::int64_t>& kept) noexcept
        {
            const std::size_t count = scores.size();
            SA_FATAL_IF(boxes.size() != count * K_BOX_VALUES,
                        "a suppression needs [N, 4] boxes and [N] scores; got {} and {} values",
                        boxes.size(), count);
            SA_FATAL_IF(criterion != SuppressionCriterion::IoU &&
                            criterion != SuppressionCriterion::DistanceIoU,
                        "a suppression criterion must be IoU or DistanceIoU; got {}",
                        static_cast<unsigned>(criterion));

            std::vector<std::int64_t> order;
            select_top_k(scores, static_cast<std::int64_t>(count), order);

            // Rows are flagged by their rank in `order`; a zero score is skipped, never kept.
            std::vector<std::uint8_t> removed(count, 0);
            for (std::size_t rank = 0; rank < count; ++rank)
            {
                if (scores[static_cast<std::size_t>(order[rank])] == 0.0f)
                    removed[rank] = 1;
            }

            kept.clear();
            for (std::size_t rank = 0; rank < count; ++rank)
            {
                if (removed[rank] != 0)
                    continue;

                const std::int64_t current = order[rank];
                kept.push_back(current);
                const SushiRuntime::span<const float> current_box =
                    boxes.subspan(static_cast<std::size_t>(current) * K_BOX_VALUES, K_BOX_VALUES);
                for (std::size_t later = rank + 1; later < count; ++later)
                {
                    if (removed[later] != 0)
                        continue;

                    const SushiRuntime::span<const float> later_box = boxes.subspan(
                        static_cast<std::size_t>(order[later]) * K_BOX_VALUES, K_BOX_VALUES);
                    const float measure =
                        measure_suppression(current_box, later_box, criterion, distance_exponent);
                    // A NaN measure fails <= and so suppresses.
                    if (!(measure <= threshold))
                        removed[later] = 1;
                }
            }
        }

        Status select_nms_detections(SushiRuntime::span<const float> boxes,
                                     SushiRuntime::span<const float> scores,
                                     const PredictionShape& shape,
                                     const NMSConfig& config,
                                     std::vector<std::vector<float>>& detections) noexcept
        {
            SA_FAIL_IF(shape.batch <= 0 || shape.anchors <= 0 || shape.num_classes <= 0,
                       StatusCode::InvalidDescription,
                       "a suppression postprocess needs positive extents; got batch {}, {} "
                       "anchors and {} classes",
                       shape.batch, shape.anchors, shape.num_classes);
            const std::size_t batch = static_cast<std::size_t>(shape.batch);
            const std::size_t anchors = static_cast<std::size_t>(shape.anchors);
            const std::size_t classes = static_cast<std::size_t>(shape.num_classes);
            SA_FAIL_IF(boxes.size() != batch * anchors * K_BOX_VALUES ||
                           scores.size() != batch * anchors * classes,
                       StatusCode::InvalidDescription,
                       "a suppression postprocess needs [{}, {}, 4] boxes and [{}, {}, {}] "
                       "scores; got {} and {} values",
                       batch, anchors, batch, anchors, classes, boxes.size(), scores.size());
            const float confidence = config.confidence_threshold;
            SA_FAIL_IF(!(confidence >= 0.0f && confidence <= 1.0f) ||
                           !(config.iou_threshold >= 0.0f && config.iou_threshold <= 1.0f),
                       StatusCode::InvalidDescription,
                       "the confidence and IoU thresholds must lie in [0, 1]; got {} and {}",
                       config.confidence_threshold, config.iou_threshold);
            SA_FAIL_IF(config.max_detections <= 0 || config.max_candidates <= 0,
                       StatusCode::InvalidDescription,
                       "the detection and candidate limits must be positive; got {} and {}",
                       config.max_detections, config.max_candidates);

            detections.resize(batch);
            for (std::size_t image = 0; image < batch; ++image)
            {
                select_image_detections(
                    boxes.subspan(image * anchors * K_BOX_VALUES, anchors * K_BOX_VALUES),
                    scores.subspan(image * anchors * classes, anchors * classes), classes, config,
                    detections[image]);
            }
            return Status();
        }

    } // namespace Vision::Detect
} // namespace SushiAI

/**************************************************************************/
/* detections_topk.cpp                                                    */
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
 * @file detections_topk.cpp
 * @brief Implements the top-k postprocess of the one-to-one head.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/detections_topk.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        Result<std::int64_t> count_top_detections(const PredictionShape& shape,
                                                  std::int64_t max_detections) noexcept
        {
            SA_FAIL_IF(shape.batch <= 0 || shape.anchors <= 0 || shape.num_classes <= 0 ||
                           max_detections <= 0,
                       StatusCode::InvalidDescription,
                       "a top-k postprocess needs positive extents; got batch {}, {} anchors, "
                       "{} classes and a limit of {}",
                       shape.batch, shape.anchors, shape.num_classes, max_detections);
            return std::min(max_detections, shape.anchors);
        }

        Status select_top_detections(SushiRuntime::span<const float> boxes,
                                     SushiRuntime::span<const float> scores,
                                     const PredictionShape& shape,
                                     std::int64_t max_detections,
                                     SushiRuntime::span<float> detections) noexcept
        {
            SA_TRY_ASSIGN(const std::int64_t kept_count,
                          count_top_detections(shape, max_detections));
            const std::size_t kept = static_cast<std::size_t>(kept_count);
            const std::size_t batch = static_cast<std::size_t>(shape.batch);
            const std::size_t anchors = static_cast<std::size_t>(shape.anchors);
            const std::size_t classes = static_cast<std::size_t>(shape.num_classes);
            SA_FATAL_IF(boxes.size() != batch * anchors * K_BOX_VALUES ||
                            scores.size() != batch * anchors * classes ||
                            detections.size() != batch * kept * K_DETECTION_VALUES,
                        "a top-k postprocess needs [{}, {}, 4] boxes, [{}, {}, {}] scores and "
                        "[{}, {}, 6] detections; got {}, {} and {} values",
                        batch, anchors, batch, anchors, classes, batch, kept, boxes.size(),
                        scores.size(), detections.size());

            std::vector<float> best(anchors);
            std::vector<float> gathered(kept * classes);
            std::vector<std::int64_t> leading;
            std::vector<std::int64_t> winners;

            for (std::size_t image = 0; image < batch; ++image)
            {
                const float* image_scores = scores.data() + image * anchors * classes;
                const float* image_boxes = boxes.data() + image * anchors * K_BOX_VALUES;
                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                {
                    const float* row = image_scores + anchor * classes;
                    best[anchor] = *std::max_element(row, row + classes);
                }
                select_top_k(SushiRuntime::span<const float>(best.data(), anchors),
                             static_cast<std::int64_t>(kept), leading);

                for (std::size_t rank = 0; rank < kept; ++rank)
                {
                    const float* row =
                        image_scores + static_cast<std::size_t>(leading[rank]) * classes;
                    std::copy(row, row + classes, gathered.begin() + rank * classes);
                }
                select_top_k(SushiRuntime::span<const float>(gathered.data(), gathered.size()),
                             static_cast<std::int64_t>(kept), winners);

                float* out = detections.data() + image * kept * K_DETECTION_VALUES;
                for (std::size_t rank = 0; rank < kept; ++rank)
                {
                    const std::size_t flat = static_cast<std::size_t>(winners[rank]);
                    const std::size_t anchor = static_cast<std::size_t>(leading[flat / classes]);
                    const float* box = image_boxes + anchor * K_BOX_VALUES;
                    float* detection = out + rank * K_DETECTION_VALUES;
                    std::copy(box, box + K_BOX_VALUES, detection);
                    detection[K_DETECTION_SCORE] = gathered[flat];
                    detection[K_DETECTION_CLASS] = static_cast<float>(flat % classes);
                }
            }
            return Status();
        }

        Status keep_confident_detections(SushiRuntime::span<const float> detections,
                                         std::int64_t batch,
                                         float confidence_threshold,
                                         std::vector<std::vector<float>>& kept) noexcept
        {
            SA_FAIL_IF(batch <= 0, StatusCode::InvalidDescription,
                       "a confidence filter needs a positive batch; got {}", batch);
            const std::size_t images = static_cast<std::size_t>(batch);
            SA_FAIL_IF(detections.size() % (images * K_DETECTION_VALUES) != 0,
                       StatusCode::InvalidDescription,
                       "a confidence filter needs [{}, k, 6] detections; got {} values", batch,
                       detections.size());
            SA_FAIL_IF(!(confidence_threshold >= 0.0f && confidence_threshold <= 1.0f),
                       StatusCode::InvalidDescription,
                       "the confidence threshold must lie in [0, 1]; got {}",
                       confidence_threshold);

            const std::size_t rows = detections.size() / (images * K_DETECTION_VALUES);
            kept.resize(images);
            for (std::size_t image = 0; image < images; ++image)
            {
                const float* image_rows = detections.data() + image * rows * K_DETECTION_VALUES;
                std::vector<float>& confident = kept[image];
                confident.clear();
                for (std::size_t rank = 0; rank < rows; ++rank)
                {
                    const float* row = image_rows + rank * K_DETECTION_VALUES;
                    if (row[K_DETECTION_SCORE] > confidence_threshold)
                        confident.insert(confident.end(), row, row + K_DETECTION_VALUES);
                }
            }
            return Status();
        }

    } // namespace Vision::Detect
} // namespace SushiAI

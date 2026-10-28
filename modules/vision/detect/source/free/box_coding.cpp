/**************************************************************************/
/* box_coding.cpp                                                         */
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
 * @file box_coding.cpp
 * @brief Implements the coding of boxes as distances about anchor points.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/box_coding.hpp>

#include <cstddef>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        namespace
        {
            /**
             * @brief Returns the anchor count A, and stops the process unless the rows are
             *        `[B, A, 4]`.
             * @param anchor_values Values of the per-anchor array.
             * @param values_per_anchor Width of the per-anchor array: K_POINT_VALUES or 1.
             */
            std::size_t check_anchor_rows(std::size_t row_values,
                                          std::size_t anchor_values,
                                          std::size_t values_per_anchor,
                                          std::size_t output_values)
            {
                const std::size_t anchors = anchor_values / values_per_anchor;
                SA_FATAL_IF(anchor_values % values_per_anchor != 0 || anchors == 0 ||
                                row_values % (anchors * K_BOX_VALUES) != 0 ||
                                output_values != row_values,
                            "box rows must be [B, A, 4] over {} anchors with an output of the "
                            "same size; got {} and {} values",
                            anchors, row_values, output_values);
                return anchors;
            }
        } // namespace

        void decode_xyxy(SushiRuntime::span<const float> distances,
                         SushiRuntime::span<const float> anchor_points,
                         SushiRuntime::span<float> boxes)
        {
            const std::size_t anchors = check_anchor_rows(distances.size(), anchor_points.size(),
                                                          K_POINT_VALUES, boxes.size());
            const std::size_t rows = distances.size() / K_BOX_VALUES;
            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                const float* point = anchor_points.data() + (row % anchors) * K_POINT_VALUES;
                const float x1 = point[0] - distances[at];
                const float y1 = point[1] - distances[at + 1];
                const float x2 = point[0] + distances[at + 2];
                const float y2 = point[1] + distances[at + 3];
                boxes[at] = x1;
                boxes[at + 1] = y1;
                boxes[at + 2] = x2;
                boxes[at + 3] = y2;
            }
        }

        void decode_xywh(SushiRuntime::span<const float> distances,
                         SushiRuntime::span<const float> anchor_points,
                         SushiRuntime::span<float> boxes)
        {
            decode_xyxy(distances, anchor_points, boxes);
            xyxy_to_xywh(boxes, boxes);
        }

        void encode_distances(SushiRuntime::span<const float> anchor_points,
                              SushiRuntime::span<const float> boxes,
                              SushiRuntime::span<float> distances)
        {
            const std::size_t anchors = check_anchor_rows(boxes.size(), anchor_points.size(),
                                                          K_POINT_VALUES, distances.size());
            const std::size_t rows = boxes.size() / K_BOX_VALUES;
            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                const float* point = anchor_points.data() + (row % anchors) * K_POINT_VALUES;
                const float left = point[0] - boxes[at];
                const float top = point[1] - boxes[at + 1];
                const float right = boxes[at + 2] - point[0];
                const float bottom = boxes[at + 3] - point[1];
                distances[at] = left;
                distances[at + 1] = top;
                distances[at + 2] = right;
                distances[at + 3] = bottom;
            }
        }

        void scale_boxes(SushiRuntime::span<const float> boxes,
                         SushiRuntime::span<const float> strides,
                         SushiRuntime::span<float> scaled)
        {
            const std::size_t anchors =
                check_anchor_rows(boxes.size(), strides.size(), 1, scaled.size());
            const std::size_t rows = boxes.size() / K_BOX_VALUES;
            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                const float stride = strides[row % anchors];
                scaled[at] = boxes[at] * stride;
                scaled[at + 1] = boxes[at + 1] * stride;
                scaled[at + 2] = boxes[at + 2] * stride;
                scaled[at + 3] = boxes[at + 3] * stride;
            }
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI

/**************************************************************************/
/* ground_truth.hpp                                                       */
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
 * @file ground_truth.hpp
 * @brief Declares the padded per-image ground truth a label assigner reads.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Holds a batch's ground truth with every image padded to the same box count.
         * @see modules/vision/detect/README.md
         */
        struct GroundTruthBatch
        {
            std::int64_t batch = 0;           /**< Images, B. */
            std::int64_t max_boxes = 0;       /**< Rows per image, M: the largest box count. */
            std::vector<std::int64_t> labels; /**< `[B, M]` classes; 0 in a padding row. */
            std::vector<float> boxes;         /**< `[B, M, 4]` x1 y1 x2 y2 in pixels. */
            std::vector<std::uint8_t> valid;  /**< `[B, M]`; 1 where the row holds a box. */
        };

        /** @brief Holds a batch's label rows as a dataset gives them, one row per box. */
        struct GroundTruthRows
        {
            SushiRuntime::span<const std::int64_t> image_indices; /**< `[n]` image of each row. */
            SushiRuntime::span<const std::int64_t> classes;       /**< `[n]` class of each row. */
            SushiRuntime::span<const float> boxes;                /**< `[n, 4]` centre, size. */
        };

        /**
         * @brief Pads label rows into per-image arrays and converts the boxes to pixel corners.
         * @param batch Images in the batch; every image index must be below it.
         * @param image_width Input width in pixels; scales x and width.
         * @param image_height Input height in pixels; scales y and height.
         * @see modules/vision/detect/README.md
         */
        void pad_ground_truth(const GroundTruthRows& rows,
                              std::int64_t batch,
                              float image_width,
                              float image_height,
                              GroundTruthBatch& padded);

    } // namespace Vision::Detect
} // namespace SushiAI

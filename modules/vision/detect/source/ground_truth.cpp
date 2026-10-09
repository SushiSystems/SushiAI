/**************************************************************************/
/* ground_truth.cpp                                                       */
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
 * @file ground_truth.cpp
 * @brief Implements the padding of label rows into per-image ground truth.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/ground_truth.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

#include <SushiAI/core/common.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        void pad_ground_truth(const GroundTruthRows& rows,
                              std::int64_t batch,
                              float image_width,
                              float image_height,
                              GroundTruthBatch& padded)
        {
            const std::size_t row_count = rows.image_indices.size();
            SA_THROW_IF(batch <= 0 || rows.classes.size() != row_count ||
                            rows.boxes.size() != row_count * K_BOX_VALUES,
                        "ground truth needs a positive batch and [n], [n] and [n, 4] rows; got "
                        "batch {} and {}, {} and {} values",
                        batch, row_count, rows.classes.size(), rows.boxes.size());

            const std::size_t images = static_cast<std::size_t>(batch);
            std::vector<std::int64_t> filled(images, 0);
            for (std::size_t row = 0; row < row_count; ++row)
            {
                const std::int64_t image = rows.image_indices[row];
                SA_THROW_IF(image < 0 || image >= batch,
                            "ground truth row {} names image {} of a batch of {}", row, image,
                            batch);
                ++filled[static_cast<std::size_t>(image)];
            }

            const std::size_t max_boxes = static_cast<std::size_t>(
                *std::max_element(filled.begin(), filled.end()));
            padded.batch = batch;
            padded.max_boxes = static_cast<std::int64_t>(max_boxes);
            padded.labels.assign(images * max_boxes, 0);
            padded.boxes.assign(images * max_boxes * K_BOX_VALUES, 0.0f);
            padded.valid.assign(images * max_boxes, 0);

            // filled restarts at zero and counts the rows already written for each image.
            std::fill(filled.begin(), filled.end(), std::int64_t{0});
            for (std::size_t row = 0; row < row_count; ++row)
            {
                const std::size_t image = static_cast<std::size_t>(rows.image_indices[row]);
                const std::size_t slot =
                    image * max_boxes + static_cast<std::size_t>(filled[image]++);
                const float* source = rows.boxes.data() + row * K_BOX_VALUES;
                const SushiRuntime::span<float> box(padded.boxes.data() + slot * K_BOX_VALUES,
                                                    K_BOX_VALUES);

                box[0] = source[0] * image_width;
                box[1] = source[1] * image_height;
                box[2] = source[2] * image_width;
                box[3] = source[3] * image_height;
                xywh_to_xyxy(box, box);

                padded.labels[slot] = rows.classes[row];
                padded.valid[slot] = (box[0] + box[1] + box[2] + box[3]) > 0.0f ? 1 : 0;
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI

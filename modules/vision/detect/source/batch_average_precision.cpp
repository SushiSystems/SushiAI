/**************************************************************************/
/* batch_average_precision.cpp                                            */
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
 * @file batch_average_precision.cpp
 * @brief Implements the feed of a batch's detections and ground truth into AveragePrecision.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/batch_average_precision.hpp>

#include <cstddef>
#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        Status add_batch(AveragePrecision& metric,
                         const std::vector<std::vector<float>>& detections,
                         const GroundTruthBatch& truth) noexcept
        {
            const auto images = static_cast<std::size_t>(truth.batch);
            const auto slots = static_cast<std::size_t>(truth.max_boxes);
            SA_FAIL_IF(detections.size() != images ||
                           truth.labels.size() != images * slots ||
                           truth.valid.size() != images * slots ||
                           truth.boxes.size() != images * slots * K_BOX_VALUES,
                       StatusCode::InvalidDescription,
                       "a batch of {} images needs {} detection arrays and [{}, {}] truth rows; "
                       "got {} arrays, {} labels, {} flags and {} box values",
                       images, images, images, slots, detections.size(), truth.labels.size(),
                       truth.valid.size(), truth.boxes.size());

            // The arrays keep their storage from image to image.
            std::vector<float> boxes;
            std::vector<std::int64_t> classes;
            for (std::size_t image = 0; image < images; ++image)
            {
                boxes.clear();
                classes.clear();
                for (std::size_t slot = image * slots; slot < (image + 1) * slots; ++slot)
                {
                    if (truth.valid[slot] == 0)
                        continue;
                    const float* const corners = truth.boxes.data() + slot * K_BOX_VALUES;
                    boxes.insert(boxes.end(), corners, corners + K_BOX_VALUES);
                    classes.push_back(truth.labels[slot]);
                }
                SA_TRY(metric.add_image(detections[image], boxes, classes));
            }

            return Status();
        }

    } // namespace Vision::Detect
} // namespace SushiAI

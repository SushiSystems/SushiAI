/**************************************************************************/
/* batch_average_precision.hpp                                            */
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
 * @file batch_average_precision.hpp
 * @brief Declares the feed of a batch's detections and ground truth into AveragePrecision.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Adds every image of a batch to @p metric, dropping the padding rows of the truth.
         * @param detections One `[n, 6]` array per image, as select_nms_detections writes them.
         * @param truth The batch's ground truth; its boxes are in the frame of @p detections.
         * @return InvalidDescription when @p detections does not hold `truth.batch` arrays or an
         *         array of @p truth is not `batch` times `max_boxes` rows.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] Status add_batch(AveragePrecision& metric,
                       const std::vector<std::vector<float>>& detections,
                             const GroundTruthBatch& truth) noexcept;

    } // namespace Vision::Detect
} // namespace SushiAI

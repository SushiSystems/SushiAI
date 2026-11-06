/**************************************************************************/
/* segmentation_metrics.hpp                                               */
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
 * @file segmentation_metrics.hpp
 * @brief Declares the metrics a per-pixel objective reports of a step that has run.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Names the metric that holds Segment::mean_iou() of the step. */
    inline constexpr std::string_view K_MEAN_IOU_METRIC = "mean_iou";

    /** @brief Names the metric that holds Segment::pixel_accuracy() of the step. */
    inline constexpr std::string_view K_PIXEL_ACCURACY_METRIC = "pixel_accuracy";

    /**
     * @brief Appends the mean IoU and the pixel accuracy of the step's batch to @p metrics,
     *        the class of a pixel being the largest of its C logits, picked on the host.
     * @param extents The extents of the step's one output.
     * @return InvalidDescription for several outputs, unreadable logits, no mask role or
     *         over 4096 classes; Incompatible for another count or a label out of bounds.
     */
    [[nodiscard]] Status measure_segmentation(const Train::StepView& step,
                                              const PixelExtents& extents,
                                              std::int32_t ignore_index,
                                              Train::Metrics& metrics) noexcept;

} // namespace SushiAI::Vision::Training

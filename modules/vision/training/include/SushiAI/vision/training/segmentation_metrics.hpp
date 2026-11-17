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
#include <SushiRuntime/core/span.hpp>
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
     * @brief Names the INT64 [N * H * W] value of the class of each pixel, which a per-pixel
     *        objective declares and measure_segmentation() reads from the step.
     */
    inline constexpr std::string_view K_PIXEL_CLASSES_VALUE = "pixel_classes";

    /**
     * @brief Returns the names measure_segmentation() appends, in the order it appends them.
     * @return A view of a static array.
     */
    [[nodiscard]] inline SushiRuntime::span<const std::string_view> segmentation_metric_names()
        noexcept
    {
        static constexpr std::string_view K_NAMES[] = {K_MEAN_IOU_METRIC, K_PIXEL_ACCURACY_METRIC};

        return SushiRuntime::span<const std::string_view>(K_NAMES, 2);
    }

    /**
     * @brief Appends the mean IoU and the pixel accuracy of the step's batch to @p metrics,
     *        the class of a pixel being the value the step holds under K_PIXEL_CLASSES_VALUE.
     * @param extents The extents of the step's logits.
     * @return InvalidDescription for a step with no class value or no mask role, or a class
     *         count above int32 or whose C by C matrix does not fit memory; Incompatible for
     *         a class value or a mask of another type or count, or a class out of bounds.
     */
    [[nodiscard]] Status measure_segmentation(const Train::StepView& step,
                                              const PixelExtents& extents,
                                              std::int32_t ignore_index,
                                              Train::Metrics& metrics) noexcept;

} // namespace SushiAI::Vision::Training

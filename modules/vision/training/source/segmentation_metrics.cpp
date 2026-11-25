/**************************************************************************/
/* segmentation_metrics.cpp                                               */
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
 * @file segmentation_metrics.cpp
 * @brief Implements the metrics a per-pixel objective reports of a step that has run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/segmentation_metrics.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>

namespace SushiAI::Vision::Training
{
    Status measure_segmentation(const Train::StepView& step, const PixelExtents& extents,
                                std::int32_t ignore_index, Train::Metrics& metrics) noexcept
    {
        const std::int64_t pixels = extents.pixels();
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const classes,
                      Train::named_value(step, K_PIXEL_CLASSES_VALUE));
        SA_FAIL_IF(classes->dtype != DType::Int64 || classes->num_elements != pixels,
                   StatusCode::Incompatible,
                   "The step's class value holds {} elements of {}; its metrics read {} of {}",
                   classes->num_elements, dtype_name(classes->dtype), pixels,
                   dtype_name(DType::Int64));
        SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                      Train::require_batch_role(step.batch, Data::K_ROLE_MASK));
        SA_FAIL_IF(mask->values.dtype != DType::Int32 || mask->values.num_elements != pixels,
                   StatusCode::Incompatible,
                   "The batch's mask holds {} elements of {}; the step's metrics read {} of {}",
                   mask->values.num_elements, dtype_name(mask->values.dtype), pixels,
                   dtype_name(DType::Int32));
        SA_FAIL_IF(extents.classes > std::numeric_limits<std::int32_t>::max(),
                   StatusCode::InvalidDescription,
                   "The step scores {} classes; a class index is an int32 and the metrics "
                   "count no more than {}",
                   extents.classes, std::numeric_limits<std::int32_t>::max());

        SA_TRY_ASSIGN(Segment::ConfusionMatrix matrix,
                      Segment::ConfusionMatrix::make(static_cast<std::int32_t>(extents.classes)));
        const std::size_t count = static_cast<std::size_t>(pixels);
        SA_TRY_ASSIGN(const std::vector<std::int64_t> predicted,
                      read_elements<std::int64_t>(*classes));
        SA_TRY(matrix.add(SushiRuntime::span<const std::int64_t>(predicted.data(), count),
                          SushiRuntime::span<const std::int32_t>(
                              mask->values.data_as<std::int32_t>(), count),
                          ignore_index));

        SA_TRY(metrics.add(Train::Metric{K_MEAN_IOU_METRIC, Segment::mean_iou(matrix), true}));

        return metrics.add(Train::Metric{K_PIXEL_ACCURACY_METRIC,
                                         Segment::pixel_accuracy(matrix), true});
    }

} // namespace SushiAI::Vision::Training

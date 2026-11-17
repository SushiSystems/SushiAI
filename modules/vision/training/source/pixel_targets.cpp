/**************************************************************************/
/* pixel_targets.cpp                                                      */
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
 * @file pixel_targets.cpp
 * @brief Implements the fill of the two leaves of the per-pixel cross entropy.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/pixel_targets.hpp>

#include <algorithm>
#include <type_traits>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>

#include "host_reals.hpp"

namespace SushiAI::Vision::Training
{
    Status fill_pixel_targets(const SushiBLAS::Tensor& mask, std::int64_t classes,
                              std::int32_t ignore_index, SushiBLAS::Tensor& labels,
                              SushiBLAS::Tensor& weights) noexcept
    {
        const std::int64_t pixels = labels.num_elements;
        SA_FAIL_IF(labels.dtype != DType::Int32 || !Detail::holds_reals(weights),
                   StatusCode::Incompatible,
                   "The step's pixel labels are {} and its pixel weights are {}; they are "
                   "written as {} and as a real floating type",
                   dtype_name(labels.dtype), dtype_name(weights.dtype),
                   dtype_name(DType::Int32));
        SA_FAIL_IF(weights.num_elements != pixels, StatusCode::Incompatible,
                   "The step holds {} pixel labels but {} pixel weights; one of each per "
                   "pixel",
                   pixels, weights.num_elements);
        SA_FAIL_IF(classes < 1, StatusCode::Incompatible,
                   "A pixel step holds at least one class; got {}", classes);
        SA_TRY(validate_pixel_labels(mask, pixels, classes, ignore_index));

        const std::int32_t* const label = mask.data_as<std::int32_t>();
        std::int64_t kept = 0;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
            kept += label[pixel] != ignore_index ? 1 : 0;

        // Every label was checked, so a refused batch leaves both leaves as they were.
        std::copy(label, label + pixels, labels.data_as<std::int32_t>());

        const double share = kept == 0 ? 0.0 : 1.0 / static_cast<double>(kept);
        Detail::write_reals(weights, [&](auto* const weight)
        {
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
            {
                const double value = label[pixel] != ignore_index ? share : 0.0;
                weight[pixel] = static_cast<std::decay_t<decltype(*weight)>>(value);
            }
        });

        return Status();
    }

} // namespace SushiAI::Vision::Training

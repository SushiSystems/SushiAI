/**************************************************************************/
/* dense_targets.cpp                                                      */
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
 * @file dense_targets.cpp
 * @brief Implements the host fills that copy the real values of a batch role into a target leaf.
 * @author Mustafa Garip
 */

#include <SushiAI/train/dense_targets.hpp>

#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Returns Incompatible unless @p values can be copied into @p targets. */
            [[nodiscard]] Status require_copyable(const SushiBLAS::Tensor& values,
                                                  const SushiBLAS::Tensor& targets)
            {
                SA_FAIL_IF(values.dtype != DType::Float32, StatusCode::Incompatible,
                           "The batch's values are {} but the compiled step reads {}",
                           dtype_name(values.dtype), dtype_name(DType::Float32));
                SA_FAIL_IF(targets.dtype != DType::Float32 && targets.dtype != DType::Float64,
                           StatusCode::Incompatible,
                           "The compiled step's targets are {}; real values are written as {} "
                           "or {}",
                           dtype_name(targets.dtype), dtype_name(DType::Float32),
                           dtype_name(DType::Float64));
                SA_FAIL_IF(values.num_elements != targets.num_elements, StatusCode::Incompatible,
                           "The batch holds {} values but the compiled step's targets hold {}",
                           values.num_elements, targets.num_elements);

                return Status();
            }

            /**
             * @brief Copies @p values into @p targets. @pre require_copyable() accepted them.
             * @return Success, or the failure of the allocation or of the copy.
             */
            [[nodiscard]] Status copy_checked(const SushiBLAS::Tensor& values,
                                              SushiBLAS::Tensor& targets) noexcept
            {
                const float* const from = values.data_as<float>();
                SA_TRY_ASSIGN(const std::vector<double> widened,
                              Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
                {
                    return std::vector<double>(from, from + values.num_elements);
                }));

                return write_reals(targets, SushiRuntime::span<const double>(widened.data(),
                                                                             widened.size()));
            }
        } // namespace

        Status fill_dense_targets(const SushiBLAS::Tensor& values, SushiBLAS::Tensor& targets) noexcept
        {
            SA_TRY(require_copyable(values, targets));

            return copy_checked(values, targets);
        }

        Status fill_binary_targets(const SushiBLAS::Tensor& labels, SushiBLAS::Tensor& targets) noexcept
        {
            SA_TRY(require_copyable(labels, targets));

            const float* const label = labels.data_as<float>();
            for (std::int64_t index = 0; index < labels.num_elements; ++index)
            {
                SA_FAIL_IF(label[index] != 0.0f && label[index] != 1.0f,
                           StatusCode::Incompatible,
                           "The label of element {} is {}; a binary label is 0 or 1", index,
                           label[index]);
            }

            // Every label was checked, so a refused batch leaves the leaf as it was.
            return copy_checked(labels, targets);
        }

    } // namespace Train
} // namespace SushiAI

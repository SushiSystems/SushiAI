/**************************************************************************/
/* next_token_targets.cpp                                                 */
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
 * @file next_token_targets.cpp
 * @brief Implements the fill of the next-token objective's target leaf.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/next_token_targets.hpp>

#include <cstdint>
#include <cstddef>
#include <limits>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Language::Training
{
    Status fill_next_token_targets(SushiRuntime::span<const std::int32_t> labels,
                                   std::int64_t vocabulary,
                                   SushiRuntime::span<std::int32_t> targets) noexcept
    {
        SA_FAIL_IF(labels.size() != targets.size(), StatusCode::Incompatible,
                   "The batch holds {} labels but the compiled step expects {}, one per "
                   "position",
                   labels.size(), targets.size());

        const std::int64_t rows = static_cast<std::int64_t>(targets.size());
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();
        SA_FAIL_IF(vocabulary < 1 || rows > K_LARGEST_INDEX / vocabulary,
                   StatusCode::Incompatible,
                   "{} positions over a vocabulary of {} cannot be indexed in {}", rows,
                   vocabulary, dtype_name(DType::Int32));

        for (std::int64_t row = 0; row < rows; ++row)
        {
            const std::int32_t label = labels[static_cast<std::size_t>(row)];
            SA_FAIL_IF(label < 0 || label >= vocabulary, StatusCode::Incompatible,
                       "The label of position {} is {} but the vocabulary holds {} tokens",
                       row, label, vocabulary);
        }

        // Every label was checked, so a refused batch leaves the leaf as it was.
        for (std::int64_t row = 0; row < rows; ++row)
        {
            targets[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(
                row * vocabulary + labels[static_cast<std::size_t>(row)]);
        }

        return Status();
    }

    Status fill_next_token_targets(const SushiBLAS::Tensor& labels, std::int64_t vocabulary,
                                   SushiBLAS::Tensor& targets) noexcept
    {
        SA_FAIL_IF(labels.dtype != DType::Int32 || targets.dtype != DType::Int32,
                   StatusCode::Incompatible,
                   "The batch's labels are {} and the step's targets are {}; both are read "
                   "as {}",
                   dtype_name(labels.dtype), dtype_name(targets.dtype),
                   dtype_name(DType::Int32));

        return fill_next_token_targets(
            SushiRuntime::span<const std::int32_t>(labels.data_as<std::int32_t>(),
                                                   static_cast<std::size_t>(labels.num_elements)),
            vocabulary,
            SushiRuntime::span<std::int32_t>(targets.data_as<std::int32_t>(),
                                             static_cast<std::size_t>(targets.num_elements)));
    }

} // namespace SushiAI::Language::Training

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
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

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

        SA_FAIL_IF(vocabulary < 1, StatusCode::Incompatible,
                   "A vocabulary of {} holds no token", vocabulary);

        const std::int64_t rows = static_cast<std::int64_t>(targets.size());

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
            targets[static_cast<std::size_t>(row)] = labels[static_cast<std::size_t>(row)];
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

        SA_TRY_ASSIGN(std::vector<std::int32_t> staged,
                      SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::vector<std::int32_t>(static_cast<std::size_t>(targets.num_elements));
        }));
        SA_TRY(fill_next_token_targets(
            SushiRuntime::span<const std::int32_t>(labels.data_as<std::int32_t>(),
                                                   static_cast<std::size_t>(labels.num_elements)),
            vocabulary, SushiRuntime::span<std::int32_t>(staged.data(), staged.size())));

        return write_elements(targets, staged);
    }

} // namespace SushiAI::Language::Training

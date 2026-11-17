/**************************************************************************/
/* next_token_inputs.cpp                                                  */
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
 * @file next_token_inputs.cpp
 * @brief Implements the host check of the token ids a batch feeds to the embedding.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/next_token_inputs.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Language::Training
{
    Status check_next_token_inputs(SushiRuntime::span<const std::int32_t> ids,
                                   std::int64_t vocabulary) noexcept
    {
        for (std::size_t position = 0; position < ids.size(); ++position)
        {
            const std::int32_t id = ids[position];
            SA_FAIL_IF(id < 0 || id >= vocabulary, StatusCode::Incompatible,
                       "The input id of position {} is {} but the embedding holds {} rows",
                       position, id, vocabulary);
        }

        return Status();
    }

    Status check_next_token_inputs(const SushiBLAS::Tensor& inputs,
                                   std::int64_t vocabulary) noexcept
    {
        SA_FAIL_IF(inputs.dtype != DType::Int32, StatusCode::Incompatible,
                   "The batch's inputs are {} but token ids are read as {}",
                   dtype_name(inputs.dtype), dtype_name(DType::Int32));

        return check_next_token_inputs(
            SushiRuntime::span<const std::int32_t>(inputs.data_as<std::int32_t>(),
                                                   static_cast<std::size_t>(inputs.num_elements)),
            vocabulary);
    }

} // namespace SushiAI::Language::Training

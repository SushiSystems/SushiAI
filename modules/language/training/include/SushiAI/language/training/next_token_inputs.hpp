/**************************************************************************/
/* next_token_inputs.hpp                                                  */
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
 * @file next_token_inputs.hpp
 * @brief Declares the host check of the token ids a batch feeds to the model's embedding.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Language::Training
{
    /**
     * @brief Checks on the host that every input id of a batch is a row of the embedding table.
     * @param ids Token ids; each must be in [0, vocabulary).
     * @param vocabulary V, the rows of the embedding table the ids index.
     * @return Success, or Incompatible naming the position, the id and the vocabulary.
     */
    [[nodiscard]] Status check_next_token_inputs(SushiRuntime::span<const std::int32_t> ids,
                                                 std::int64_t vocabulary) noexcept;

    /**
     * @brief Checks the ids of an INT32 input tensor as the span overload does.
     * @return Success, or Incompatible for another element type or an id outside the table.
     */
    [[nodiscard]] Status check_next_token_inputs(const SushiBLAS::Tensor& inputs,
                                                 std::int64_t vocabulary) noexcept;

} // namespace SushiAI::Language::Training

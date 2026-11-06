/**************************************************************************/
/* host_values.hpp                                                        */
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
 * @file host_values.hpp
 * @brief Declares the host read of one element of a float32 or float64 tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Checks that the host can read @p tensor as @p elements real numbers.
         * @param what Names the tensor in the message, e.g. "the logits".
         * @return Success, InvalidDescription for an element type that is neither float32
         *         nor float64, or Incompatible for another element count.
         */
        [[nodiscard]] Status require_host_reals(const SushiBLAS::Tensor& tensor,
                                                std::int64_t elements, std::string_view what) noexcept;

        /**
         * @brief Returns element @p index of @p tensor as a double.
         * @pre require_host_reals() accepted @p tensor and @p index is below its count.
         */
        [[nodiscard]] double host_real(const SushiBLAS::Tensor& tensor,
                                       std::int64_t index) noexcept;

    } // namespace Train
} // namespace SushiAI

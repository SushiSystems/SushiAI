/**************************************************************************/
/* host_reals.hpp                                                         */
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
 * @file host_reals.hpp
 * @brief Defines the access to a float32 or float64 tensor from the host by its element type.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Vision::Training::Detail
{
    /** @brief Reports whether @p tensor holds float32 or float64 elements. */
    [[nodiscard]] inline bool holds_reals(const SushiBLAS::Tensor& tensor) noexcept
    {
        return tensor.dtype == DType::Float32 || tensor.dtype == DType::Float64;
    }

    /**
     * @brief Calls @p fill once with a double* to as many host values as @p tensor holds, all
     *        of which it writes, then copies them over the tensor.
     * @pre holds_reals(tensor), and no recorded task reads or writes the tensor.
     * @return Success, or the failure of the allocation or of the copy.
     */
    template<typename Fill>
    [[nodiscard]] Status fill_reals(const SushiBLAS::Tensor& tensor, Fill&& fill)
    {
        SA_TRY_ASSIGN(std::vector<double> staged,
                      ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                     [&tensor]()
        {
            return std::vector<double>(static_cast<std::size_t>(tensor.num_elements), 0.0);
        }));
        std::forward<Fill>(fill)(staged.data());

        return ::SushiAI::write_reals(
            tensor, SushiRuntime::span<const double>(staged.data(), staged.size()));
    }

} // namespace SushiAI::Vision::Training::Detail

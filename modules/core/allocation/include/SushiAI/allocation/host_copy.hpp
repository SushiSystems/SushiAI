/**************************************************************************/
/* host_copy.hpp                                                          */
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
 * @file host_copy.hpp
 * @brief Declares the copies that move a tensor's elements between host memory and its storage.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    /**
     * @brief Copies elements [first, first + count) of @p tensor into @p destination and waits.
     * @param element_bytes The width of one host element; it must equal the tensor's.
     * @return InvalidDescription for a tensor without storage, a strided one, a range outside
     *         its elements or another element width, or the Status of what the copy threw.
     */
    [[nodiscard]] Status copy_to_host(const SushiBLAS::Tensor& tensor, std::int64_t first,
                                      std::int64_t count, std::size_t element_bytes,
                                      void* destination) noexcept;

    /**
     * @brief Copies @p count host elements from @p source into @p tensor from element @p first,
     *        and waits.
     * @param element_bytes The width of one host element; it must equal the tensor's.
     * @return The failures copy_to_host() returns, for the same causes.
     */
    [[nodiscard]] Status copy_from_host(const SushiBLAS::Tensor& tensor, std::int64_t first,
                                        std::int64_t count, std::size_t element_bytes,
                                        const void* source) noexcept;

    /**
     * @brief Copies elements of @p tensor from @p first into all of @p destination.
     * @tparam T A host type as wide as the tensor's element.
     * @return What copy_to_host() returns.
     */
    template<typename T>
    [[nodiscard]] Status read_elements(const SushiBLAS::Tensor& tensor, std::int64_t first,
                                       SushiRuntime::span<T> destination) noexcept
    {
        return copy_to_host(tensor, first, static_cast<std::int64_t>(destination.size()),
                            sizeof(T), destination.data());
    }

    /**
     * @brief Returns every element of @p tensor as a host vector.
     * @tparam T A host type as wide as the tensor's element.
     * @return The elements, or what copy_to_host() returns.
     */
    template<typename T>
    [[nodiscard]] Result<std::vector<T>> read_elements(const SushiBLAS::Tensor& tensor) noexcept
    {
        std::vector<T> values(static_cast<std::size_t>(tensor.num_elements > 0
                                                           ? tensor.num_elements
                                                           : 0));
        SA_TRY(read_elements(tensor, 0, SushiRuntime::span<T>(values.data(), values.size())));
        return values;
    }

    /**
     * @brief Copies all of @p source into elements of @p tensor from @p first.
     * @tparam T A host type as wide as the tensor's element.
     * @return What copy_from_host() returns.
     */
    template<typename T>
    [[nodiscard]] Status write_elements(const SushiBLAS::Tensor& tensor, std::int64_t first,
                                        SushiRuntime::span<const T> source) noexcept
    {
        return copy_from_host(tensor, first, static_cast<std::int64_t>(source.size()), sizeof(T),
                              source.data());
    }

    /**
     * @brief Copies @p source over every element of @p tensor.
     * @tparam T A host type as wide as the tensor's element.
     * @return InvalidDescription when @p source and the tensor hold different counts, or what
     *         copy_from_host() returns.
     */
    template<typename T>
    [[nodiscard]] Status write_elements(const SushiBLAS::Tensor& tensor,
                                        const std::vector<T>& source) noexcept
    {
        SA_FAIL_IF(static_cast<std::int64_t>(source.size()) != tensor.num_elements,
                   StatusCode::InvalidDescription,
                   "{} host values cannot replace the {} elements of a tensor", source.size(),
                   tensor.num_elements);
        return write_elements(tensor, 0,
                              SushiRuntime::span<const T>(source.data(), source.size()));
    }

    /**
     * @brief Returns every element of a FLOAT32 or FLOAT64 @p tensor, widened to double.
     * @return The values, InvalidDescription for another element type, or what
     *         copy_to_host() returns.
     */
    [[nodiscard]] Result<std::vector<double>> read_reals(const SushiBLAS::Tensor& tensor) noexcept;

    /**
     * @brief Writes @p values over every element of a FLOAT32 or FLOAT64 @p tensor, rounded to
     *        its element type.
     * @return InvalidDescription for another element type or another count, or what
     *         copy_from_host() returns.
     */
    [[nodiscard]] Status write_reals(const SushiBLAS::Tensor& tensor,
                                     SushiRuntime::span<const double> values) noexcept;

} // namespace SushiAI

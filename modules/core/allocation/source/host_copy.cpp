/**************************************************************************/
/* host_copy.cpp                                                          */
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
 * @file host_copy.cpp
 * @brief Implements the copies between host memory and a tensor's storage.
 * @author Mustafa Garip
 */

#include <SushiAI/allocation/host_copy.hpp>

#include <cstddef>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace
    {
        /**
         * @brief Returns the byte offset in the storage of element @p first of @p tensor.
         * @return The offset, or InvalidDescription when a copy cannot address the range.
         */
        [[nodiscard]] Result<std::size_t> checked_offset(const SushiBLAS::Tensor& tensor,
                                                         std::int64_t first, std::int64_t count,
                                                         std::size_t element_bytes)
        {
            SA_FAIL_IF(!tensor.storage, StatusCode::InvalidDescription,
                       "A host copy needs a tensor with storage");
            SA_FAIL_IF(element_bytes != dtype_size(tensor.dtype), StatusCode::InvalidDescription,
                       "A host element of {} bytes cannot hold a {} element of {} bytes",
                       element_bytes, dtype_name(tensor.dtype), dtype_size(tensor.dtype));
            SA_FAIL_IF(first < 0 || count < 0 || first > tensor.num_elements - count,
                       StatusCode::InvalidDescription,
                       "Elements [{}, {} + {}) lie outside a tensor of {} elements", first, first,
                       count, tensor.num_elements);
            SA_TRY_ASSIGN(const bool contiguous,
                          Detail::guard_value(Detail::SeamPhase::Execute,
                                              [&tensor]() { return tensor.is_contiguous(); }));
            SA_FAIL_IF(!contiguous, StatusCode::InvalidDescription,
                       "A host copy moves a dense range; a strided view has none");

            return static_cast<std::size_t>(tensor.storage_offset + first) * element_bytes;
        }
    } // namespace

    Status copy_to_host(const SushiBLAS::Tensor& tensor, std::int64_t first, std::int64_t count,
                        std::size_t element_bytes, void* destination) noexcept
    {
        SA_TRY_ASSIGN(const std::size_t offset,
                      checked_offset(tensor, first, count, element_bytes));
        if (count == 0)
            return Status();

        return Detail::guard(Detail::SeamPhase::Execute, [&]()
        {
            tensor.storage->copy_to_host(destination, offset,
                                         static_cast<std::size_t>(count) * element_bytes);
            return Status();
        });
    }

    Status copy_from_host(const SushiBLAS::Tensor& tensor, std::int64_t first, std::int64_t count,
                          std::size_t element_bytes, const void* source) noexcept
    {
        SA_TRY_ASSIGN(const std::size_t offset,
                      checked_offset(tensor, first, count, element_bytes));
        if (count == 0)
            return Status();

        return Detail::guard(Detail::SeamPhase::Execute, [&]()
        {
            tensor.storage->copy_from_host(source, offset,
                                           static_cast<std::size_t>(count) * element_bytes);
            return Status();
        });
    }

    Result<std::vector<double>> read_reals(const SushiBLAS::Tensor& tensor) noexcept
    {
        if (tensor.dtype == DType::Float64)
            return read_elements<double>(tensor);

        SA_FAIL_IF(tensor.dtype != DType::Float32, StatusCode::InvalidDescription,
                   "Real values are read from {} or {}; the tensor is {}",
                   dtype_name(DType::Float32), dtype_name(DType::Float64),
                   dtype_name(tensor.dtype));
        SA_TRY_ASSIGN(const std::vector<float> singles, read_elements<float>(tensor));

        return Detail::guard_value(Detail::SeamPhase::Allocate, [&singles]()
        {
            return std::vector<double>(singles.begin(), singles.end());
        });
    }

    Status write_reals(const SushiBLAS::Tensor& tensor,
                       SushiRuntime::span<const double> values) noexcept
    {
        SA_FAIL_IF(static_cast<std::int64_t>(values.size()) != tensor.num_elements,
                   StatusCode::InvalidDescription,
                   "{} host values cannot replace the {} elements of a tensor", values.size(),
                   tensor.num_elements);

        if (tensor.dtype == DType::Float64)
            return write_elements(tensor, 0, values);

        SA_FAIL_IF(tensor.dtype != DType::Float32, StatusCode::InvalidDescription,
                   "Real values are written to {} or {}; the tensor is {}",
                   dtype_name(DType::Float32), dtype_name(DType::Float64),
                   dtype_name(tensor.dtype));
        SA_TRY_ASSIGN(const std::vector<float> singles,
                      Detail::guard_value(Detail::SeamPhase::Allocate, [&values]()
        {
            std::vector<float> narrowed;
            narrowed.reserve(values.size());
            for (const double value : values)
                narrowed.push_back(static_cast<float>(value));
            return narrowed;
        }));

        return write_elements(tensor, 0,
                              SushiRuntime::span<const float>(singles.data(), singles.size()));
    }

} // namespace SushiAI

/**************************************************************************/
/* tensor_values.hpp                                                      */
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
 * @file tensor_values.hpp
 * @brief Declares the functions that write and read the elements of a tensor from the host.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Returns @p values each rounded to the nearest float, as a FLOAT32 tensor holds them. */
        inline std::vector<float> round_to_float32(SushiRuntime::span<const double> values)
        {
            std::vector<float> rounded(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
                rounded[i] = static_cast<float>(values[i]);
            return rounded;
        }

        /** @brief Copies @p values, converted to Element, over every element of @p tensor. */
        template<typename Element>
        void write_converted(const SushiBLAS::Tensor& tensor,
                             SushiRuntime::span<const double> values)
        {
            std::vector<Element> converted(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
                converted[i] = static_cast<Element>(values[i]);
            const Status written = write_elements(tensor, converted);
            EXPECT_TRUE(written.is_ok()) << written.message();
        }

        /** @brief Returns every element of @p tensor, read as Element, converted to double. */
        template<typename Element>
        std::vector<double> read_converted(const SushiBLAS::Tensor& tensor)
        {
            Result<std::vector<Element>> read = read_elements<Element>(tensor);
            EXPECT_TRUE(read.has_value()) << read.status().message();
            if (!read.has_value())
                return std::vector<double>(static_cast<std::size_t>(tensor.num_elements), 0.0);
            return std::vector<double>(read.value().begin(), read.value().end());
        }

        /** @brief Copies host doubles into a FLOAT32, FLOAT64, INT32 or INT64 tensor. */
        inline void write_tensor(const SushiBLAS::Tensor& tensor,
                                 SushiRuntime::span<const double> values) noexcept
        {
            EXPECT_EQ(static_cast<std::size_t>(tensor.num_elements), values.size());

            if (tensor.dtype == DType::Float64)
                write_converted<double>(tensor, values);
            else if (tensor.dtype == DType::Int32)
                write_converted<std::int32_t>(tensor, values);
            else if (tensor.dtype == DType::Int64)
                write_converted<std::int64_t>(tensor, values);
            else
                write_converted<float>(tensor, values);
        }

        /** @brief Copies a FLOAT32, FLOAT64, INT32 or INT64 tensor to the host as doubles. */
        inline std::vector<double> read_tensor(const SushiBLAS::Tensor& tensor) noexcept
        {
            if (tensor.dtype == DType::Float64)
                return read_converted<double>(tensor);
            if (tensor.dtype == DType::Int32)
                return read_converted<std::int32_t>(tensor);
            if (tensor.dtype == DType::Int64)
                return read_converted<std::int64_t>(tensor);
            return read_converted<float>(tensor);
        }

        /**
         * @brief Returns a tensor's element bytes, for a bit-for-bit comparison.
         */
        inline std::vector<unsigned char> read_tensor_bytes(const SushiBLAS::Tensor& tensor) noexcept
        {
            const std::size_t element = dtype_size(tensor.dtype);
            std::vector<unsigned char> bytes(static_cast<std::size_t>(tensor.num_elements)
                                             * element);
            const Status read = copy_to_host(tensor, 0, tensor.num_elements, element,
                                             bytes.data());
            EXPECT_TRUE(read.is_ok()) << read.message();
            return bytes;
        }

        /**
         * @brief Returns the bytes of rows [first, first + count) of a tensor whose elements
         *        hold @p rows equal rows one after another.
         * @pre @p rows divides the element count, and the range lies inside the rows.
         */
        inline std::vector<unsigned char> read_row_bytes(const SushiBLAS::Tensor& tensor,
                                                         std::int64_t rows, std::int64_t first,
                                                         std::int64_t count) noexcept
        {
            const std::vector<unsigned char> all = read_tensor_bytes(tensor);
            const std::size_t per_row = all.size() / static_cast<std::size_t>(rows);
            const std::size_t start = per_row * static_cast<std::size_t>(first);
            const std::size_t length = per_row * static_cast<std::size_t>(count);
            return std::vector<unsigned char>(all.data() + start, all.data() + start + length);
        }

        /**
         * @brief Reports the first byte at which two element buffers differ.
         * @param a One buffer.
         * @param b The other.
         * @return The index, or a.size() when they are identical.
         */
        inline std::size_t first_difference(const std::vector<unsigned char>& a,
                                            const std::vector<unsigned char>& b) noexcept
        {
            const std::size_t shared = a.size() < b.size() ? a.size() : b.size();
            for (std::size_t i = 0; i < shared; ++i)
                if (a[i] != b[i]) return i;
            return shared;
        }

    } // namespace Test
} // namespace SushiAI

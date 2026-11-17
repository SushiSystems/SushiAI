/**************************************************************************/
/* safetensors_reader.hpp                                                 */
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
 * @file safetensors_reader.hpp
 * @brief Declares the reader of a safetensors file: its tensors by name, dtype, shape and byte range.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Names the element type of a safetensors tensor.
         *
         * It is its own enumeration because bfloat16 is no DType; the four types are the ones
         * SafetensorsReader::read_float32 widens.
         */
        enum class SafetensorsType
        {
            /** @brief IEEE 754 binary16, "F16". */
            Float16,
            /** @brief The high half of a float32, "BF16". */
            BFloat16,
            /** @brief IEEE 754 binary32, "F32". */
            Float32,
            /** @brief IEEE 754 binary64, "F64". */
            Float64,
        };

        /**
         * @brief Returns the bytes of one element of @p type.
         * @param type The element type.
         * @return 2, 2, 4 or 8.
         */
        [[nodiscard]] std::size_t safetensors_type_size(SafetensorsType type) noexcept;

        /**
         * @brief Returns the name the file spells @p type as.
         * @param type The element type.
         * @return "F16", "BF16", "F32" or "F64", a literal the library owns.
         */
        [[nodiscard]] std::string_view safetensors_type_name(SafetensorsType type) noexcept;

        /** @brief Names one tensor of a safetensors file and where its bytes lie. */
        struct SafetensorsTensor
        {
            /** @brief Holds the key the header states it under. */
            std::string name;

            /** @brief Holds its element type. */
            SafetensorsType type = SafetensorsType::Float32;

            /** @brief Holds its extents. */
            Shape shape{};

            /** @brief Holds the offset of its first byte from the start of the file. */
            std::size_t offset = 0;

            /** @brief Holds its length in bytes: its element count times its element size. */
            std::size_t length = 0;
        };

        /**
         * @class SafetensorsReader
         * @brief Lists the tensors of a safetensors file held in memory and hands out views of them.
         *
         * The reader copies no byte; the bytes it was read from must outlive it. The tensors are
         * listed in the order of their bytes, and bytes no range names are allowed.
         */
        class SafetensorsReader final
        {
            public:
                /**
                 * @brief Reads the header of @p bytes and checks every tensor it names.
                 * @param bytes The file; must outlive the reader and every view of it.
                 * @param origin Names the file in a message.
                 * @return InvalidFile naming the header or the tensor at fault; Unsupported
                 *         naming the tensor and its dtype for a dtype the reader does not read.
                 */
                [[nodiscard]] static Result<SafetensorsReader> read(
                    SushiRuntime::span<const std::byte> bytes, std::string_view origin) noexcept;

                /** @brief Returns the tensors in the order of their bytes. */
                [[nodiscard]] SushiRuntime::span<const SafetensorsTensor> tensors() const noexcept;

                /** @brief Returns the tensor stored under @p name, or null when none is. */
                [[nodiscard]] const SafetensorsTensor* find(std::string_view name) const noexcept;

                /**
                 * @brief Returns the bytes of @p tensor.
                 * @pre @p tensor is one of tensors().
                 */
                [[nodiscard]] SushiRuntime::span<const std::byte> contents(
                    const SafetensorsTensor& tensor) const noexcept;

                /**
                 * @brief Returns the elements of @p tensor as float32, in the order stored.
                 * @pre @p tensor is one of tensors().
                 * @return The widened elements; a float64 element rounds to the nearest float32.
                 */
                [[nodiscard]] std::vector<float> read_float32(
                    const SafetensorsTensor& tensor) const noexcept;

            private:
                /** @brief Holds the file's bytes and the tensors read from them. */
                SafetensorsReader(SushiRuntime::span<const std::byte> bytes,
                                  std::vector<SafetensorsTensor> tensors) noexcept;

                /** @brief Holds a view of the whole file. */
                SushiRuntime::span<const std::byte> bytes_;

                /** @brief Holds the tensors in the order of their bytes. */
                std::vector<SafetensorsTensor> tensors_;
        };
    } // namespace IO
} // namespace SushiAI

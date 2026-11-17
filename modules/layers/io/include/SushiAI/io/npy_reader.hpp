/**************************************************************************/
/* npy_reader.hpp                                                         */
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
 * @file npy_reader.hpp
 * @brief Declares the reader of a NumPy .npy file: its header and the bytes of its one array.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names the element type of a .npy array; every one is little-endian. */
        enum class NpyType
        {
            /** @brief IEEE 754 binary16, "<f2". */
            Float16,
            /** @brief IEEE 754 binary32, "<f4". */
            Float32,
            /** @brief IEEE 754 binary64, "<f8". */
            Float64,
            /** @brief Signed 8-bit integer, "|i1". */
            Int8,
            /** @brief Unsigned 8-bit integer, "|u1". */
            UInt8,
            /** @brief Signed 16-bit integer, "<i2". */
            Int16,
            /** @brief Unsigned 16-bit integer, "<u2". */
            UInt16,
            /** @brief Signed 32-bit integer, "<i4". */
            Int32,
            /** @brief Unsigned 32-bit integer, "<u4". */
            UInt32,
            /** @brief Signed 64-bit integer, "<i8". */
            Int64,
            /** @brief Unsigned 64-bit integer, "<u8". */
            UInt64,
        };

        /**
         * @brief Returns the bytes of one element of @p type.
         * @param type The element type.
         * @return 1, 2, 4 or 8.
         */
        [[nodiscard]] std::size_t npy_type_size(NpyType type) noexcept;

        /**
         * @brief Returns the descriptor the header spells @p type as, such as "<f4".
         * @param type The element type.
         * @return A literal the library owns.
         */
        [[nodiscard]] std::string_view npy_type_name(NpyType type) noexcept;

        /** @brief Names the array of a .npy file and where its bytes lie. */
        struct NpyArray
        {
            /** @brief Holds its element type. */
            NpyType type = NpyType::Float32;

            /** @brief Holds its extents; a scalar has rank 0. */
            Shape shape{};

            /** @brief Holds whether the bytes are column-major, the header's fortran_order. */
            bool fortran_order = false;

            /** @brief Holds the offset of its first data byte from the start of the file. */
            std::size_t offset = 0;

            /** @brief Holds the length of its data in bytes: elements times element size. */
            std::size_t length = 0;
        };

        /**
         * @class NpyReader
         * @brief Reads the header of a .npy file held in memory and hands out its array.
         *
         * Versions 1, 2 and 3 of the format are read. The reader holds a view of the bytes it
         * was read from and copies none of them; the bytes must outlive it. Bytes after the
         * data the shape needs are allowed.
         */
        class NpyReader final
        {
            public:
                /**
                 * @brief Reads the header of @p bytes and checks the array it describes.
                 * @param bytes The file; must outlive the reader and every view of it.
                 * @param origin Names the file in a message.
                 * @return InvalidFile for a bad magic, header or data length; Unsupported, naming
                 *         the descriptor, for an object, big-endian, structured or unread type,
                 *         and for another version, an extent of 0 or a rank above K_MAX_RANK.
                 */
                [[nodiscard]] static Result<NpyReader> read(
                    SushiRuntime::span<const std::byte> bytes, std::string_view origin) noexcept;

                /** @brief Returns the array the file holds. */
                [[nodiscard]] const NpyArray& array() const noexcept;

                /** @brief Returns the data bytes of the array, in the stored order. */
                [[nodiscard]] SushiRuntime::span<const std::byte> contents() const noexcept;

                /**
                 * @brief Returns the elements as float32 in row-major order.
                 * @return The widened elements, a column-major array transposed to row-major;
                 *         a float64 or integer element rounds to the nearest float32.
                 */
                [[nodiscard]] std::vector<float> read_float32() const noexcept;

            private:
                /** @brief Holds the file's bytes and the array read from them. */
                NpyReader(SushiRuntime::span<const std::byte> bytes, NpyArray array) noexcept;

                /** @brief Holds a view of the whole file. */
                SushiRuntime::span<const std::byte> bytes_;

                /** @brief Holds the array the header describes. */
                NpyArray array_;
        };
    } // namespace IO
} // namespace SushiAI

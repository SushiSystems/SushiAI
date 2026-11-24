/**************************************************************************/
/* zip_writer.hpp                                                         */
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
 * @file zip_writer.hpp
 * @brief Declares the writer of a zip archive of stored entries, the inverse of ZipArchive.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @class ZipWriter
         * @brief Builds a zip archive of stored entries in memory.
         * Entries are method 0 with CRC-32 and date 1980-01-01; sizes, offsets or counts past
         * 32 or 16 bits use zip64 fields. Every method may throw std::bad_alloc, so a caller
         * returning a Result runs it inside a seam guard.
         */
        class ZipWriter final
        {
            public:
                /**
                 * @brief Appends an entry.
                 * @param name The path the archive stores it under, at most 65535 bytes.
                 * @param bytes Its contents, copied.
                 * @param alignment The data starts at a multiple of it, through a padding extra
                 *                  field; 1 for none.
                 */
                void add(std::string_view name, SushiRuntime::span<const std::byte> bytes,
                         std::size_t alignment = 1);

                /**
                 * @brief Appends the central directory and the end records.
                 * @return The whole archive; the writer is empty afterwards.
                 */
                [[nodiscard]] std::vector<std::byte> finish();

            private:
                /** @brief Holds what the central directory states of one entry. */
                struct Written
                {
                    std::string name;        /**< The stored path. */
                    std::uint32_t crc = 0;   /**< The CRC-32 of the contents. */
                    std::uint64_t size = 0;  /**< The bytes of the contents. */
                    std::uint64_t local = 0; /**< The offset of the local header. */
                };

                /** @brief Holds the archive written so far. */
                std::vector<std::byte> bytes_;

                /** @brief Holds the entries in the order they were added. */
                std::vector<Written> entries_;
        };
    } // namespace IO
} // namespace SushiAI

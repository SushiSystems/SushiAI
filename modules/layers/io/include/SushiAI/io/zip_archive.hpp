/**************************************************************************/
/* zip_archive.hpp                                                        */
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
 * @file zip_archive.hpp
 * @brief Declares the reader of the entries of a zip archive held in memory.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names one entry of a zip archive and where its bytes lie in the archive. */
        struct ZipEntry
        {
            /** @brief Holds the path the archive stores the entry under. */
            std::string name;

            /**
             * @brief Holds the offset of the entry's first byte from the archive's start.
             *
             * For a deflated entry it is the first byte of the compressed form.
             */
            std::size_t offset = 0;

            /** @brief Holds the entry's size in bytes, after decoding when it is deflated. */
            std::size_t size = 0;

            /** @brief Holds whether the archive stores the entry deflated, method 8. */
            bool deflated = false;
        };

        /**
         * @class ZipArchive
         * @brief Lists the entries of a zip archive and hands out views of their bytes.
         *
         * Stored and deflated entries are read, and any other method is refused by name. A
         * deflated entry is decoded and its CRC-32 checked when the archive is read; a stored
         * one is a view of the bytes the archive was read from.
         */
        class ZipArchive final
        {
            public:
                /**
                 * @brief Reads the central directory of @p bytes, plain or zip64.
                 * @param bytes The archive; must outlive the archive and every view of it.
                 * @param origin Names the archive in a message.
                 * @return InvalidFile for bytes that are not a whole zip archive and, naming the
                 *         entry, for a deflated one whose stream, size or CRC-32 is wrong;
                 *         Unsupported, naming the entry, for a method other than 0 and 8.
                 */
                [[nodiscard]] static Result<ZipArchive> read(
                    SushiRuntime::span<const std::byte> bytes, std::string_view origin) noexcept;

                /** @brief Returns the entry stored under @p name, or null when none is. */
                [[nodiscard]] const ZipEntry* find(std::string_view name) const noexcept;

                /** @brief Returns the entries in the order the central directory lists them. */
                [[nodiscard]] SushiRuntime::span<const ZipEntry> entries() const noexcept;

                /**
                 * @brief Returns the bytes of @p entry, decoded when it is deflated.
                 * @pre @p entry is one of entries().
                 */
                [[nodiscard]] SushiRuntime::span<const std::byte> contents(
                    const ZipEntry& entry) const noexcept;

            private:
                /** @brief Holds the archive's bytes, its entries and the bytes of its deflated ones. */
                ZipArchive(SushiRuntime::span<const std::byte> bytes,
                           std::vector<ZipEntry> entries,
                           std::vector<std::vector<std::byte>> decoded) noexcept;

                /** @brief Holds a view of the whole archive. */
                SushiRuntime::span<const std::byte> bytes_;

                /** @brief Holds the entries in central-directory order. */
                std::vector<ZipEntry> entries_;

                /** @brief Holds the decoded bytes of each entry, empty for a stored one. */
                std::vector<std::vector<std::byte>> decoded_;
        };
    } // namespace IO
} // namespace SushiAI

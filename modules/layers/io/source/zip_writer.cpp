/**************************************************************************/
/* zip_writer.cpp                                                         */
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
 * @file zip_writer.cpp
 * @brief Implements the writer of a zip archive of stored entries.
 * @author Mustafa Garip
 */

#include <SushiAI/io/zip_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include "crc32.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the signature of a local header. */
            constexpr std::uint32_t K_LOCAL_SIGNATURE = 0x04034b50u;

            /** @brief Holds the signature of a central-directory header. */
            constexpr std::uint32_t K_CENTRAL_SIGNATURE = 0x02014b50u;

            /** @brief Holds the signature of the end of the central directory. */
            constexpr std::uint32_t K_END_SIGNATURE = 0x06054b50u;

            /** @brief Holds the signature of the zip64 end of the central directory. */
            constexpr std::uint32_t K_END64_SIGNATURE = 0x06064b50u;

            /** @brief Holds the signature of the locator of the zip64 end record. */
            constexpr std::uint32_t K_LOCATOR64_SIGNATURE = 0x07064b50u;

            /** @brief Holds the bytes of a local header before its name. */
            constexpr std::size_t K_LOCAL_BYTES = 30;

            /** @brief Holds the identifier of the zip64 extra field. */
            constexpr std::uint16_t K_ZIP64_EXTRA = 0x0001;

            /** @brief Holds the identifier of the padding extra field PyTorch writes. */
            constexpr std::uint16_t K_PADDING_EXTRA = 0x4246;

            /** @brief Holds the 32-bit value that sends a field to the zip64 extra field. */
            constexpr std::uint32_t K_ZIP64_MARK = 0xFFFFFFFFu;

            /** @brief Holds the version a reader needs for a plain stored entry, 2.0. */
            constexpr std::uint16_t K_VERSION_PLAIN = 20;

            /** @brief Holds the version a reader needs for zip64, 4.5. */
            constexpr std::uint16_t K_VERSION_ZIP64 = 45;

            /** @brief Holds the MS-DOS date of 1980-01-01. */
            constexpr std::uint16_t K_DOS_DATE = 0x0021;

            /** @brief Appends the low @p count bytes of @p value, low byte first. */
            void put(std::vector<std::byte>& out, std::uint64_t value, std::size_t count)
            {
                for (std::size_t i = 0; i < count; ++i)
                    out.push_back(static_cast<std::byte>((value >> (8u * i)) & 0xFFu));
            }

            /** @brief Returns whether @p value needs the zip64 field. */
            [[nodiscard]] bool is_wide(std::uint64_t value) noexcept
            {
                return value >= K_ZIP64_MARK;
            }

            /** @brief Returns @p value, or the zip64 mark when it does not fit 32 bits. */
            [[nodiscard]] std::uint64_t narrow(std::uint64_t value) noexcept
            {
                return is_wide(value) ? K_ZIP64_MARK : value;
            }
        } // namespace

        void ZipWriter::add(std::string_view name, SushiRuntime::span<const std::byte> bytes,
                            std::size_t alignment)
        {
            Written entry;
            entry.name = std::string(name);
            entry.crc = Detail::crc32(bytes);
            entry.size = bytes.size();
            entry.local = bytes_.size();
            const bool wide = is_wide(entry.size);

            std::vector<std::byte> extra;
            if (wide)
            {
                put(extra, K_ZIP64_EXTRA, 2);
                put(extra, 16, 2);
                put(extra, entry.size, 8);
                put(extra, entry.size, 8);
            }
            if (alignment > 1)
            {
                const std::size_t unpadded = bytes_.size() + K_LOCAL_BYTES + name.size() +
                                             extra.size() + 4;
                const std::size_t padding = (alignment - unpadded % alignment) % alignment;
                put(extra, K_PADDING_EXTRA, 2);
                put(extra, padding, 2);
                extra.insert(extra.end(), padding, std::byte{0});
            }

            put(bytes_, K_LOCAL_SIGNATURE, 4);
            put(bytes_, wide ? K_VERSION_ZIP64 : K_VERSION_PLAIN, 2);
            put(bytes_, 0, 2);
            put(bytes_, 0, 2);
            put(bytes_, 0, 2);
            put(bytes_, K_DOS_DATE, 2);
            put(bytes_, entry.crc, 4);
            put(bytes_, narrow(entry.size), 4);
            put(bytes_, narrow(entry.size), 4);
            put(bytes_, name.size(), 2);
            put(bytes_, extra.size(), 2);
            bytes_.insert(bytes_.end(), reinterpret_cast<const std::byte*>(name.data()),
                          reinterpret_cast<const std::byte*>(name.data()) + name.size());
            bytes_.insert(bytes_.end(), extra.begin(), extra.end());
            bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
            entries_.push_back(std::move(entry));
        }

        std::vector<std::byte> ZipWriter::finish()
        {
            const std::uint64_t directory = bytes_.size();
            for (const Written& entry : entries_)
            {
                std::vector<std::byte> extra;
                if (is_wide(entry.size) || is_wide(entry.local))
                {
                    std::vector<std::byte> fields;
                    if (is_wide(entry.size))
                    {
                        put(fields, entry.size, 8);
                        put(fields, entry.size, 8);
                    }
                    if (is_wide(entry.local))
                        put(fields, entry.local, 8);
                    put(extra, K_ZIP64_EXTRA, 2);
                    put(extra, fields.size(), 2);
                    extra.insert(extra.end(), fields.begin(), fields.end());
                }
                const std::uint16_t version = extra.empty() ? K_VERSION_PLAIN : K_VERSION_ZIP64;

                put(bytes_, K_CENTRAL_SIGNATURE, 4);
                put(bytes_, version, 2);
                put(bytes_, version, 2);
                put(bytes_, 0, 2);
                put(bytes_, 0, 2);
                put(bytes_, 0, 2);
                put(bytes_, K_DOS_DATE, 2);
                put(bytes_, entry.crc, 4);
                put(bytes_, narrow(entry.size), 4);
                put(bytes_, narrow(entry.size), 4);
                put(bytes_, entry.name.size(), 2);
                put(bytes_, extra.size(), 2);
                put(bytes_, 0, 2);
                put(bytes_, 0, 2);
                put(bytes_, 0, 2);
                put(bytes_, 0, 4);
                put(bytes_, narrow(entry.local), 4);
                bytes_.insert(bytes_.end(), reinterpret_cast<const std::byte*>(entry.name.data()),
                              reinterpret_cast<const std::byte*>(entry.name.data()) +
                                  entry.name.size());
                bytes_.insert(bytes_.end(), extra.begin(), extra.end());
            }

            const std::uint64_t size = bytes_.size() - directory;
            const std::uint64_t count = entries_.size();
            const bool wide = count >= 0xFFFFu || is_wide(size) || is_wide(directory);
            if (wide)
            {
                const std::uint64_t record = bytes_.size();
                put(bytes_, K_END64_SIGNATURE, 4);
                put(bytes_, 44, 8);
                put(bytes_, K_VERSION_ZIP64, 2);
                put(bytes_, K_VERSION_ZIP64, 2);
                put(bytes_, 0, 4);
                put(bytes_, 0, 4);
                put(bytes_, count, 8);
                put(bytes_, count, 8);
                put(bytes_, size, 8);
                put(bytes_, directory, 8);

                put(bytes_, K_LOCATOR64_SIGNATURE, 4);
                put(bytes_, 0, 4);
                put(bytes_, record, 8);
                put(bytes_, 1, 4);
            }

            put(bytes_, K_END_SIGNATURE, 4);
            put(bytes_, 0, 2);
            put(bytes_, 0, 2);
            put(bytes_, wide ? 0xFFFFu : count, 2);
            put(bytes_, wide ? 0xFFFFu : count, 2);
            put(bytes_, wide ? K_ZIP64_MARK : size, 4);
            put(bytes_, wide ? K_ZIP64_MARK : directory, 4);
            put(bytes_, 0, 2);

            entries_.clear();
            return std::exchange(bytes_, std::vector<std::byte>());
        }
    } // namespace IO
} // namespace SushiAI

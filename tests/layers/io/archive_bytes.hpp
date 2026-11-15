/**************************************************************************/
/* archive_bytes.hpp                                                      */
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
 * @file archive_bytes.hpp
 * @brief Declares the helpers that write zip archives and pickle streams in memory for the io tests.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names one entry of a zip archive a test writes: its path and its bytes. */
        struct ZipFixtureEntry
        {
            std::string name;          /**< The path the archive stores it under. */
            std::string contents;      /**< Its bytes. */
            std::uint16_t method = 0;  /**< Its compression method; 0 stores it. */
        };

        /** @brief Returns the bytes of @p text, one byte per character. */
        inline std::vector<std::byte> bytes_of(std::string_view text)
        {
            std::vector<std::byte> out(text.size());
            for (std::size_t i = 0; i < text.size(); ++i)
                out[i] = static_cast<std::byte>(static_cast<unsigned char>(text[i]));
            return out;
        }

        /** @brief Returns a view of @p bytes. */
        inline SushiRuntime::span<const std::byte> view_of(const std::vector<std::byte>& bytes)
        {
            return SushiRuntime::span<const std::byte>(bytes.data(), bytes.size());
        }

        /** @brief Appends @p value to @p out as @p width little-endian bytes. */
        inline void put_le(std::string& out, std::uint64_t value, std::size_t width)
        {
            for (std::size_t i = 0; i < width; ++i)
                out.push_back(static_cast<char>((value >> (8u * i)) & 0xFFu));
        }

        /**
         * @brief Returns a zip archive of @p entries, each written as stored with no extra
         *        field, then a central directory and an end record with no comment.
         */
        inline std::vector<std::byte> zip_of(const std::vector<ZipFixtureEntry>& entries)
        {
            std::string out;
            std::vector<std::size_t> offsets;
            for (const ZipFixtureEntry& entry : entries)
            {
                offsets.push_back(out.size());
                put_le(out, 0x04034b50u, 4);
                put_le(out, 20, 2);
                put_le(out, 0, 2);
                put_le(out, entry.method, 2);
                put_le(out, 0, 4);
                put_le(out, 0, 4);
                put_le(out, entry.contents.size(), 4);
                put_le(out, entry.contents.size(), 4);
                put_le(out, entry.name.size(), 2);
                put_le(out, 0, 2);
                out += entry.name;
                out += entry.contents;
            }
            const std::size_t directory = out.size();
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                const ZipFixtureEntry& entry = entries[i];
                put_le(out, 0x02014b50u, 4);
                put_le(out, 20, 2);
                put_le(out, 20, 2);
                put_le(out, 0, 2);
                put_le(out, entry.method, 2);
                put_le(out, 0, 4);
                put_le(out, 0, 4);
                put_le(out, entry.contents.size(), 4);
                put_le(out, entry.contents.size(), 4);
                put_le(out, entry.name.size(), 2);
                put_le(out, 0, 2);
                put_le(out, 0, 2);
                put_le(out, 0, 2);
                put_le(out, 0, 2);
                put_le(out, 0, 4);
                put_le(out, offsets[i], 4);
                out += entry.name;
            }
            const std::size_t directory_size = out.size() - directory;
            put_le(out, 0x06054b50u, 4);
            put_le(out, 0, 2);
            put_le(out, 0, 2);
            put_le(out, entries.size(), 2);
            put_le(out, entries.size(), 2);
            put_le(out, directory_size, 4);
            put_le(out, directory, 4);
            put_le(out, 0, 2);
            return bytes_of(out);
        }
    } // namespace Test
} // namespace SushiAI

/**************************************************************************/
/* zip_archive.cpp                                                        */
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
 * @file zip_archive.cpp
 * @brief Defines the reader of the central directory and the stored entries of a zip archive.
 * @author Mustafa Garip
 */

#include <SushiAI/io/zip_archive.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the signature of the end of the central directory. */
            constexpr std::uint32_t K_END_SIGNATURE = 0x06054b50u;

            /** @brief Holds the signature of the zip64 end of the central directory. */
            constexpr std::uint32_t K_END64_SIGNATURE = 0x06064b50u;

            /** @brief Holds the signature of the locator of the zip64 end record. */
            constexpr std::uint32_t K_LOCATOR64_SIGNATURE = 0x07064b50u;

            /** @brief Holds the signature of a central-directory file header. */
            constexpr std::uint32_t K_CENTRAL_SIGNATURE = 0x02014b50u;

            /** @brief Holds the signature of a local file header. */
            constexpr std::uint32_t K_LOCAL_SIGNATURE = 0x04034b50u;

            /** @brief Holds the bytes of the end record before its comment. */
            constexpr std::size_t K_END_BYTES = 22;

            /** @brief Holds the bytes of the zip64 end locator. */
            constexpr std::size_t K_LOCATOR64_BYTES = 20;

            /** @brief Holds the bytes of the zip64 end record before its extensible data. */
            constexpr std::size_t K_END64_BYTES = 56;

            /** @brief Holds the bytes of a central-directory header before its name. */
            constexpr std::size_t K_CENTRAL_BYTES = 46;

            /** @brief Holds the bytes of a local header before its name. */
            constexpr std::size_t K_LOCAL_BYTES = 30;

            /** @brief Holds the longest comment an end record may carry. */
            constexpr std::size_t K_MAX_COMMENT = 0xFFFF;

            /** @brief Holds the identifier of the zip64 extra field. */
            constexpr std::uint16_t K_ZIP64_EXTRA = 0x0001;

            /** @brief Holds the 32-bit value that sends a field to the zip64 extra field. */
            constexpr std::uint32_t K_ZIP64_MARK = 0xFFFFFFFFu;

            /** @brief Holds the method number of a stored entry. */
            constexpr std::uint16_t K_STORED = 0;

            /** @brief Reads the little-endian integer of @p width bytes at @p at. */
            std::uint64_t read_le(SushiRuntime::span<const std::byte> bytes, std::size_t at,
                                  std::size_t width) noexcept
            {
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < width; ++i)
                    value |= static_cast<std::uint64_t>(bytes[at + i]) << (8u * i);
                return value;
            }

            /** @brief Returns whether @p count bytes from @p at lie inside @p bytes. */
            bool fits(SushiRuntime::span<const std::byte> bytes, std::uint64_t at,
                      std::uint64_t count) noexcept
            {
                return at <= bytes.size() && count <= bytes.size() - at;
            }

            /** @brief Holds where the central directory lies and how many entries it lists. */
            struct Directory
            {
                std::uint64_t offset = 0; /**< Its first byte. */
                std::uint64_t size = 0;   /**< Its length in bytes. */
                std::uint64_t count = 0;  /**< The entries it lists. */
            };

            /**
             * @brief Returns the offset of the end record, searching back over its comment.
             * @return InvalidFile when no end record is found.
             */
            Result<std::size_t> find_end(SushiRuntime::span<const std::byte> bytes,
                                         std::string_view origin)
            {
                SA_FAIL_IF(bytes.size() < K_END_BYTES, StatusCode::InvalidFile,
                           "'{}' holds {} bytes, fewer than a zip archive's end record", origin,
                           bytes.size());
                const std::size_t last = bytes.size() - K_END_BYTES;
                const std::size_t first = last > K_MAX_COMMENT ? last - K_MAX_COMMENT : 0;
                std::size_t at = last + 1;
                while (at > first && read_le(bytes, at - 1, 4) != K_END_SIGNATURE)
                    --at;
                SA_FAIL_IF(at == first, StatusCode::InvalidFile,
                           "'{}' is not a zip archive: it has no end of central directory record",
                           origin);
                return at - 1;
            }

            /**
             * @brief Reads the place of the central directory from the end record at @p end.
             * @return InvalidFile for a zip64 record that is missing or out of the file.
             */
            Result<Directory> read_directory(SushiRuntime::span<const std::byte> bytes,
                                             std::size_t end, std::string_view origin)
            {
                Directory directory;
                directory.count = read_le(bytes, end + 10, 2);
                directory.size = read_le(bytes, end + 12, 4);
                directory.offset = read_le(bytes, end + 16, 4);
                const bool wide = directory.count == 0xFFFFu || directory.size == K_ZIP64_MARK ||
                                  directory.offset == K_ZIP64_MARK;
                if (!wide)
                    return directory;

                SA_FAIL_IF(end < K_LOCATOR64_BYTES ||
                               read_le(bytes, end - K_LOCATOR64_BYTES, 4) != K_LOCATOR64_SIGNATURE,
                           StatusCode::InvalidFile,
                           "'{}' marks its central directory as zip64 but has no zip64 locator",
                           origin);
                const std::uint64_t record = read_le(bytes, end - K_LOCATOR64_BYTES + 8, 8);
                SA_FAIL_IF(!fits(bytes, record, K_END64_BYTES) ||
                               read_le(bytes, static_cast<std::size_t>(record), 4) !=
                                   K_END64_SIGNATURE,
                           StatusCode::InvalidFile,
                           "'{}' names a zip64 end record at byte {} that is not one", origin,
                           record);
                const std::size_t at = static_cast<std::size_t>(record);
                directory.count = read_le(bytes, at + 32, 8);
                directory.size = read_le(bytes, at + 40, 8);
                directory.offset = read_le(bytes, at + 48, 8);
                return directory;
            }

            /**
             * @brief Replaces each marked size or offset of an entry by its zip64 extra field.
             * @return InvalidFile when a marked field has no value in the extra field.
             */
            Status widen_entry(SushiRuntime::span<const std::byte> extra, std::uint64_t& size,
                               std::uint64_t& compressed, std::uint64_t& local,
                               const std::string& name, std::string_view origin)
            {
                std::size_t at = 0;
                while (at + 4 <= extra.size())
                {
                    const std::uint64_t id = read_le(extra, at, 2);
                    const std::size_t length = static_cast<std::size_t>(read_le(extra, at + 2, 2));
                    SA_FAIL_IF(!fits(extra, at + 4, length), StatusCode::InvalidFile,
                               "'{}': the extra field of '{}' runs past its end", origin, name);
                    if (id == K_ZIP64_EXTRA)
                    {
                        std::size_t field = at + 4;
                        std::uint64_t* const marked[] = {&size, &compressed, &local};
                        for (std::uint64_t* value : marked)
                        {
                            if (*value != K_ZIP64_MARK)
                                continue;
                            SA_FAIL_IF(field + 8 > at + 4 + length, StatusCode::InvalidFile,
                                       "'{}': the zip64 field of '{}' is too short", origin,
                                       name);
                            *value = read_le(extra, field, 8);
                            field += 8;
                        }
                        return Status();
                    }
                    at += 4 + length;
                }
                SA_FAIL_IF(size == K_ZIP64_MARK || compressed == K_ZIP64_MARK ||
                               local == K_ZIP64_MARK,
                           StatusCode::InvalidFile,
                           "'{}': the entry '{}' marks a zip64 size but has no zip64 field",
                           origin, name);
                return Status();
            }

            /**
             * @brief Reads the central-directory header at @p at into an entry.
             * @param at Advanced past the header, its name, extra field and comment.
             * @return InvalidFile for a header out of the file; Unsupported for a compressed
             *         entry.
             */
            Result<ZipEntry> read_entry(SushiRuntime::span<const std::byte> bytes,
                                        std::size_t& at, std::string_view origin)
            {
                SA_FAIL_IF(!fits(bytes, at, K_CENTRAL_BYTES) ||
                               read_le(bytes, at, 4) != K_CENTRAL_SIGNATURE,
                           StatusCode::InvalidFile,
                           "'{}': byte {} should open a central-directory header", origin, at);
                const std::uint64_t method = read_le(bytes, at + 10, 2);
                std::uint64_t compressed = read_le(bytes, at + 20, 4);
                std::uint64_t size = read_le(bytes, at + 24, 4);
                const std::size_t name_length = static_cast<std::size_t>(read_le(bytes, at + 28, 2));
                const std::size_t extra_length =
                    static_cast<std::size_t>(read_le(bytes, at + 30, 2));
                const std::size_t comment_length =
                    static_cast<std::size_t>(read_le(bytes, at + 32, 2));
                std::uint64_t local = read_le(bytes, at + 42, 4);
                const std::size_t name_at = at + K_CENTRAL_BYTES;
                SA_FAIL_IF(!fits(bytes, name_at, name_length + extra_length + comment_length),
                           StatusCode::InvalidFile,
                           "'{}': the central-directory header at byte {} runs past the file",
                           origin, at);

                ZipEntry entry;
                entry.name.assign(reinterpret_cast<const char*>(bytes.data() + name_at),
                                  name_length);
                SA_TRY(widen_entry(bytes.subspan(name_at + name_length, extra_length), size,
                                   compressed, local, entry.name, origin));
                at = name_at + name_length + extra_length + comment_length;

                SA_FAIL_IF(method != K_STORED, StatusCode::Unsupported,
                           "'{}': the entry '{}' is compressed with method {}, and only stored "
                           "entries are read",
                           origin, entry.name, method);
                SA_FAIL_IF(compressed != size, StatusCode::InvalidFile,
                           "'{}': the stored entry '{}' states {} bytes stored and {} bytes long",
                           origin, entry.name, compressed, size);
                SA_FAIL_IF(!fits(bytes, local, K_LOCAL_BYTES) ||
                               read_le(bytes, static_cast<std::size_t>(local), 4) !=
                                   K_LOCAL_SIGNATURE,
                           StatusCode::InvalidFile,
                           "'{}': the entry '{}' names a local header at byte {} that is not one",
                           origin, entry.name, local);
                const std::size_t local_at = static_cast<std::size_t>(local);
                const std::uint64_t data = local + K_LOCAL_BYTES + read_le(bytes, local_at + 26, 2) +
                                           read_le(bytes, local_at + 28, 2);
                SA_FAIL_IF(!fits(bytes, data, size), StatusCode::InvalidFile,
                           "'{}': the {} bytes of the entry '{}' run past the file", origin, size,
                           entry.name);
                entry.offset = static_cast<std::size_t>(data);
                entry.size = static_cast<std::size_t>(size);
                return entry;
            }
        } // namespace

        ZipArchive::ZipArchive(SushiRuntime::span<const std::byte> bytes,
                               std::vector<ZipEntry> entries) noexcept
            : bytes_(bytes), entries_(std::move(entries))
        {

        }

        Result<ZipArchive> ZipArchive::read(SushiRuntime::span<const std::byte> bytes,
                                            std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const std::size_t end, find_end(bytes, origin));
            SA_TRY_ASSIGN(const Directory directory, read_directory(bytes, end, origin));
            SA_FAIL_IF(!fits(bytes, directory.offset, directory.size), StatusCode::InvalidFile,
                       "'{}' places its central directory of {} bytes at byte {}, past the file",
                       origin, directory.size, directory.offset);
            SA_FAIL_IF(directory.count > directory.size / K_CENTRAL_BYTES,
                       StatusCode::InvalidFile,
                       "'{}' lists {} entries in a central directory of {} bytes", origin,
                       directory.count, directory.size);

            std::vector<ZipEntry> entries;
            entries.reserve(static_cast<std::size_t>(directory.count));
            std::size_t at = static_cast<std::size_t>(directory.offset);
            for (std::uint64_t i = 0; i < directory.count; ++i)
            {
                SA_TRY_ASSIGN(ZipEntry entry, read_entry(bytes, at, origin));
                entries.push_back(std::move(entry));
            }
            return ZipArchive(bytes, std::move(entries));
        }

        const ZipEntry* ZipArchive::find(std::string_view name) const noexcept
        {
            for (const ZipEntry& entry : entries_)
            {
                if (entry.name == name)
                    return &entry;
            }
            return nullptr;
        }

        SushiRuntime::span<const ZipEntry> ZipArchive::entries() const noexcept
        {
            return SushiRuntime::span<const ZipEntry>(entries_.data(), entries_.size());
        }

        SushiRuntime::span<const std::byte> ZipArchive::contents(const ZipEntry& entry) const noexcept
        {
            return bytes_.subspan(entry.offset, entry.size);
        }
    } // namespace IO
} // namespace SushiAI

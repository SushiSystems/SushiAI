/**************************************************************************/
/* token_file.cpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file token_file.cpp
 * @brief Implements the writer and the readers of the token file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/token_file.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <ios>
#include <limits>
#include <ostream>


#include "disk.hpp"
#include "little_endian.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names the writer in a message. */
        constexpr std::string_view K_WRITER = "write_token_file";

        /** @brief Names the reader of the header in a message. */
        constexpr std::string_view K_HEADER_READER = "read_token_file_header";

        /** @brief Names the reader of a range in a message. */
        constexpr std::string_view K_RANGE_READER = "read_token_range";

        /** @brief Holds the offset of the format version in the header. */
        constexpr std::size_t K_VERSION_OFFSET = 4;

        /** @brief Holds the offset of the vocabulary size in the header. */
        constexpr std::size_t K_VOCABULARY_OFFSET = 8;

        /** @brief Holds the offset of the byte count of one id in the header. */
        constexpr std::size_t K_ID_BYTES_OFFSET = 12;

        /** @brief Holds the offset of the id count in the header. */
        constexpr std::size_t K_COUNT_OFFSET = 16;

        /** @brief Holds how many ids the writer converts before it writes them. */
        constexpr std::size_t K_WRITE_CHUNK_IDS = 16384;

        /** @brief Holds the most ids a file has, so that its byte count fits 64 bits. */
        constexpr std::uint64_t K_MAX_TOKEN_COUNT =
            (std::numeric_limits<std::uint64_t>::max() - K_TOKEN_FILE_HEADER_BYTES) /
            K_TOKEN_FILE_ID_BYTES;

        /** @brief Holds the bytes of one header. */
        using HeaderBytes = std::array<char, K_TOKEN_FILE_HEADER_BYTES>;

        /** @brief Returns the byte offset of an id of a token file. */
        [[nodiscard]] constexpr std::uint64_t offset_of(std::uint64_t index) noexcept
        {
            return K_TOKEN_FILE_HEADER_BYTES + index * K_TOKEN_FILE_ID_BYTES;
        }

        /** @brief Reports whether a vocabulary size is one a token file may state. */
        [[nodiscard]] constexpr bool is_vocabulary_size(std::int64_t size) noexcept
        {
            return size >= 1 && size <= Text::K_MAX_VOCABULARY_SIZE;
        }

        /** @brief Fails for the first id outside [0, vocabulary_size), naming its index. */
        [[nodiscard]] Status check_ids(SushiRuntime::span<const Text::TokenID> ids,
                                       std::int32_t vocabulary_size)
        {
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                SA_FAIL_IF(ids[i] < 0 || ids[i] >= vocabulary_size, StatusCode::InvalidDescription,
                           "{}: the id {} at index {} is outside [0, {})", K_WRITER, ids[i], i,
                           vocabulary_size);
            }

            return Status();
        }

        /** @brief Writes the header and the ids of a version 1 file to a stream. */
        void write_version_1(std::ostream& out, SushiRuntime::span<const Text::TokenID> ids,
                             std::int32_t vocabulary_size)
        {
            HeaderBytes header;
            K_TOKEN_FILE_MAGIC.copy(header.data(), K_TOKEN_FILE_MAGIC.size());
            Detail::store_u32(header.data() + K_VERSION_OFFSET, K_TOKEN_FILE_VERSION);
            Detail::store_u32(header.data() + K_VOCABULARY_OFFSET,
                              static_cast<std::uint32_t>(vocabulary_size));
            Detail::store_u32(header.data() + K_ID_BYTES_OFFSET,
                              static_cast<std::uint32_t>(K_TOKEN_FILE_ID_BYTES));
            Detail::store_u64(header.data() + K_COUNT_OFFSET, ids.size());
            out.write(header.data(), static_cast<std::streamsize>(header.size()));

            std::array<char, K_WRITE_CHUNK_IDS * K_TOKEN_FILE_ID_BYTES> chunk;

            for (std::size_t done = 0; done < ids.size();)
            {
                const std::size_t count = std::min(K_WRITE_CHUNK_IDS, ids.size() - done);
                for (std::size_t i = 0; i < count; ++i)
                {
                    Detail::store_u32(chunk.data() + i * K_TOKEN_FILE_ID_BYTES,
                                      static_cast<std::uint32_t>(ids[done + i]));
                }

                out.write(chunk.data(),
                          static_cast<std::streamsize>(count * K_TOKEN_FILE_ID_BYTES));
                done += count;
            }
        }

        /**
         * @brief Returns the header of a version 1 file after checking it against the file.
         * @param file_bytes The byte count of the whole file.
         */
        [[nodiscard]] Result<TokenFileHeader> parse_version_1(const HeaderBytes& bytes,
                                                              std::uint64_t file_bytes)
        {
            const std::uint32_t vocabulary = Detail::load_u32(bytes.data() + K_VOCABULARY_OFFSET);
            const std::uint32_t id_bytes = Detail::load_u32(bytes.data() + K_ID_BYTES_OFFSET);
            const std::uint64_t count = Detail::load_u64(bytes.data() + K_COUNT_OFFSET);

            SA_FAIL_IF(!is_vocabulary_size(vocabulary), StatusCode::InvalidFile,
                       "{}: byte {} states a vocabulary of {} tokens; it has 1 to {}",
                       K_HEADER_READER, K_VOCABULARY_OFFSET, vocabulary,
                       Text::K_MAX_VOCABULARY_SIZE);
            SA_FAIL_IF(id_bytes != K_TOKEN_FILE_ID_BYTES, StatusCode::InvalidFile,
                       "{}: byte {} states ids of {} bytes; an id has {}", K_HEADER_READER,
                       K_ID_BYTES_OFFSET, id_bytes, K_TOKEN_FILE_ID_BYTES);
            SA_FAIL_IF(count > K_MAX_TOKEN_COUNT, StatusCode::InvalidFile,
                       "{}: byte {} states {} ids; a file has at most {}", K_HEADER_READER,
                       K_COUNT_OFFSET, count, K_MAX_TOKEN_COUNT);

            const std::uint64_t declared = offset_of(count);

            SA_FAIL_IF(file_bytes < declared, StatusCode::InvalidFile,
                       "{}: the file ends at byte {}; its header declares {} bytes",
                       K_HEADER_READER, file_bytes, declared);
            SA_FAIL_IF(file_bytes > declared, StatusCode::InvalidFile,
                       "{}: {} bytes follow the end the header declares at byte {}",
                       K_HEADER_READER, file_bytes - declared, declared);

            TokenFileHeader header;
            header.vocabulary_size = static_cast<std::int32_t>(vocabulary);
            header.token_count = count;
            return header;
        }

        /** @brief Reads and checks the header of an open file of a known byte count. */
        [[nodiscard]] Result<TokenFileHeader> read_header(std::istream& in,
                                                          std::uint64_t file_bytes)
        {
            SA_FAIL_IF(file_bytes < K_TOKEN_FILE_HEADER_BYTES, StatusCode::InvalidFile,
                       "{}: the file ends at byte {}, inside its header of {} bytes",
                       K_HEADER_READER, file_bytes, K_TOKEN_FILE_HEADER_BYTES);

            HeaderBytes bytes;
            in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(bytes.size()),
                       StatusCode::InvalidFile, "{}: the read ended at byte {} of the header",
                       K_HEADER_READER, in.gcount());
            SA_FAIL_IF(std::string_view(bytes.data(), K_TOKEN_FILE_MAGIC.size()) !=
                           K_TOKEN_FILE_MAGIC,
                       StatusCode::InvalidFile,
                       "{}: the file does not open with '{}'; it is no token file",
                       K_HEADER_READER, K_TOKEN_FILE_MAGIC);

            const std::uint32_t version = Detail::load_u32(bytes.data() + K_VERSION_OFFSET);
            SA_FAIL_IF(version < 1 || version > K_TOKEN_FILE_VERSION, StatusCode::InvalidFile,
                       "{}: byte {} states format version {}; this build reads versions 1 to {}",
                       K_HEADER_READER, K_VERSION_OFFSET, version, K_TOKEN_FILE_VERSION);

            return parse_version_1(bytes, file_bytes);
        }

        /** @brief Reads a range of ids from a stream; may throw what the stream throws. */
        [[nodiscard]] Status read_range(std::istream& in, const TokenFileHeader& header,
                                        std::uint64_t first, SushiRuntime::span<Text::TokenID> ids)
        {
            SA_FAIL_IF(first > header.token_count || ids.size() > header.token_count - first,
                       StatusCode::InvalidDescription,
                       "{}: {} ids from id {} were asked of a file of {}", K_RANGE_READER,
                       ids.size(), first, header.token_count);
            SA_FAIL_IF(!is_vocabulary_size(header.vocabulary_size), StatusCode::InvalidDescription,
                       "{}: the header states a vocabulary of {} tokens; it has 1 to {}",
                       K_RANGE_READER, header.vocabulary_size, Text::K_MAX_VOCABULARY_SIZE);

            const std::uint64_t begin = offset_of(first);
            const std::uint64_t byte_count = ids.size() * K_TOKEN_FILE_ID_BYTES;
            const std::uint64_t reach =
                static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max());

            SA_FAIL_IF(begin > reach || byte_count > reach - begin, StatusCode::Unsupported,
                       "{}: byte {} is beyond what a stream can seek to", K_RANGE_READER,
                       begin + byte_count);

            in.clear();
            in.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
            SA_FAIL_IF(!in, StatusCode::InvalidFile, "{}: the stream cannot seek to byte {}",
                       K_RANGE_READER, begin);

            char* const bytes = reinterpret_cast<char*>(ids.data());
            in.read(bytes, static_cast<std::streamsize>(byte_count));
            SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(byte_count),
                       StatusCode::InvalidFile,
                       "{}: the file ends at byte {}; the ids asked for end at byte {}",
                       K_RANGE_READER, begin + static_cast<std::uint64_t>(in.gcount()),
                       begin + byte_count);

            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                // The four bytes of an id are read before the id is written over them.
                const Text::TokenID id = static_cast<Text::TokenID>(
                    Detail::load_u32(bytes + i * K_TOKEN_FILE_ID_BYTES));

                SA_FAIL_IF(id < 0 || id >= header.vocabulary_size, StatusCode::InvalidFile,
                           "{}: the id {} at byte {} is outside [0, {})", K_RANGE_READER, id,
                           offset_of(first + i), header.vocabulary_size);
                ids[i] = id;
            }

            return Status();
        }
    } // namespace

    Status write_token_file(const std::filesystem::path& file,
                            SushiRuntime::span<const Text::TokenID> ids,
                            std::int32_t vocabulary_size)
    {
        SA_FAIL_IF(!is_vocabulary_size(vocabulary_size), StatusCode::InvalidDescription,
                   "{}: a vocabulary of {} tokens; it has 1 to {}", K_WRITER, vocabulary_size,
                   Text::K_MAX_VOCABULARY_SIZE);
        SA_TRY(check_ids(ids, vocabulary_size));

        return Detail::at_boundary(K_WRITER, [&]() -> Status
        {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            SA_FAIL_IF(!out.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for writing",
                       Detail::display_of(file), K_WRITER);

            write_version_1(out, ids, vocabulary_size);
            out.flush();
            SA_FAIL_IF(!out, StatusCode::InvalidFile,
                       "'{}': {}: the stream failed while {} bytes were written",
                       Detail::display_of(file), K_WRITER, offset_of(ids.size()));

            return Status();
        }, file);
    }

    Result<TokenFileHeader> read_token_file_header(const std::filesystem::path& file)
    {
        return Detail::at_boundary(K_HEADER_READER, [&]() -> Result<TokenFileHeader>
        {
            SA_TRY_ASSIGN(const std::uint64_t file_bytes,
                          Detail::size_of_file(file, K_HEADER_READER));

            std::ifstream in(file, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading",
                       Detail::display_of(file), K_HEADER_READER);

            Result<TokenFileHeader> header = read_header(in, file_bytes);
            if (!header.has_value())
                return SA_ANNOTATE(header.status(), "'{}'", Detail::display_of(file));

            return header;
        }, file);
    }

    Status read_token_range(std::istream& in, const TokenFileHeader& header, std::uint64_t first,
                            SushiRuntime::span<Text::TokenID> ids)
    {
        return Detail::at_boundary(K_RANGE_READER, [&]() -> Status
        {
            return read_range(in, header, first, ids);
        });
    }

    Status read_token_range(const std::filesystem::path& file, const TokenFileHeader& header,
                            std::uint64_t first, SushiRuntime::span<Text::TokenID> ids)
    {
        return Detail::at_boundary(K_RANGE_READER, [&]() -> Status
        {
            std::ifstream in(file, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading",
                       Detail::display_of(file), K_RANGE_READER);

            return SA_ANNOTATE(read_range(in, header, first, ids), "'{}'",
                               Detail::display_of(file));
        }, file);
    }
} // namespace SushiAI::Language::Datasets

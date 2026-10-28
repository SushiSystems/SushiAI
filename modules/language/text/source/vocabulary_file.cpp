/**************************************************************************/
/* vocabulary_file.cpp                                                    */
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
 * @file vocabulary_file.cpp
 * @brief Implements the reader and the writer of the vocabulary file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/text/vocabulary_file.hpp>

#include <limits>
#include <string>
#include <utility>
#include <vector>


#include <SushiAI/language/text/token.hpp>

#include "disk.hpp"
#include "little_endian.hpp"

namespace SushiAI::Language::Text
{
    namespace
    {
        /** @brief Names the writer in a message. */
        constexpr std::string_view K_WRITER = "write_vocabulary_file";

        /** @brief Names the reader in a message. */
        constexpr std::string_view K_READER = "read_vocabulary_file";

        /** @brief Holds the offset of the format version in the header. */
        constexpr std::size_t K_VERSION_OFFSET = 4;

        /** @brief Holds the offset of the byte token count in the header. */
        constexpr std::size_t K_BYTE_COUNT_OFFSET = 8;

        /** @brief Holds the offset of the special token count in the header. */
        constexpr std::size_t K_SPECIAL_COUNT_OFFSET = 12;

        /** @brief Holds the offset of the merge count in the header. */
        constexpr std::size_t K_MERGE_COUNT_OFFSET = 16;

        /** @brief Holds the offset of the byte count of the special section in the header. */
        constexpr std::size_t K_SPECIAL_BYTES_OFFSET = 20;

        /** @brief Holds the byte count of one merge record: its left and its right token. */
        constexpr std::size_t K_MERGE_BYTES = 2 * Detail::K_WORD_BYTES;

        /** @brief Holds the most bytes the special section has. */
        constexpr std::uint64_t K_MAX_SPECIAL_BYTES =
            static_cast<std::uint64_t>(K_MAX_SPECIAL_TOKENS) *
            (Detail::K_WORD_BYTES + K_MAX_SPECIAL_NAME_BYTES);

        /** @brief Holds the most bytes a vocabulary file has. */
        constexpr std::uint64_t K_MAX_FILE_BYTES =
            K_VOCABULARY_FILE_HEADER_BYTES + K_MAX_SPECIAL_BYTES +
            static_cast<std::uint64_t>(K_MAX_VOCABULARY_SIZE) * K_MERGE_BYTES;

        /** @brief Appends a value as four little-endian bytes. */
        void append_u32(std::string& bytes, std::uint32_t value)
        {
            char word[Detail::K_WORD_BYTES];
            Detail::store_u32(word, value);
            bytes.append(word, Detail::K_WORD_BYTES);
        }

        /** @brief Returns the bytes of a version 1 file that holds a vocabulary. */
        [[nodiscard]] std::string serialise(const Vocabulary& vocabulary)
        {
            const std::vector<std::string>& names = vocabulary.special_names();
            const std::vector<Merge>& merges = vocabulary.merges();

            std::size_t special_bytes = 0;
            for (const std::string& name : names)
                special_bytes += Detail::K_WORD_BYTES + name.size();

            std::string bytes;
            bytes.reserve(K_VOCABULARY_FILE_HEADER_BYTES + special_bytes +
                          merges.size() * K_MERGE_BYTES);

            bytes.append(K_VOCABULARY_FILE_MAGIC);
            append_u32(bytes, K_VOCABULARY_FILE_VERSION);
            append_u32(bytes, static_cast<std::uint32_t>(K_BYTE_TOKEN_COUNT));
            append_u32(bytes, static_cast<std::uint32_t>(names.size()));
            append_u32(bytes, static_cast<std::uint32_t>(merges.size()));
            append_u32(bytes, static_cast<std::uint32_t>(special_bytes));

            for (const std::string& name : names)
            {
                append_u32(bytes, static_cast<std::uint32_t>(name.size()));
                bytes.append(name);
            }

            for (const Merge merge : merges)
            {
                append_u32(bytes, static_cast<std::uint32_t>(merge.left));
                append_u32(bytes, static_cast<std::uint32_t>(merge.right));
            }

            return bytes;
        }

        /**
         * @brief Reads the names of the special section of a version 1 file.
         * @param end Where the section ends in @p bytes.
         * @param names Receives the names after end-of-text.
         * @return Success; InvalidFile, naming the byte offset, for a record out of bounds.
         */
        [[nodiscard]] Status parse_special_names(std::string_view bytes, std::size_t count,
                                                 std::size_t end, std::vector<std::string>& names)
        {
            std::size_t cursor = K_VOCABULARY_FILE_HEADER_BYTES;

            for (std::size_t i = 0; i < count; ++i)
            {
                SA_FAIL_IF(end - cursor < Detail::K_WORD_BYTES, StatusCode::InvalidFile,
                           "{}: special token {} begins at byte {}, past the special section, "
                           "which ends at byte {}",
                           K_READER, i, cursor, end);

                const std::size_t length = Detail::load_u32(bytes.data() + cursor);
                SA_FAIL_IF(length < 1 || length > K_MAX_SPECIAL_NAME_BYTES, StatusCode::InvalidFile,
                           "{}: special token {} at byte {} has a name of {} bytes; it has 1 to {}",
                           K_READER, i, cursor, length, K_MAX_SPECIAL_NAME_BYTES);

                cursor += Detail::K_WORD_BYTES;
                SA_FAIL_IF(end - cursor < length, StatusCode::InvalidFile,
                           "{}: the name of special token {} at byte {} holds {} bytes, past the "
                           "special section, which ends at byte {}",
                           K_READER, i, cursor, length, end);

                const std::string_view name = bytes.substr(cursor, length);
                SA_FAIL_IF(i == 0 && name != K_END_OF_TEXT_NAME, StatusCode::InvalidFile,
                           "{}: the first special token at byte {} is '{}', not '{}'", K_READER,
                           cursor, name, K_END_OF_TEXT_NAME);

                if (i > 0)
                    names.emplace_back(name);

                cursor += length;
            }

            SA_FAIL_IF(cursor != end, StatusCode::InvalidFile,
                       "{}: the special tokens end at byte {}; their section ends at byte {}",
                       K_READER, cursor, end);
            return Status();
        }

        /**
         * @brief Reads the merge records of a version 1 file.
         * @param begin Where the records begin in @p bytes.
         * @return The merges; InvalidFile, naming the byte offset, for an id above what a
         *         token id holds.
         */
        [[nodiscard]] Result<std::vector<Merge>> parse_merges(std::string_view bytes,
                                                             std::size_t begin, std::size_t count)
        {
            /** @brief Holds the largest value a token id has. */
            constexpr std::uint32_t K_LARGEST_ID =
                static_cast<std::uint32_t>(std::numeric_limits<TokenID>::max());

            std::vector<Merge> merges;
            merges.reserve(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t offset = begin + i * K_MERGE_BYTES;
                const std::uint32_t left = Detail::load_u32(bytes.data() + offset);
                const std::uint32_t right =
                    Detail::load_u32(bytes.data() + offset + Detail::K_WORD_BYTES);

                SA_FAIL_IF(left > K_LARGEST_ID || right > K_LARGEST_ID, StatusCode::InvalidFile,
                           "{}: merge {} at byte {} joins the tokens {} and {}; an id is at "
                           "most {}",
                           K_READER, i, offset, left, right, K_LARGEST_ID);

                merges.push_back(Merge{static_cast<TokenID>(left), static_cast<TokenID>(right)});
            }

            return Result<std::vector<Merge>>(std::move(merges));
        }

        /**
         * @brief Returns the vocabulary of a file whose header states format version 1.
         * @pre @p bytes holds a whole header.
         */
        [[nodiscard]] Result<Vocabulary> parse_version_1(std::string_view bytes)
        {
            const std::uint32_t byte_count = Detail::load_u32(bytes.data() + K_BYTE_COUNT_OFFSET);
            const std::uint32_t specials = Detail::load_u32(bytes.data() + K_SPECIAL_COUNT_OFFSET);
            const std::uint32_t merges = Detail::load_u32(bytes.data() + K_MERGE_COUNT_OFFSET);
            const std::uint32_t special_bytes =
                Detail::load_u32(bytes.data() + K_SPECIAL_BYTES_OFFSET);

            SA_FAIL_IF(byte_count != static_cast<std::uint32_t>(K_BYTE_TOKEN_COUNT),
                       StatusCode::InvalidFile, "{}: byte {} states {} byte tokens; there are {}",
                       K_READER, K_BYTE_COUNT_OFFSET, byte_count, K_BYTE_TOKEN_COUNT);
            SA_FAIL_IF(specials < 1 || specials > static_cast<std::uint32_t>(K_MAX_SPECIAL_TOKENS),
                       StatusCode::InvalidFile,
                       "{}: byte {} states {} special tokens; there are 1 to {}", K_READER,
                       K_SPECIAL_COUNT_OFFSET, specials, K_MAX_SPECIAL_TOKENS);

            const std::uint32_t room =
                static_cast<std::uint32_t>(K_MAX_VOCABULARY_SIZE - K_BYTE_TOKEN_COUNT) - specials;

            SA_FAIL_IF(merges > room, StatusCode::InvalidFile,
                       "{}: byte {} states {} merges; {} special tokens leave room for {}",
                       K_READER, K_MERGE_COUNT_OFFSET, merges, specials, room);
            SA_FAIL_IF(special_bytes > K_MAX_SPECIAL_BYTES, StatusCode::InvalidFile,
                       "{}: byte {} states a special section of {} bytes; it has at most {}",
                       K_READER, K_SPECIAL_BYTES_OFFSET, special_bytes, K_MAX_SPECIAL_BYTES);

            const std::size_t special_end = K_VOCABULARY_FILE_HEADER_BYTES + special_bytes;
            const std::size_t declared = special_end + merges * K_MERGE_BYTES;

            SA_FAIL_IF(bytes.size() < declared, StatusCode::InvalidFile,
                       "{}: the file ends at byte {}; its header declares {} bytes", K_READER,
                       bytes.size(), declared);
            SA_FAIL_IF(bytes.size() > declared, StatusCode::InvalidFile,
                       "{}: {} bytes follow the end the header declares at byte {}", K_READER,
                       bytes.size() - declared, declared);

            std::vector<std::string> names;
            SA_TRY(parse_special_names(bytes, specials, special_end, names));
            SA_TRY_ASSIGN(std::vector<Merge> pairs, parse_merges(bytes, special_end, merges));

            return Vocabulary::make(std::move(pairs), std::move(names));
        }
    } // namespace

    Status write_vocabulary_file(const std::filesystem::path& file, const Vocabulary& vocabulary) noexcept
    {
        return Detail::at_boundary(K_WRITER, [&]() -> Status
        {
            return Detail::write_file(file, serialise(vocabulary), K_WRITER);
        }, file);
    }

    Result<Vocabulary> parse_vocabulary_file(std::string_view bytes) noexcept
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<Vocabulary>
        {
            SA_FAIL_IF(bytes.size() < K_VOCABULARY_FILE_HEADER_BYTES, StatusCode::InvalidFile,
                       "{}: the file ends at byte {}, inside its header of {} bytes", K_READER,
                       bytes.size(), K_VOCABULARY_FILE_HEADER_BYTES);
            SA_FAIL_IF(bytes.substr(0, K_VOCABULARY_FILE_MAGIC.size()) != K_VOCABULARY_FILE_MAGIC,
                       StatusCode::InvalidFile,
                       "{}: the file does not open with '{}'; it is no vocabulary file", K_READER,
                       K_VOCABULARY_FILE_MAGIC);

            const std::uint32_t version = Detail::load_u32(bytes.data() + K_VERSION_OFFSET);
            SA_FAIL_IF(version < 1 || version > K_VOCABULARY_FILE_VERSION, StatusCode::InvalidFile,
                       "{}: byte {} states format version {}; this build reads versions 1 to {}",
                       K_READER, K_VERSION_OFFSET, version, K_VOCABULARY_FILE_VERSION);

            return parse_version_1(bytes);
        });
    }

    Result<Vocabulary> read_vocabulary_file(const std::filesystem::path& file) noexcept
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<Vocabulary>
        {
            SA_TRY_ASSIGN(const std::string bytes,
                          Detail::read_file(file, K_MAX_FILE_BYTES, K_READER));

            Result<Vocabulary> parsed = parse_vocabulary_file(bytes);
            if (!parsed.has_value())
                return SA_ANNOTATE(parsed.status(), "'{}'", Detail::display_of(file));

            return parsed;
        }, file);
    }
} // namespace SushiAI::Language::Text

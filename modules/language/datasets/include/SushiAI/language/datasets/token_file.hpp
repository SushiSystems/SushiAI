/**************************************************************************/
/* token_file.hpp                                                         */
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
 * @file token_file.hpp
 * @brief Declares the token file: the ids of a tokenised corpus, written once and read in ranges.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <string_view>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/text/token.hpp>

/** @brief Holds the token file of a corpus and the dataset that reads it in windows. */
namespace SushiAI::Language::Datasets
{
    /** @brief Holds the four bytes a token file opens with. */
    inline constexpr std::string_view K_TOKEN_FILE_MAGIC = "SATK";

    /** @brief Holds the format version this build writes, and the newest it reads. */
    inline constexpr std::uint32_t K_TOKEN_FILE_VERSION = 1;

    /** @brief Holds the byte count of the header of a token file. */
    inline constexpr std::uint64_t K_TOKEN_FILE_HEADER_BYTES = 24;

    /** @brief Holds the byte count of one id of a token file. */
    inline constexpr std::uint64_t K_TOKEN_FILE_ID_BYTES = 4;

    /** @brief Holds what the header of a token file states about its ids. */
    struct TokenFileHeader
    {
        /** @brief Counts the tokens an id may name; every id is in [0, vocabulary_size). */
        std::int32_t vocabulary_size = 0;

        /** @brief Counts the ids of the file. */
        std::uint64_t token_count = 0;
    };

    /**
     * @brief Writes ids in the current format version, replacing the file.
     * @param vocabulary_size In [1, Text::K_MAX_VOCABULARY_SIZE].
     * @return Success; InvalidDescription, naming the index, for an id outside
     *         [0, vocabulary_size), before the file is opened; InvalidFile, opening with
     *         the path, when the file cannot be opened or written.
     * @see modules/language/datasets/README.md, section "The token file".
     */
    [[nodiscard]] Status write_token_file(const std::filesystem::path& file,
                                          SushiRuntime::span<const Text::TokenID> ids,
                                          std::int32_t vocabulary_size);

    /**
     * @brief Reads and checks the header of a token file of any version this build knows.
     * @return The header; FileNotFound; InvalidFile, opening with the path and naming the
     *         byte offset, for a file that ends before or goes on past the end its header
     *         declares, or for a field out of bounds.
     */
    [[nodiscard]] Result<TokenFileHeader> read_token_file_header(
        const std::filesystem::path& file);

    /**
     * @brief Reads the ids [first, first + ids.size()) of a token file from its stream.
     * @param in A binary stream over the file @p header was read from.
     * @param ids Receives the ids; unspecified after a failure.
     * @return Success; InvalidDescription for a range outside the file; InvalidFile, naming
     *         the byte offset, when the stream ends early or an id is outside
     *         [0, header.vocabulary_size).
     */
    [[nodiscard]] Status read_token_range(std::istream& in, const TokenFileHeader& header,
                                          std::uint64_t first,
                                          SushiRuntime::span<Text::TokenID> ids);

    /**
     * @brief Opens a token file and reads the ids [first, first + ids.size()) of it.
     * @return What the stream overload returns, opening with the path; InvalidFile when the
     *         file cannot be opened.
     */
    [[nodiscard]] Status read_token_range(const std::filesystem::path& file,
                                          const TokenFileHeader& header, std::uint64_t first,
                                          SushiRuntime::span<Text::TokenID> ids);
} // namespace SushiAI::Language::Datasets

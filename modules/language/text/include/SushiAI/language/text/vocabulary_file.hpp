/**************************************************************************/
/* vocabulary_file.hpp                                                    */
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
 * @file vocabulary_file.hpp
 * @brief Declares the reader and the writer of the vocabulary file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

/** @brief Holds the tokens of a text, the vocabulary and the byte-pair tokenizer. */
namespace SushiAI::Language::Text
{
    /** @brief Holds the four bytes a vocabulary file opens with. */
    inline constexpr std::string_view K_VOCABULARY_FILE_MAGIC = "SAVB";

    /** @brief Holds the format version this build writes, and the newest it reads. */
    inline constexpr std::uint32_t K_VOCABULARY_FILE_VERSION = 1;

    /** @brief Holds the byte count of the header of a vocabulary file. */
    inline constexpr std::size_t K_VOCABULARY_FILE_HEADER_BYTES = 24;

    /**
     * @brief Writes a vocabulary in the current format version, replacing the file.
     * @return Success; InvalidFile, opening with the path, when the file cannot be opened or
     *         written; OutOfMemory.
     * @see modules/language/text/README.md, section "The vocabulary file".
     */
    [[nodiscard]] Status write_vocabulary_file(const std::filesystem::path& file,
                                               const Vocabulary& vocabulary) noexcept;

    /**
     * @brief Returns the vocabulary a serialised vocabulary file holds.
     * @param bytes The whole file.
     * @return The vocabulary; InvalidFile, naming the byte offset, for bytes that end early,
     *         go on past their declared end, hold a field out of bounds or state a format
     *         version this build does not read; InvalidDescription for a merge
     *         Vocabulary::make refuses.
     */
    [[nodiscard]] Result<Vocabulary> parse_vocabulary_file(std::string_view bytes) noexcept;

    /**
     * @brief Reads a vocabulary file of any format version this build knows.
     * @return What parse_vocabulary_file returns, opening with the path; FileNotFound when
     *         the path names nothing; InvalidFile when it cannot be read.
     */
    [[nodiscard]] Result<Vocabulary> read_vocabulary_file(const std::filesystem::path& file) noexcept;
} // namespace SushiAI::Language::Text

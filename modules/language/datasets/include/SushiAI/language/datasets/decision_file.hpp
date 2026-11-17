/**************************************************************************/
/* decision_file.hpp                                                      */
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
 * @file decision_file.hpp
 * @brief Declares the decision file: the records of a decision set, written once and read whole.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <ostream>
#include <string_view>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Datasets
{
    /** @brief Holds the four bytes a decision file opens with. */
    inline constexpr std::string_view K_DECISION_FILE_MAGIC = "SADC";

    /** @brief Holds the format version this build writes, and the newest it reads. */
    inline constexpr std::uint32_t K_DECISION_FILE_VERSION = 1;

    /** @brief Holds the byte count of the header of a decision file. */
    inline constexpr std::uint64_t K_DECISION_FILE_HEADER_BYTES = 20;

    /** @brief Holds the most ids one text or one option of a record has. */
    inline constexpr std::uint32_t K_MAX_DECISION_LIST_TOKENS = 1u << 20;

    /** @brief Holds the most options a choice record has. */
    inline constexpr std::uint32_t K_MAX_DECISION_OPTIONS = 1u << 16;

    /** @brief Names the kind of answer a decision record asks for. */
    enum class DecisionType : std::uint8_t
    {
        Choice, /**< One option out of a list; the answer is its index. */
        YesNo,  /**< A yes or a no; the answer is a bool. */
        Number  /**< A real number; the answer is a float. */
    };

    /** @brief Holds one question: its text, its options for a choice, and its answer. */
    struct DecisionRecord
    {
        /** @brief Names the kind of answer; the only answer field read is the one it names. */
        DecisionType type = DecisionType::Choice;

        /** @brief Holds the ids of the question's text; never empty. */
        std::vector<Text::TokenID> text;

        /** @brief Holds the ids of each option of a choice; empty for the other types. */
        std::vector<std::vector<Text::TokenID>> options;

        /** @brief Holds the index of the right option of a choice; 0 for the other types. */
        std::int32_t choice = 0;

        /** @brief Holds the answer of a yes-or-no; false for the other types. */
        bool yes = false;

        /** @brief Holds the answer of a number, finite; 0 for the other types. */
        float number = 0.0f;
    };

    /** @brief Returns whether two records agree in every field. */
    [[nodiscard]] bool operator==(const DecisionRecord& left, const DecisionRecord& right) noexcept;

    /** @brief Returns whether two records differ in a field. */
    [[nodiscard]] bool operator!=(const DecisionRecord& left, const DecisionRecord& right) noexcept;

    /** @brief Holds what a decision file states: its vocabulary and every record. */
    struct DecisionFile
    {
        /** @brief Counts the tokens an id may name; every id is in [0, vocabulary_size). */
        std::int32_t vocabulary_size = 0;

        /** @brief Holds the records in the order of the file. */
        std::vector<DecisionRecord> records;
    };

    /**
     * @brief Writes records in the current format version, replacing the file.
     * @param vocabulary_size In [1, Text::K_MAX_VOCABULARY_SIZE].
     * @return Success; InvalidDescription, naming the record, for a record that breaks the rules
     *         of its type or an id outside [0, vocabulary_size), before the file is opened;
     *         InvalidFile, opening with the path, when the file cannot be opened or written.
     * @see modules/language/datasets/README.md, section "The decision file".
     */
    [[nodiscard]] Status write_decision_file(const std::filesystem::path& file,
                                             SushiRuntime::span<const DecisionRecord> records,
                                             std::int32_t vocabulary_size) noexcept;

    /**
     * @brief Writes records in the current format version to a binary stream.
     * @param out A binary stream; nothing is written to it when an argument is refused.
     * @return As for the path overload; InvalidFile, naming the byte count, when the stream fails.
     */
    [[nodiscard]] Status write_decision_file(std::ostream& out,
                                             SushiRuntime::span<const DecisionRecord> records,
                                             std::int32_t vocabulary_size) noexcept;

    /**
     * @brief Reads and checks a whole decision file.
     * @return The vocabulary and the records; FileNotFound; InvalidFile, opening with the path
     *         and naming the byte offset, for a file that ends inside a field, goes on past its
     *         last record, or holds a field or a record out of its bounds.
     */
    [[nodiscard]] Result<DecisionFile> read_decision_file(
        const std::filesystem::path& file) noexcept;

    /**
     * @brief Reads and checks a whole decision file from the start of its stream.
     * @param in A binary stream over the whole file, positioned at its first byte.
     * @param file_bytes The byte count of the whole file.
     * @return What the path overload returns, without the path in a message.
     */
    [[nodiscard]] Result<DecisionFile> read_decision_file(std::istream& in,
                                                          std::uint64_t file_bytes) noexcept;
} // namespace SushiAI::Language::Datasets

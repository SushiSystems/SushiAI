/**************************************************************************/
/* bpe_trainer.hpp                                                        */
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
 * @file bpe_trainer.hpp
 * @brief Declares the trainer that learns the merges of a byte-pair vocabulary from a corpus.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

/** @brief Holds the tokens of a text, the vocabulary and the byte-pair tokenizer. */
namespace SushiAI::Language::Text
{
    /** @brief Holds the fewest times a pair occurs in the corpus for it to become a token. */
    inline constexpr std::uint64_t K_MIN_MERGE_COUNT = 2;

    /** @brief Holds the most bytes of a corpus the trainer reads. */
    inline constexpr std::uint64_t K_MAX_CORPUS_BYTES = 0xFFFFFFFFull;

    /** @brief Holds what the caller chooses about a training run. */
    struct BPETrainerOptions
    {
        /** @brief Holds the token count to reach, special tokens included. */
        std::int32_t vocabulary_size = 0;

        /** @brief Holds the special tokens after end-of-text, as Vocabulary::make takes them. */
        std::vector<std::string> special_names;
    };

    /**
     * @brief Merges the most frequent pair of a corpus until the vocabulary has the size asked.
     * @param corpus Read as bytes; of two pairs equally frequent, the smaller ids merge first.
     * @return The vocabulary, smaller than asked when no pair occurs K_MIN_MERGE_COUNT times;
     *         InvalidDescription, Unsupported or OutOfMemory, as the README section states.
     * @see modules/language/text/README.md, section "The trainer".
     */
    [[nodiscard]] Result<Vocabulary> train_bpe(std::string_view corpus,
                                               const BPETrainerOptions& options) noexcept;
} // namespace SushiAI::Language::Text

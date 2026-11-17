/**************************************************************************/
/* decision_set.hpp                                                       */
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
 * @file decision_set.hpp
 * @brief Declares the synthetic decision set: a seeded generator whose records each follow one
 *        learnable rule of their type.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::DecisionModels
{
    /** @brief Counts the token ids of the set; id 0 pads and no text holds it. */
    inline constexpr std::int32_t K_DECISION_SET_VOCABULARY = 32;

    /** @brief Holds the id of the first key of a choice question; the keys are 1 to 8. */
    inline constexpr Text::TokenID K_CHOICE_KEY_FIRST = 1;

    /** @brief Counts the keys of a choice question, and the option ids that answer them. */
    inline constexpr std::int32_t K_CHOICE_KEYS = 8;

    /** @brief Holds the id of the option that answers key 1; the options are 9 to 16. */
    inline constexpr Text::TokenID K_CHOICE_OPTION_FIRST = 9;

    /** @brief Holds the id of the first token of a yes-or-no question; they are 17 to 24. */
    inline constexpr Text::TokenID K_YES_NO_FIRST = 17;

    /** @brief Counts the tokens of a yes-or-no question; the first half are positive. */
    inline constexpr std::int32_t K_YES_NO_TOKENS = 8;

    /** @brief Holds the id of the first token of a number question; they are 25 to 31. */
    inline constexpr Text::TokenID K_NUMBER_FIRST = 25;

    /** @brief Counts the tokens of a number question. */
    inline constexpr std::int32_t K_NUMBER_TOKENS = 7;

    /** @brief Counts the tokens of a yes-or-no question and of a number question. */
    inline constexpr std::int32_t K_SCALAR_TEXT_TOKENS = 3;

    /** @brief Counts the options the smallest choice record holds. */
    inline constexpr std::int32_t K_MIN_CHOICE_OPTIONS = 2;

    /** @brief Holds what the caller chooses about a synthetic set. */
    struct DecisionSetOptions
    {
        /** @brief Seeds the draws; the same seed gives the same records. */
        std::uint64_t seed = 1;

        /** @brief Counts the records; a third of each type, the extra ones going to the first. */
        std::int64_t records = 0;

        /** @brief Counts K, the most options a choice holds, in [K_MIN_CHOICE_OPTIONS, 8]. */
        std::int32_t options = 4;
    };

    /**
     * @brief Returns the id of the option that answers the choice question with key @p key.
     * @param key In [K_CHOICE_KEY_FIRST, K_CHOICE_KEY_FIRST + K_CHOICE_KEYS).
     */
    [[nodiscard]] constexpr Text::TokenID right_option_of(Text::TokenID key) noexcept
    {
        return key - K_CHOICE_KEY_FIRST + K_CHOICE_OPTION_FIRST;
    }

    /**
     * @brief Returns whether a yes-or-no question answers yes: two of its tokens or more are
     *        positive, which is the first half of the ids of that type.
     * @param text The ids of the question.
     */
    [[nodiscard]] bool yes_no_answer_of(SushiRuntime::span<const Text::TokenID> text) noexcept;

    /**
     * @brief Returns the answer of a number question: the mean of (id - 28) / 3 over its ids,
     *        a value in [-1, 1].
     * @param text The ids of the question.
     */
    [[nodiscard]] float number_answer_of(SushiRuntime::span<const Text::TokenID> text) noexcept;

    /**
     * @brief Draws a decision set from the rules of the three types, in a seeded order.
     * @return The records, or InvalidDescription for a negative count or options out of range.
     * @see modules/language/decision_models/README.md, section "The synthetic set".
     */
    [[nodiscard]] Result<std::vector<Datasets::DecisionRecord>> generate_decision_set(
        const DecisionSetOptions& options) noexcept;

    /**
     * @brief Writes the set generate_decision_set() draws as a decision file.
     * @return Success, or what generate_decision_set() and write_decision_file() return.
     */
    [[nodiscard]] Status write_decision_set(const std::filesystem::path& file,
                                            const DecisionSetOptions& options) noexcept;
} // namespace SushiAI::Language::DecisionModels

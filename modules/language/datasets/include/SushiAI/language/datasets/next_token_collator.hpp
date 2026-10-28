/**************************************************************************/
/* next_token_collator.hpp                                                */
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
 * @file next_token_collator.hpp
 * @brief Declares the collator of the next-token task.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/datasets/collator.hpp>

/** @brief Holds the token file of a corpus and the dataset that reads it in windows. */
namespace SushiAI::Language::Datasets
{
    /** @brief Names the next-token task in a training file. */
    inline constexpr std::string_view K_TASK_NEXT_TOKEN = "next_token";

    /**
     * @brief Writes T ids of a window as the input and the T ids after them as labels.
     *
     * A window is T + 1 ids w. Row n of the input holds w[0..T) and row n of the role labels,
     * int32 of [N, T], holds w[1..T], so labels[n][t] is the id that follows inputs[n][t].
     */
    class NextTokenCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator of windows of @p window_length positions.
             * @return The collator, never null; InvalidDescription for a length outside
             *         [1, K_MAX_WINDOW_TOKENS - 1] or a vocabulary size outside
             *         [1, Text::K_MAX_VOCABULARY_SIZE].
             */
            [[nodiscard]] static Result<std::unique_ptr<NextTokenCollator>> make(
                std::int64_t window_length, std::int32_t vocabulary_size) noexcept;

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] std::int64_t window_length() const noexcept override;

            /** @brief Returns window_length() + 1: the input ids and the label of the last. */
            [[nodiscard]] std::int64_t tokens_per_window() const noexcept override;

            [[nodiscard]] std::int32_t vocabulary_size() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;
            [[nodiscard]] Status collate(SushiRuntime::span<const Text::TokenID> tokens,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the description of labels over @p vocabulary_size classes. */
            NextTokenCollator(std::int64_t window_length, std::int32_t vocabulary_size);

            /// @brief Counts the positions of one input row.
            std::int64_t window_length_;

            /// @brief Counts the tokens an id may name.
            std::int32_t vocabulary_size_;

            /// @brief Holds the description of the labels role.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Language::Datasets

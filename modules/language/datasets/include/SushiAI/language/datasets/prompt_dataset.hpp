/**************************************************************************/
/* prompt_dataset.hpp                                                     */
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
 * @file prompt_dataset.hpp
 * @brief Declares PromptDataset, the one batch that holds the ids a generation starts from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Datasets
{
    /** @brief Names the type of the data block make_prompt_dataset() reads. */
    inline constexpr std::string_view K_PROMPT_DATA_TYPE = "prompt";

    /** @brief Names the role that holds how many leading ids of the row are the prompt. */
    inline constexpr std::string_view K_ROLE_PROMPT_LENGTH = "prompt_length";

    /** @brief Holds the row of a prompt: its ids, then zeros, and the length of the prompt. */
    struct PromptRow
    {
        /** @brief Holds exactly the window's ids: the kept ids first, zeros behind them. */
        std::vector<Text::TokenID> ids;

        /** @brief Counts the leading ids of the row that are the prompt. */
        std::int32_t length = 0;
    };

    /**
     * @brief Lays out the row of a prompt: the last @p window ids of @p ids, or all of them
     *        when fewer, then zeros up to the window.
     * @return The row, or InvalidDescription for no id, a negative id or a window outside 1 to
     *         K_MAX_WINDOW_TOKENS.
     */
    [[nodiscard]] Result<PromptRow> lay_out_prompt(const std::vector<Text::TokenID>& ids,
                                                   std::int64_t window) noexcept;

    /** @brief Holds the keys of a "prompt" data block. */
    struct PromptSection
    {
        /** @brief Holds the text a generation starts from. */
        std::string text;

        /** @brief Holds the path of the vocabulary file the text is encoded with. */
        std::string vocabulary;

        /** @brief Counts the positions of the row. */
        std::int64_t window = 0;
    };

    /**
     * @brief Reads "text", "vocabulary" and "window" of a "prompt" data block.
     * @param context Names the block in a message.
     * @return The keys, or InvalidDescription for a block that is no object, a key it does
     *         not accept, or a key that is absent or of another kind.
     */
    [[nodiscard]] Result<PromptSection> read_prompt_section(const Config::Value& section,
                                                            std::string_view context) noexcept;

    /** @brief Publishes one batch of one row: the last ids of a prompt, then zeros. */
    class PromptDataset final : public Data::Dataset
    {
        public:
            /**
             * @brief Keeps the last @p window ids of @p ids, or all of them when fewer.
             * @param engine Allocates the tensors of the batch; it must outlive the dataset.
             * @return The dataset, or the refusal of lay_out_prompt().
             */
            [[nodiscard]] static Result<std::unique_ptr<PromptDataset>> make(
                SushiBLAS::Engine& engine, const std::vector<Text::TokenID>& ids,
                std::int64_t window) noexcept;

            PromptDataset(const PromptDataset&) = delete;
            PromptDataset& operator=(const PromptDataset&) = delete;
            PromptDataset(PromptDataset&&) = delete;
            PromptDataset& operator=(PromptDataset&&) = delete;
            ~PromptDataset() override = default;

            /** @brief Returns one, the rows of the batch. */
            [[nodiscard]] std::int64_t batch_size() const noexcept override;

            /** @brief Returns [window]. */
            [[nodiscard]] Shape sample_shape() const override;

            /** @brief Returns Int32, the element type of a token id. */
            [[nodiscard]] DType sample_dtype() const noexcept override;

            /** @brief Returns the one role, K_ROLE_PROMPT_LENGTH, an int32 of [1]. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /** @brief Returns one, the batches of an epoch. */
            [[nodiscard]] std::size_t batch_count() const noexcept override;

            /** @brief Returns "prompt". */
            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Writes the kept ids to the front of the row, zeros behind them, and
             *        their count under K_ROLE_PROMPT_LENGTH.
             * @pre index is 0; any other stops the process.
             */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override;

            /**
             * @brief Does what read() does; the one batch holds one row, so @p rows names it.
             * @return As for read().
             * @pre What Data::Dataset::read_rows() states.
             */
            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override;

        private:
            /** @brief Takes the parts make() checked. */
            PromptDataset(SushiBLAS::Engine& engine, PromptRow row, std::int64_t window);

            /// @brief Allocates the tensors of the batch; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Holds the row the batch publishes.
            PromptRow row_;

            /// @brief Counts the positions of the row.
            std::int64_t window_;

            /// @brief Holds the description of the length role.
            std::vector<Data::TargetDescription> targets_;
    };

    /**
     * @brief Builds a PromptDataset from "text", "vocabulary" and "window" of a data block.
     * @param section "vocabulary" names the vocabulary file "text" is encoded with.
     * @return The dataset, or InvalidDescription for a key that does not fit, or the failure
     *         of the vocabulary file, the encoder or PromptDataset::make.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_prompt_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context) noexcept;

} // namespace SushiAI::Language::Datasets

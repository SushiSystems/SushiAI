/**************************************************************************/
/* decision_dataset.hpp                                                   */
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
 * @file decision_dataset.hpp
 * @brief Declares the dataset that reads batches of decision records, and its data block.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>

namespace SushiAI::Language::Datasets
{
    /** @brief Names the type of the data block make_decisions_dataset() reads. */
    inline constexpr std::string_view K_DECISIONS_DATA_TYPE = "decisions";

    /**
     * @brief Names the role that holds how many ids of each row of the inputs are the text, as
     *        one float32 per record.
     */
    inline constexpr std::string_view K_ROLE_LENGTHS = "lengths";

    /** @brief Names the role that holds the ids of the options of a choice, K rows of T ids. */
    inline constexpr std::string_view K_ROLE_OPTIONS = "options";

    /** @brief Names the role that holds how many ids of each option row are the option, as float32. */
    inline constexpr std::string_view K_ROLE_OPTION_LENGTHS = "option_lengths";

    /** @brief Names the role that holds 1 for each option a record has and 0 for a padded one. */
    inline constexpr std::string_view K_ROLE_OPTION_VALID = "option_valid";

    /** @brief Names the role that holds the index of the right option of a choice, as int32. */
    inline constexpr std::string_view K_ROLE_CHOICE = "choice";

    /** @brief Names the role that holds 1 for a yes and 0 for a no, as one float32 per record. */
    inline constexpr std::string_view K_ROLE_YES_NO = "yes_no";

    /** @brief Names the role that holds the value of a number question, as one float32 per record. */
    inline constexpr std::string_view K_ROLE_NUMBER = "number";

    /** @brief Names the role that holds one 1 per row, in the column of its type. */
    inline constexpr std::string_view K_ROLE_TYPE_MASK = "type_mask";

    /** @brief Counts the columns of the type mask: Choice, YesNo and Number, in that order. */
    inline constexpr std::int64_t K_DECISION_TYPES = 3;

    /** @brief Holds the extents every row of a batch of decisions is padded to. */
    struct DecisionShape
    {
        /** @brief Counts T, the ids of a text and of an option, in [1, K_MAX_WINDOW_TOKENS]. */
        std::int64_t window = 0;

        /** @brief Counts K, the options of a choice, in [1, K_MAX_DECISION_OPTIONS]. */
        std::int64_t options = 0;
    };

    /** @brief Holds the values of the rows of one batch, each role as a flat host array. */
    struct DecisionRows
    {
        /** @brief Holds [N, T] ids: each text, then zeros. */
        std::vector<std::int32_t> tokens;

        /** @brief Holds [N, 1] counts of the ids of each text. */
        std::vector<float> lengths;

        /** @brief Holds [N, K, T] ids: each option, then zeros; an absent option is all zeros. */
        std::vector<std::int32_t> options;

        /** @brief Holds [N, K] counts of the ids of each option; 0 for an absent option. */
        std::vector<float> option_lengths;

        /** @brief Holds [N, K] flags: 1 for an option of at least one id, 0 for any other. */
        std::vector<float> option_valid;

        /** @brief Holds [N] indices of the right option of a choice; 0 for any other type. */
        std::vector<std::int32_t> choice;

        /** @brief Holds [N, 1] answers: 1 for a yes, 0 for a no and for every other type. */
        std::vector<float> yes_no;

        /** @brief Holds [N, 1] values of a number question; 0 for any other type. */
        std::vector<float> number;

        /** @brief Holds [N, 3] one-hot rows over Choice, YesNo and Number. */
        std::vector<float> type_mask;
    };

    /**
     * @brief Fails unless a record fits the extents a batch pads to.
     * @param index The position of the record in its file, named in the refusal.
     * @return Success; InvalidDescription for a text or an option longer than T, or a choice
     *         with more than K options.
     */
    [[nodiscard]] Status check_decision_fits(const DecisionRecord& record,
                                             const DecisionShape& shape,
                                             std::size_t index) noexcept;

    /**
     * @brief Lays out the rows of a batch from records, with no engine.
     * @return The rows; InvalidDescription for extents outside their bounds or a record that
     *         does not fit, or OutOfMemory.
     * @see modules/language/datasets/README.md, section "DecisionDataset".
     */
    [[nodiscard]] Result<DecisionRows> lay_out_decisions(
        SushiRuntime::span<const DecisionRecord> records, const DecisionShape& shape) noexcept;

    /** @brief Holds what the caller chooses about a decision dataset. */
    struct DecisionDatasetOptions
    {
        /** @brief Counts the records of a batch. */
        std::int64_t batch_size = 16;

        /** @brief Holds T and K. */
        DecisionShape shape;

        /** @brief Holds the name a run report prints. */
        std::string name = "decisions";
    };

    /**
     * @brief Builds batch i from records [i * batch size, (i + 1) * batch size) of a file.
     *
     * The file is read whole when the dataset is made. The input is the [N, T] int32 texts and
     * the eight roles follow targets(). read() keeps no state, so threads may call it at once.
     * @pre The engine outlives the dataset.
     * @see modules/language/datasets/README.md, section "DecisionDataset".
     */
    class DecisionDataset final : public Data::Dataset
    {
        public:
            /**
             * @brief Reads a decision file and checks every record against the options.
             * @return The dataset, never null; InvalidDescription for a batch size below 1,
             *         extents out of bounds, a record that does not fit them, or a file of
             *         fewer records than one batch; what read_decision_file returns.
             */
            [[nodiscard]] static Result<std::unique_ptr<DecisionDataset>> make(
                SushiBLAS::Engine& engine, const std::filesystem::path& file,
                DecisionDatasetOptions options) noexcept;

            DecisionDataset(const DecisionDataset&) = delete;
            DecisionDataset& operator=(const DecisionDataset&) = delete;
            DecisionDataset(DecisionDataset&&) = delete;
            DecisionDataset& operator=(DecisionDataset&&) = delete;
            ~DecisionDataset() override = default;

            /** @brief Returns the records per batch. */
            [[nodiscard]] std::int64_t batch_size() const noexcept override;

            /** @brief Returns [T], the shape of one text row. */
            [[nodiscard]] Shape sample_shape() const override;

            /** @brief Returns Int32, the element type of the ids a text row holds. */
            [[nodiscard]] DType sample_dtype() const noexcept override;

            /** @brief Returns the eight roles every batch carries. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /** @brief Returns how many whole batches the records of the file make. */
            [[nodiscard]] std::size_t batch_count() const noexcept override;

            /** @brief Returns the name a run report prints. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /** @brief Returns the token count the file states; every id is below it. */
            [[nodiscard]] std::int32_t vocabulary_size() const noexcept;

            /**
             * @brief Lays out the records of one batch into @p batch.
             * @return Success, or the failure of an allocation.
             * @pre @p index is below batch_count(); one that is not stops the process.
             */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override;

            /**
             * @brief Lays out rows @p rows of one batch, and no other.
             * @return As for read().
             * @pre What Data::Dataset::read_rows() states.
             */
            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override;

        private:
            /** @brief Takes the parts make() checked. */
            DecisionDataset(SushiBLAS::Engine& engine, DecisionFile file,
                            DecisionDatasetOptions options);

            /// @brief Allocates the tensors of each batch; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Holds the vocabulary and the records of the file.
            DecisionFile file_;

            /// @brief Holds the batch size, the extents and the name.
            DecisionDatasetOptions options_;

            /// @brief Holds the description of each role of a batch.
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Holds the keys of a "decisions" data block. */
    struct DecisionsSection
    {
        /** @brief Holds the path of the decision file. */
        std::string file;

        /** @brief Holds the batch size, T and K the block states. */
        DecisionDatasetOptions options;
    };

    /**
     * @brief Reads "file", "window", "options" and "batch_size" of a "decisions" data block.
     * @param context Names the block in a message.
     * @return The keys, or InvalidDescription for a block that is no object, a key it does
     *         not accept, or a key that is absent, of another kind or not positive.
     */
    [[nodiscard]] Result<DecisionsSection> read_decisions_section(
        const Config::Value& section, std::string_view context) noexcept;

    /**
     * @brief Builds a DecisionDataset from the keys of a data block.
     * @param section "file", "window" (T) and "options" (K) are required; "batch_size" is that
     *        of DecisionDatasetOptions when absent.
     * @return The dataset, or InvalidDescription for a key that does not fit, or the failure of
     *         DecisionDataset::make.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_decisions_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context) noexcept;
} // namespace SushiAI::Language::Datasets

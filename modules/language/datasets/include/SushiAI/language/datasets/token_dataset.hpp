/**************************************************************************/
/* token_dataset.hpp                                                      */
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
 * @file token_dataset.hpp
 * @brief Declares the dataset that reads each batch of windows from a token file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/datasets/collator.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>

/** @brief Holds the token file of a corpus and the dataset that reads it in windows. */
namespace SushiAI::Language::Datasets
{
    /** @brief Holds what the caller chooses about a token dataset. */
    struct TokenDatasetOptions
    {
        /** @brief Counts the windows of a batch. */
        std::int64_t batch_size = 16;

        /** @brief Counts the ids between two window starts; the window length when absent. */
        std::optional<std::int64_t> stride;

        /** @brief Holds the name a run report prints. */
        std::string name = "tokens";
    };

    /**
     * @brief Builds batch i from windows [i * batch size, (i + 1) * batch size) of a token file.
     *
     * Window w begins at id w * stride; those past the last whole batch are not read. read()
     * keeps no state, so threads may call it at once, each with its own Batch.
     * @pre The engine outlives the dataset.
     * @see modules/language/datasets/README.md, section "Windows".
     */
    class TokenDataset final : public Data::Dataset
    {
        public:
            /**
             * @brief Reads the header of a token file and checks the options against it.
             * @return The dataset, never null; InvalidDescription for a null collator, a
             *         batch size or a stride below 1, or a file of fewer windows than one
             *         batch; Incompatible for a file whose vocabulary is above the
             *         collator's; what read_token_file_header returns.
             */
            [[nodiscard]] static Result<std::unique_ptr<TokenDataset>> make(
                SushiBLAS::Engine& engine, std::filesystem::path file,
                std::unique_ptr<const Collator> collator, TokenDatasetOptions options);

            TokenDataset(const TokenDataset&) = delete;
            TokenDataset& operator=(const TokenDataset&) = delete;
            TokenDataset(TokenDataset&&) = delete;
            TokenDataset& operator=(TokenDataset&&) = delete;
            ~TokenDataset() override = default;

            /** @brief Returns the windows per batch. */
            [[nodiscard]] std::int64_t batch_size() const noexcept override;

            /** @brief Returns [T], the shape of one input row. */
            [[nodiscard]] Shape sample_shape() const override;

            /** @brief Returns Int32, the element type of the ids an input row holds. */
            [[nodiscard]] DType sample_dtype() const noexcept override;

            /** @brief Returns the roles every batch carries, as the collator states them. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /** @brief Returns how many whole batches the windows of the file make. */
            [[nodiscard]] std::size_t batch_count() const noexcept override;

            /** @brief Returns the name a run report prints. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /** @brief Returns the token count the file states; every id read is below it. */
            [[nodiscard]] std::int32_t vocabulary_size() const noexcept;

            /**
             * @brief Reads the windows of one batch and collates them into @p batch.
             * @param batch Written in place when it holds this dataset's layout and nothing
             *              else references its tensors; allocated anew otherwise.
             * @return Success; InvalidFile, naming the file and the byte offset, for an id
             *         outside the vocabulary or a file that ends early; an allocation failure.
             * @pre @p index is below batch_count(); one that is not stops the process.
             */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override;

        private:
            /** @brief Takes the parts make() checked. */
            TokenDataset(SushiBLAS::Engine& engine, std::filesystem::path file,
                         TokenFileHeader header, std::unique_ptr<const Collator> collator,
                         WindowLayout layout, std::uint64_t windows,
                         TokenDatasetOptions options) noexcept;

            /// @brief Allocates the tensors of each batch; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Holds the path of the token file.
            std::filesystem::path file_;

            /// @brief Holds what the file's header stated when make() read it.
            TokenFileHeader header_;

            /// @brief Writes the roles of the task; never null.
            std::unique_ptr<const Collator> collator_;

            /// @brief Holds the ids a window reads and the stride between two windows.
            WindowLayout layout_;

            /// @brief Counts the whole windows of the file.
            std::uint64_t windows_;

            /// @brief Holds the batch size and the name.
            TokenDatasetOptions options_;
    };
} // namespace SushiAI::Language::Datasets

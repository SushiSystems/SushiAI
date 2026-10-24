/**************************************************************************/
/* table_dataset.hpp                                                      */
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
 * @file table_dataset.hpp
 * @brief Declares the dataset that collates each batch of a tabular sample list.
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

#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what the caller chooses about a table dataset. */
    struct TableDatasetOptions
    {
        /** @brief Counts the samples of a batch. */
        std::int64_t batch_size = 16;

        /** @brief Holds the name a run report prints. */
        std::string name = "table";
    };

    /**
     * @brief Builds batch i from rows [i * batch size, (i + 1) * batch size) of a table.
     *
     * The rows keep the order of the list, and those past the last whole batch are not
     * read. No file is opened and no image is fitted: a sample is a view of its row.
     * @pre The engine outlives the dataset.
     */
    class TableDataset final : public Data::Dataset
    {
        public:
            /**
             * @brief Checks the options against the list and takes the list and the collator.
             * @return The dataset, never null; InvalidDescription for a null collator, a
             *         table with no feature or with another row count than the list has
             *         samples, a batch size below 1, or fewer samples than one batch.
             */
            [[nodiscard]] static Result<std::unique_ptr<TableDataset>> make(
                SushiBLAS::Engine& engine, SampleList list,
                std::unique_ptr<const Collator> collator, TableDatasetOptions options);

            TableDataset(const TableDataset&) = delete;
            TableDataset& operator=(const TableDataset&) = delete;
            TableDataset(TableDataset&&) = delete;
            TableDataset& operator=(TableDataset&&) = delete;
            ~TableDataset() override = default;

            /** @brief Returns the samples per batch. */
            [[nodiscard]] std::int64_t batch_size() const noexcept override;

            /** @brief Returns [features], the shape of one input sample. */
            [[nodiscard]] Shape sample_shape() const override;

            /** @brief Returns the roles every batch carries, as the collator states them. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /** @brief Returns how many whole batches the list holds. */
            [[nodiscard]] std::size_t batch_count() const noexcept override;

            /** @brief Returns the name a run report prints. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Collates the rows of one batch into @p batch.
             * @param batch Written in place when it holds this dataset's layout and nothing
             *              else references its tensors; allocated anew otherwise.
             * @return Success, or the failure of the allocation or of the collator.
             * @pre @p index is below batch_count(); one that is not stops the process.
             */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override;

            /**
             * @brief Collates rows @p rows of one batch into @p batch, and no other row.
             * @return As for read().
             * @pre What Data::Dataset::read_rows() states.
             */
            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override;

        private:
            /** @brief Takes the parts make() checked. */
            TableDataset(SushiBLAS::Engine& engine, SampleList list,
                         std::unique_ptr<const Collator> collator,
                         TableDatasetOptions options) noexcept;

            /// @brief Allocates the tensors of each batch; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Holds the samples and the table their rows are in.
            SampleList list_;

            /// @brief Writes the inputs and the roles of the task; never null.
            std::unique_ptr<const Collator> collator_;

            /// @brief Holds the batch size and the name.
            TableDatasetOptions options_;
    };
} // namespace SushiAI::Vision::Datasets

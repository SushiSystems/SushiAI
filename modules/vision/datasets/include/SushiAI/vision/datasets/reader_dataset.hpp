/**************************************************************************/
/* reader_dataset.hpp                                                     */
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
 * @file reader_dataset.hpp
 * @brief Declares the dataset that builds each batch of a sample list when it is asked for.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
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
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what the caller chooses about a reader dataset. */
    struct ReaderDatasetOptions
    {
        /** @brief Holds the input size and how an image is fitted to it. */
        SampleFitOptions fit{};

        /** @brief Counts the channels of the input, in [1, Imaging::K_MAX_CHANNELS]. */
        std::int32_t channels = Imaging::K_RGB_CHANNELS;

        /** @brief Counts the samples of a batch. */
        std::int64_t batch_size = 16;

        /** @brief Holds the name a run report prints. */
        std::string name = "reader";

        /** @brief Selects a new order of the samples in every epoch; false keeps the list's. */
        bool shuffle = false;

        /** @brief Holds the seed of the order of every epoch; read when shuffle is set. */
        std::uint64_t seed = 0;
    };

    /**
     * @brief Builds batch i from positions [i * batch size, (i + 1) * batch size) of an epoch.
     *
     * A position is a sample of the list: the same one, or under shuffle the one
     * Data::EpochPermutation gives the epoch. Positions past the last whole batch are not
     * read. read() changes nothing, so threads may call it at once, each with its own Batch.
     * @pre The engine outlives the dataset.
     */
    class ReaderDataset final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /**
             * @brief Checks the options against the list and takes the list and the collator.
             *
             * The samples go through a FitPipeline of options.fit and options.channels.
             * @return What the form that takes a pipeline returns.
             */
            [[nodiscard]] static Result<std::unique_ptr<ReaderDataset>> make(
                SushiBLAS::Engine& engine, SampleList list,
                std::unique_ptr<const Collator> collator, ReaderDatasetOptions options) noexcept;

            /**
             * @brief Checks the options against the list and takes the three parts.
             * @param pipeline Makes samples of options.fit.size and options.channels.
             * @return The dataset, never null; InvalidDescription for a null collator or
             *         pipeline, a size, a channel count or a batch size outside its bounds,
             *         or a list that holds fewer samples than one batch.
             */
            [[nodiscard]] static Result<std::unique_ptr<ReaderDataset>> make(
                SushiBLAS::Engine& engine, SampleList list,
                std::unique_ptr<const Collator> collator,
                std::unique_ptr<const SamplePipeline> pipeline, ReaderDatasetOptions options) noexcept;

            ReaderDataset(const ReaderDataset&) = delete;
            ReaderDataset& operator=(const ReaderDataset&) = delete;
            ReaderDataset(ReaderDataset&&) = delete;
            ReaderDataset& operator=(ReaderDataset&&) = delete;
            ~ReaderDataset() override = default;

            /** @brief Returns the samples per batch. */
            [[nodiscard]] std::int64_t batch_size() const noexcept override;

            /** @brief Returns [height, width, channels], the shape of one input sample. */
            [[nodiscard]] Shape sample_shape() const override;

            /** @brief Returns the roles every batch carries, as the collator states them. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /** @brief Returns how many whole batches the list holds. */
            [[nodiscard]] std::size_t batch_count() const noexcept override;

            /** @brief Returns the name a run report prints. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Makes the samples of one batch of the epoch and collates them.
             * @param batch Written in place when it holds this dataset's layout and nothing
             *              else references its tensors; allocated anew otherwise.
             * @return Success; Incompatible, opening with the file, for another channel count;
             *         or the failure of the pipeline, the allocation or the collator.
             * @pre @p index is below batch_count(); one that is not stops the process.
             */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override;

            /**
             * @brief Makes and collates the samples of rows @p rows of one batch, and no other.
             * @return As for read().
             * @pre What Data::Dataset::read_rows() states.
             */
            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override;

            /**
             * @brief Sets the epoch whose order and whose draws the next reads take.
             * @pre What Data::EpochAware::begin_epoch() states; the epoch is 0 until a call.
             */
            void begin_epoch(std::int64_t epoch) const noexcept override;

        private:
            /** @brief Takes the parts make() checked. */
            ReaderDataset(SushiBLAS::Engine& engine, SampleList list,
                          std::unique_ptr<const Collator> collator,
                          std::unique_ptr<const SamplePipeline> pipeline,
                          ReaderDatasetOptions options) noexcept;

            /// @brief Allocates the tensors of each batch; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Holds the samples and the files their pixels are in.
            SampleList list_;

            /// @brief Writes the roles of the task; never null.
            std::unique_ptr<const Collator> collator_;

            /// @brief Makes the fitted samples of a batch; never null.
            std::unique_ptr<const SamplePipeline> pipeline_;

            /// @brief Holds the input size, the channels, the batch size, the order and the name.
            ReaderDatasetOptions options_;

            /// @brief Holds the epoch begin_epoch() set; a read takes it once, at its start.
            mutable std::atomic<std::int64_t> epoch_{0};
    };
} // namespace SushiAI::Vision::Datasets

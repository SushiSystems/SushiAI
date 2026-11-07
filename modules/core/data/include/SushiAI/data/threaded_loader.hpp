/**************************************************************************/
/* threaded_loader.hpp                                                    */
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
 * @file threaded_loader.hpp
 * @brief Declares the dataset that reads the batches of another one ahead, on threads.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Data
    {
        /** @brief Names where a loader reads the batches of its dataset. */
        enum class ReadExecutor : std::uint8_t
        {
            OwnThreads,   /**< On threads the loader starts; the default. */
            CallingThread /**< On the thread that asked, at the moment a batch is queued. */
        };

        /**
         * @struct LoaderOptions
         * @brief Holds what the caller chooses about a threaded loader.
         */
        struct LoaderOptions
        {
            /** @brief Counts the threads that read; at least 1. */
            std::int32_t workers = 2;

            /** @brief Counts the batches read ahead of the caller, and held; at least 1. */
            std::int32_t prefetch = 2;

            /**
             * @brief Names where the reads run.
             *
             * CallingThread starts no thread: a batch is read on the thread that queues it, so
             * the order of the reads is fixed and `workers` is only checked. `prefetch` bounds
             * the reads ahead as before.
             */
            ReadExecutor executor = ReadExecutor::OwnThreads;
        };

        /**
         * @class ThreadedLoader
         * @brief Reads the batches of a dataset ahead on its own threads, in index order.
         *
         * One thread at a time calls read() and begin_epoch(). The destructor waits for the
         * reads that are running and joins the threads.
         * @see modules/core/data/README.md, section "The loader on threads".
         */
        class ThreadedLoader final : public Dataset, public EpochAware
        {
            public:
                /**
                 * @brief Takes a dataset and starts the threads, which wait for the first read.
                 * @param inner Read from the threads, several batches at once; never null.
                 * @return The loader, never null; InvalidDescription for a null dataset or
                 *         an option below 1; the failure of starting a thread.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ThreadedLoader>> make(
                    std::unique_ptr<Dataset> inner, const LoaderOptions& options) noexcept;

                ThreadedLoader(const ThreadedLoader&) = delete;
                ThreadedLoader& operator=(const ThreadedLoader&) = delete;
                ThreadedLoader(ThreadedLoader&&) = delete;
                ThreadedLoader& operator=(ThreadedLoader&&) = delete;
                ~ThreadedLoader() override;

                int64_t batch_size() const noexcept override;
                Shape sample_shape() const override;
                DType sample_dtype() const noexcept override;
                const std::vector<TargetDescription>& targets() const noexcept override;
                std::size_t batch_count() const noexcept override;
                std::string_view name() const noexcept override;

                /**
                 * @brief Hands over batch @p index, waiting for the thread that reads it.
                 * @param batch Exchanged with the batch a thread filled, which is reused.
                 * @return The status the inner dataset gave for @p index; a failure leaves
                 *         every other index readable.
                 * @pre @p index is below batch_count(); one that is not stops the process.
                 */
                [[nodiscard]] Status read(std::size_t index, Batch& batch) const override;

                /**
                 * @brief Reads rows @p rows of batch @p index from the inner dataset, on the
                 *        caller's thread; what was read ahead is kept and not used.
                 * @return What the inner dataset's read_rows() returns.
                 * @pre What Dataset::read_rows() states.
                 */
                [[nodiscard]] Status read_rows(std::size_t index, RowRange rows,
                                               Batch& batch) const override;

                /**
                 * @brief Drops what was read ahead, forwards the epoch and reads on from 0.
                 * @pre What EpochAware::begin_epoch() states.
                 */
                void begin_epoch(std::int64_t epoch) const noexcept override;

            private:
                struct State;

                /** @brief Takes the state make() built. */
                explicit ThreadedLoader(std::unique_ptr<State> state) noexcept;

                /** @brief Holds the dataset, the batches read ahead and the threads. */
                std::unique_ptr<State> state_;
        };

    } // namespace Data
} // namespace SushiAI

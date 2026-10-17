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
                    std::unique_ptr<Dataset> inner, const LoaderOptions& options);

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

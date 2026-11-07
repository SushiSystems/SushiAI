/**************************************************************************/
/* in_memory_dataset.hpp                                                  */
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

/**
 * @file in_memory_dataset.hpp
 * @brief Declares the datasets that serve batches built in memory and journal every read.
 * @author Mustafa Garip
 * @see modules/core/data/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/batch_allocator.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/in_memory_epoch_aware.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @class InMemoryDataset
         * @brief A Dataset whose batches are built in memory, with no device, file or clock;
         *        it journals the batch indices it reads.
         *
         * A batch has the described layout unless the caller gives a BatchMaker. A read of the
         * index set with fail_at() returns InvalidFile. The journal is guarded.
         */
        class InMemoryDataset : public Dataset
        {
            public:
                /** @brief Builds the batch of an epoch and an index. */
                using BatchMaker = std::function<Batch(std::int64_t epoch, std::size_t index)>;

                /**
                 * @brief Describes a dataset of @p batches batches.
                 * @param batches How many batches an epoch holds.
                 * @param batch_size The samples per batch.
                 * @param input The input every batch carries.
                 * @param targets The roles every batch carries.
                 * @param maker Builds a batch; empty for the described layout, with no element.
                 */
                InMemoryDataset(std::size_t batches, int64_t batch_size,
                                const InputDescription& input,
                                std::vector<TargetDescription> targets,
                                BatchMaker maker = BatchMaker())
                    : batches_(batches), batch_size_(batch_size), input_(input),
                      targets_(std::move(targets)), maker_(std::move(maker))
                {

                }

                int64_t batch_size() const noexcept override { return batch_size_; }
                Shape sample_shape() const override { return input_.sample_shape; }
                DType sample_dtype() const noexcept override { return input_.dtype; }
                const std::vector<TargetDescription>& targets() const noexcept override
                {
                    return targets_;
                }
                std::size_t batch_count() const noexcept override { return batches_; }
                std::string_view name() const noexcept override { return "in-memory"; }

                /**
                 * @brief Makes every later read of @p index return InvalidFile.
                 * @param index A batch of the dataset.
                 */
                void fail_at(std::size_t index)
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    failing_ = index;
                }

                /** @brief Appends @p index to the journal and builds the batch of the epoch. */
                [[nodiscard]] Status read(std::size_t index, Batch& batch) const override
                {
                    SA_FATAL_IF(index >= batches_, "Batch {} is out of range; the dataset has {}",
                                index, batches_);

                    {
                        const std::lock_guard<std::mutex> lock(mutex_);
                        journal_.push_back(index);
                        SA_FAIL_IF(index == failing_, StatusCode::InvalidFile,
                                   "batch {} is unreadable", index);
                    }

                    if (maker_)
                    {
                        batch = maker_(current_epoch(), index);
                        return Status();
                    }

                    InMemoryBatchAllocator allocator;
                    SA_TRY_ASSIGN(batch, allocate_batch(allocator, batch_size_, input_, targets_));
                    return Status();
                }

                /** @brief Appends @p index to the journal and reads the whole batch. */
                [[nodiscard]] Status read_rows(std::size_t index, RowRange rows,
                                               Batch& batch) const override
                {
                    SA_TRY(check_rows(*this, index, rows));
                    return read(index, batch);
                }

                /** @brief Returns every index read, first read first. */
                [[nodiscard]] std::vector<std::size_t> journal() const
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    return journal_;
                }

            protected:
                /** @brief Returns the epoch a read builds its batch for; 0 for this class. */
                virtual std::int64_t current_epoch() const noexcept
                {
                    return 0;
                }

            private:
                /// @brief Holds how many batches an epoch has.
                std::size_t batches_;

                /// @brief Holds the samples per batch.
                int64_t batch_size_;

                /// @brief Holds the input every batch carries.
                InputDescription input_;

                /// @brief Holds the roles every batch carries.
                std::vector<TargetDescription> targets_;

                /// @brief Holds the builder of a batch, or an empty function.
                BatchMaker maker_;

                /// @brief Guards the journal and the failing index.
                mutable std::mutex mutex_;

                /// @brief Holds the indices read, in order.
                mutable std::vector<std::size_t> journal_;

                /// @brief Holds the index whose read fails; no index when none does.
                std::size_t failing_ = static_cast<std::size_t>(-1);
        };

        /**
         * @class InMemoryEpochDataset
         * @brief An InMemoryDataset that is EpochAware: its batch maker is told the epoch last
         *        begun.
         */
        class InMemoryEpochDataset final : public InMemoryDataset, public InMemoryEpochAware
        {
            public:
                using InMemoryDataset::InMemoryDataset;

            protected:
                /** @brief Returns the epoch last begun, or 0 before the first call. */
                std::int64_t current_epoch() const noexcept override
                {
                    return epoch() < 0 ? 0 : epoch();
                }
        };

    } // namespace Data
} // namespace SushiAI

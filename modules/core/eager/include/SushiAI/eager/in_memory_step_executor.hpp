/**************************************************************************/
/* in_memory_step_executor.hpp                                            */
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
 * @file in_memory_step_executor.hpp
 * @brief Declares InMemoryStepExecutor, a StepExecutor that runs no region.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @class InMemoryStepExecutor
         * @brief Journals the work of each step and answers with a scripted status; touches no
         *        device, file, thread or clock.
         *
         * It records no region and builds no program, so it leaves the program of every region
         * where it was and the storage bank as it found it.
         */
        class InMemoryStepExecutor final : public StepExecutor
        {
            public:
                /** @brief Holds what one call of execute() was handed. */
                struct Seen
                {
                    /** @brief How many regions the step held. */
                    std::size_t regions = 0;

                    /** @brief How many keys the step dropped. */
                    std::size_t dropped = 0;

                    /** @brief The highest region key, or K_NO_REGION for a step with no region. */
                    RegionKey highest_key = K_NO_REGION;

                    /** @brief Whether the step held trailing work. */
                    bool has_trailing = false;

                    /** @brief The key of the trailing work. */
                    RegionKey trailing_key = K_NO_REGION;

                    /** @brief The number of steps finished before this one. */
                    std::uint64_t run_index = 0;
                };

                /**
                 * @brief Makes every later execute() return @p outcome.
                 * @param outcome A failure, or a success to run steps again.
                 */
                void script_execute(Status outcome) noexcept
                {
                    execute_outcome_ = std::move(outcome);
                }

                /**
                 * @brief Makes every later discard() return @p outcome.
                 * @param outcome A failure, or a success to discard again.
                 */
                void script_discard(Status outcome) noexcept
                {
                    discard_outcome_ = std::move(outcome);
                }

                /** @brief Journals @p work, then returns the scripted status. */
                [[nodiscard]] Status execute(StepWork& work, StorageBank&) override
                {
                    Seen seen;
                    seen.regions = work.regions.size();
                    seen.dropped = work.dropped.size();
                    for (const RegionWork& region : work.regions)
                    {
                        if (region.order.key > seen.highest_key)
                            seen.highest_key = region.order.key;
                    }
                    seen.has_trailing = work.trailing != nullptr;
                    seen.trailing_key = work.trailing_key;
                    seen.run_index = work.run_index;
                    seen_.push_back(seen);

                    return execute_outcome_;
                }

                /** @brief Counts the call and returns the scripted status, a success by default. */
                [[nodiscard]] Status discard() override
                {
                    ++discard_count_;
                    return discard_outcome_;
                }

                /** @brief Returns what each call of execute() was handed, first call first. */
                [[nodiscard]] const std::vector<Seen>& seen() const noexcept
                {
                    return seen_;
                }

                /** @brief Returns how many times discard() was called. */
                [[nodiscard]] std::size_t discard_count() const noexcept
                {
                    return discard_count_;
                }

            private:
                /// @brief Holds the status execute() returns.
                Status execute_outcome_;

                /// @brief Holds the status discard() returns.
                Status discard_outcome_;

                /// @brief Holds the journal of execute().
                std::vector<Seen> seen_;

                /// @brief Counts the calls of discard().
                std::size_t discard_count_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI

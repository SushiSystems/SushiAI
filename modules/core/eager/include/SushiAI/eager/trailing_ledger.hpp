/**************************************************************************/
/* trailing_ledger.hpp                                                    */
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
 * @file trailing_ledger.hpp
 * @brief Declares TrailingLedger, which plans the trailing work of a step and remembers it.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>

#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Holds what a step does with the trailing work the ledger remembers. */
        struct TrailingPlan
        {
            bool retire = false; /**< The remembered trailing work leaves the graph first. */
            bool record = false; /**< The step's trailing work is recorded after the regions. */
        };

        /**
         * @brief Decides which trailing work a step records, behind which regions, and remembers
         *        the work that was recorded; touches no runtime, device or thread.
         *
         * The executor asks for a plan, then carries it out on its runtime and reports the
         * recording back. The trailing work runs once per run, after every region, because its
         * key lies above theirs.
         */
        class TrailingLedger final
        {
            public:
                /**
                 * @brief Plans what @p work does with the remembered trailing work.
                 * @return The plan, or InvalidDescription when the trailing work would be recorded
                 *         under no key or under a key that does not lie above every region.
                 */
                [[nodiscard]] Result<TrailingPlan> plan(const StepWork& work) const;

                /** @brief Keeps @p key before the recording, so a failed one can be dropped. */
                void begin_recording(RegionKey key) noexcept;

                /** @brief Remembers @p recorder, which recorded @p operations tasks. */
                void finish_recording(Graph::StepRecorder& recorder,
                                      std::size_t operations) noexcept;

                /** @brief Forgets the key and the recorder. */
                void clear() noexcept;

                /** @brief Returns the key the trailing work is kept under, or K_NO_REGION. */
                [[nodiscard]] RegionKey key() const noexcept
                {
                    return key_;
                }

                /** @brief Returns how many tasks the remembered trailing work recorded. */
                [[nodiscard]] std::size_t operation_count() const noexcept
                {
                    return recorder_ != nullptr ? operations_ : 0;
                }

                /** @brief Returns how many times a recording was begun. */
                [[nodiscard]] std::size_t record_count() const noexcept
                {
                    return records_;
                }

                /** @brief Refreshes the remembered recorder's host state before a replay. */
                void before_replay();

                /** @brief Tells the remembered recorder that a replay finished. */
                void after_replay();

            private:
                /// @brief The recorder of the trailing work, or null; not owned.
                Graph::StepRecorder* recorder_ = nullptr;

                /// @brief The key the trailing work is recorded under, or K_NO_REGION.
                RegionKey key_ = K_NO_REGION;

                /// @brief The tasks the trailing work recorded.
                std::size_t operations_ = 0;

                /// @brief How many recordings were begun.
                std::size_t records_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI

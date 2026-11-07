/**************************************************************************/
/* region_executor.hpp                                                    */
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
 * @file region_executor.hpp
 * @brief Declares RegionExecutor, the StepExecutor over a SushiRuntime DynamicGraph.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/execution/runtime_backend.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/region_emitter.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/trailing_ledger.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Keeps one runtime region per region of the step and runs them as one graph.
         * @see modules/core/eager/README.md
         */
        class RegionExecutor final : public StepExecutor
        {
            public:
                /**
                 * @brief Starts with no region, on @p runtime.
                 * @param backend Allocates the regions' tensors, names the device and gives the
                 *        runtime the regions run on; must outlive the executor.
                 * @param selector Picks the form of a fused node; null selects by device.
                 */
                RegionExecutor(RuntimeBackend& backend,
                               const Graph::FusionSelector* selector = nullptr) noexcept;

                RegionExecutor(const RegionExecutor&) = delete;
                RegionExecutor& operator=(const RegionExecutor&) = delete;
                RegionExecutor(RegionExecutor&&) = delete;
                RegionExecutor& operator=(RegionExecutor&&) = delete;
                ~RegionExecutor() override;

                /**
                 * @brief Applies @p work to the dynamic graph and runs one step of it.
                 * @return Success, also for a step that holds no task, or the first failure.
                 */
                [[nodiscard]] Status execute(StepWork& work, StorageBank& storage) override;

                /** @brief Drops every region and the trailing work. */
                [[nodiscard]] Status discard() override;

                /** @brief Returns how many times the runtime has composed the dynamic graph. */
                [[nodiscard]] std::size_t compose_count() const noexcept;

                /** @brief Returns how many regions are live, the trailing one excluded. */
                [[nodiscard]] std::size_t region_count() const noexcept
                {
                    return live_.size();
                }

                /** @brief Returns how many times region @p id has been lowered into a recorder. */
                [[nodiscard]] std::size_t emit_count(RegionID id) const noexcept;

                /** @brief Returns how many times any region has been lowered into a recorder. */
                [[nodiscard]] std::size_t emit_count() const noexcept
                {
                    return emits_total_;
                }

                /** @brief Returns how many times the trailing work has been recorded. */
                [[nodiscard]] std::size_t trailing_record_count() const noexcept
                {
                    return trailing_.record_count();
                }

                /** @brief Returns the program of the live region under @p key, or null. */
                [[nodiscard]] const RegionProgram* program_of(RegionKey key) const noexcept;

                /** @brief Returns the bytes the pools of the live regions have allocated. */
                [[nodiscard]] std::size_t allocated_bytes() const noexcept;

                /** @brief Returns the tasks of the live regions and of the trailing work. */
                [[nodiscard]] std::size_t operation_count() const noexcept;

                /** @brief Returns the report the last run refilled. */
                [[nodiscard]] const SushiRuntime::RunReport& last_report() const noexcept
                {
                    return report_;
                }

            private:
                /** @brief Holds what one recorded region owns. */
                /** @brief Holds the storage of a program's imports and exports, in their order. */
                struct SharedStorage
                {
                    std::vector<StorageID> imports; /**< One id per RegionProgram::imports. */
                    std::vector<StorageID> exports; /**< One id per RegionProgram::exports. */
                };

                struct LiveRegion
                {
                    /** @brief Takes @p compiled and binds an empty pool and an emitter to it. */
                    LiveRegion(RuntimeBackend& backend, RegionID region,
                               RegionProgram compiled, SharedStorage placed,
                               const Graph::FusionSelector* selector);

                    LiveRegion(const LiveRegion&) = delete;
                    LiveRegion& operator=(const LiveRegion&) = delete;
                    LiveRegion(LiveRegion&&) = delete;
                    LiveRegion& operator=(LiveRegion&&) = delete;
                    ~LiveRegion() = default;

                    RegionID id;            /**< The region across steps. */
                    RegionProgram program;  /**< The compiled region. */
                    SharedStorage shared;   /**< Where program's imports and exports live. */
                    Graph::TensorPool pool; /**< The region's arenas and tensors. */
                    RegionEmitter emitter;  /**< Records program through pool. */
                };

                /** @brief Maps a key of the last step to the region that held it. */
                using Retired = std::unordered_map<RegionKey, std::unique_ptr<LiveRegion>>;

                /**
                 * @brief Drops what @p work replaces or removes from the dynamic graph.
                 * @param trailing What the step does with the trailing work.
                 * @param retired Receives each replaced region, by the key it held.
                 */
                void retire(const StepWork& work, const TrailingPlan& trailing, Retired& retired);

                /**
                 * @brief Records @p program under the key of @p order, in a pool of its own.
                 * @param shared The storage of the program's imports and exports.
                 * @return Success, or the failure of an allocation or of the recording.
                 */
                [[nodiscard]] Status install(const RegionOrder& order, RegionProgram program,
                                             SharedStorage shared, StorageBank& storage);

                /** @brief Records the trailing work of @p work when @p trailing says so. */
                [[nodiscard]] Status record_trailing(const StepWork& work,
                                                     const TrailingPlan& trailing);

                /** @brief Calls the preludes and the trailing hooks around one run. */
                [[nodiscard]] Status run(std::uint64_t run_index);

                /// @brief The backend whose engine allocates the regions and records their tasks.
                RuntimeBackend* backend_;

                /// @brief The caller's selector, or null to select by device.
                const Graph::FusionSelector* selector_;

                /// @brief The regions of the last step, by key.
                std::map<RegionKey, std::unique_ptr<LiveRegion>> live_;

                /// @brief How many times each region was lowered, by region id.
                std::unordered_map<RegionID, std::size_t> emits_;

                /// @brief Plans the trailing work and remembers what was recorded.
                TrailingLedger trailing_;

                /// @brief Keeps the late-count tasks of the trailing work while it is recorded.
                SushiBLAS::LateCountHold trailing_counts_;

                /// @brief How many times any region was lowered.
                std::size_t emits_total_ = 0;

                /// @brief The report each run refills.
                SushiRuntime::RunReport report_;

                /// @brief The graph that composes the regions; declared last so it dies first.
                SushiRuntime::DynamicGraph graph_;
        };

    } // namespace Eager
} // namespace SushiAI

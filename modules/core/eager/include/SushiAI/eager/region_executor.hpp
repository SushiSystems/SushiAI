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

#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/region_emitter.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/step_executor.hpp>
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
                 * @brief Starts with no region, on the runtime of @p engine.
                 * @param engine Must outlive the executor.
                 * @param selector Picks the form of a fused node; null selects by device.
                 */
                explicit RegionExecutor(SushiBLAS::Engine& engine,
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
                    return trailing_records_;
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
                    LiveRegion(SushiBLAS::Engine& engine, RegionID region, RegionProgram compiled,
                               SharedStorage placed, const Graph::FusionSelector* selector);

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
                 * @param retired Receives each replaced region, by the key it held.
                 */
                void retire(const StepWork& work, Retired& retired);

                /**
                 * @brief Records @p program under the key of @p order, in a pool of its own.
                 * @param shared The storage of the program's imports and exports.
                 * @return Success, or the failure of an allocation or of the recording.
                 */
                [[nodiscard]] Status install(const RegionOrder& order, RegionProgram program,
                                             SharedStorage shared, StorageBank& storage);

                /** @brief Records the trailing work of @p work unless it is recorded already. */
                [[nodiscard]] Status record_trailing(const StepWork& work);

                /** @brief Calls the preludes and the trailing hooks around one run. */
                [[nodiscard]] Status run(std::uint64_t run_index);

                /// @brief The engine whose runtime runs the regions.
                SushiBLAS::Engine* engine_;

                /// @brief The caller's selector, or null to select by device.
                const Graph::FusionSelector* selector_;

                /// @brief The regions of the last step, by key.
                std::map<RegionKey, std::unique_ptr<LiveRegion>> live_;

                /// @brief How many times each region was lowered, by region id.
                std::unordered_map<RegionID, std::size_t> emits_;

                /// @brief The recorder of the trailing work, or null; not owned.
                Graph::StepRecorder* trailing_ = nullptr;

                /// @brief The key the trailing work is recorded under, or K_NO_REGION.
                RegionKey trailing_key_ = K_NO_REGION;

                /// @brief Keeps the late-count tasks of the trailing work while it is recorded.
                SushiBLAS::LateCountHold trailing_counts_;

                /// @brief The tasks the trailing work recorded.
                std::size_t trailing_operations_ = 0;

                /// @brief How many times the trailing work was recorded.
                std::size_t trailing_records_ = 0;

                /// @brief How many times any region was lowered.
                std::size_t emits_total_ = 0;

                /// @brief The report each run refills.
                SushiRuntime::RunReport report_;

                /// @brief The graph that composes the regions; declared last so it dies first.
                SushiRuntime::DynamicGraph graph_;
        };

    } // namespace Eager
} // namespace SushiAI

/**************************************************************************/
/* step_executor.hpp                                                      */
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
 * @file step_executor.hpp
 * @brief Declares StepExecutor, the interface that runs one planned step.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Holds one region of a step with the decision taken for it. */
        struct RegionWork
        {
            RegionOrder order;     /**< What the step does with the region. */
            RegionProgram program; /**< The compiled region; filled when the action is Build. */
        };

        /**
         * @brief Holds everything an executor needs to run one step.
         * @see modules/core/eager/README.md
         */
        struct StepWork
        {
            /** @brief The regions of the step, in ascending key order. */
            std::vector<RegionWork> regions;

            /** @brief The last step's keys of the regions this step does not have. */
            std::vector<RegionKey> dropped;

            /** @brief The storage of each boundary value and leaf, by value id. */
            SushiRuntime::span<const StorageID> storage_of_value;

            /** @brief Records the work that runs after every region, or null; not owned. */
            Graph::StepRecorder* trailing = nullptr;

            /** @brief The key the trailing work is recorded under, above every region. */
            RegionKey trailing_key = K_NO_REGION;

            /** @brief The number of steps the session has finished. */
            std::uint64_t run_index = 0;
        };

        /**
         * @brief Applies the decisions of one step and runs it until it has finished.
         * @see modules/core/eager/README.md
         */
        class StepExecutor
        {
            public:
                /** @brief Destroys the executor. */
                virtual ~StepExecutor() = default;

                /**
                 * @brief Drops, records and builds what @p work says, then runs every region.
                 * @param work The step; the program of each built region is moved from.
                 * @param storage Holds the tensor of every id @p work names.
                 * @return Success, or the first failure; after one, call discard().
                 */
                [[nodiscard]] virtual Status execute(StepWork& work, StorageBank& storage) = 0;

                /**
                 * @brief Drops every region, so the next step must build all of them.
                 * @return Success, or OutOfMemory when a drop could not be listed.
                 */
                [[nodiscard]] virtual Status discard() = 0;
        };

    } // namespace Eager
} // namespace SushiAI

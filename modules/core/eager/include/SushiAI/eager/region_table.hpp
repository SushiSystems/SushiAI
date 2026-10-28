/**************************************************************************/
/* region_table.hpp                                                       */
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
 * @file region_table.hpp
 * @brief Declares RegionTable, which decides what a step keeps of the step before it.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/region_signature.hpp>
#include <SushiAI/eager/slot_ledger.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Says what a step does with one region. */
        enum class RegionAction : std::uint8_t
        {
            Keep,   /**< The recorded region runs again as it is. */
            Record, /**< The compiled region is recorded again, under another key. */
            Build,  /**< The region is compiled and recorded. */
        };

        /** @brief Holds the decision for one region of a step. */
        struct RegionOrder
        {
            RegionID id = K_NO_REGION_ID;              /**< The region across steps. */
            RegionKey key = K_NO_REGION;               /**< Its key in this step. */
            RegionKey previous_key = K_NO_REGION;      /**< Its key in the last step, or none. */
            RegionAction action = RegionAction::Build; /**< What the step does with it. */
        };

        /**
         * @brief Holds what one step keeps, rebuilds, drops, allocates and frees.
         * @see modules/core/eager/README.md
         */
        struct StepPlan
        {
            /** @brief One decision per region, in the order of Partition::regions. */
            std::vector<RegionOrder> regions;

            /** @brief The last step's keys of the regions this step does not have. */
            std::vector<RegionKey> dropped;

            /** @brief The storage of each boundary value and leaf, by value id. */
            std::vector<StorageID> storage_of_value;

            /** @brief The storage this step needs and the session does not hold. */
            std::vector<SlotRecord> created;

            /** @brief The storage no region of this step names. */
            std::vector<StorageID> released;
        };

        /**
         * @brief Remembers the regions of the last step by name and compares a new step with them.
         * @see modules/core/eager/README.md
         */
        class RegionTable final
        {
            public:
                /**
                 * @brief Starts with no region.
                 * @param ids Hands out the id of each new boundary slot; must outlive the table.
                 */
                explicit RegionTable(StorageIDSource& ids) noexcept;

                RegionTable(const RegionTable&) = delete;
                RegionTable& operator=(const RegionTable&) = delete;
                RegionTable(RegionTable&&) = delete;
                RegionTable& operator=(RegionTable&&) = delete;
                ~RegionTable() = default;

                /**
                 * @brief Plans one step and records it as the step to compare the next with.
                 * @param boundaries What find_boundaries() returned for @p partition.
                 * @param preset The storage the caller placed each value in, by value id.
                 * @pre Every leaf has storage and no buffer is placed twice; call forget() before
                 *      planning again.
                 * @see modules/core/eager/README.md
                 */
                [[nodiscard]] StepPlan plan(const Graph::Graph& graph, const Partition& partition,
                                            const BoundaryMap& boundaries,
                                            SushiRuntime::span<const StorageID> preset);

                /**
                 * @brief Forgets every region, so the next step builds all of them.
                 * @return The ids of the boundary slots the table held.
                 */
                [[nodiscard]] std::vector<StorageID> forget();

                /** @brief Returns the names of the regions this table has seen. */
                [[nodiscard]] const RegionNames& names() const noexcept
                {
                    return names_;
                }

                /** @brief Returns how many regions the last planned step had. */
                [[nodiscard]] std::size_t live_count() const noexcept
                {
                    return live_.size();
                }

            private:
                /** @brief Holds what the last step recorded for one region. */
                struct Entry
                {
                    bool live = false;                           /**< In the last step. */
                    std::uint64_t step = 0;                      /**< The step that saw it last. */
                    RegionKey key = K_NO_REGION;                 /**< Its key in that step. */
                    RegionSignature signature;                   /**< Structure and storage. */
                    RegionSignature structure;                   /**< Structure alone. */
                    std::vector<ValuePosition> export_positions; /**< Where its exports sit. */
                    std::vector<StorageID> export_storage;       /**< Where its exports live. */
                };

                /// @brief Gives each region one id across steps.
                RegionNames names_;

                /// @brief Matches each shared buffer with a slot.
                SlotLedger slots_;

                /// @brief What the last step recorded, by region id.
                std::vector<Entry> entries_;

                /// @brief The ids of the regions of the last step.
                std::vector<RegionID> live_;

                /// @brief Counts plan() calls; never 0 after the first.
                std::uint64_t step_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI

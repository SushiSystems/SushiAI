/**************************************************************************/
/* region_program.hpp                                                     */
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
 * @file region_program.hpp
 * @brief Declares RegionProgram, one region as a graph the lowering can record on its own.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief States how the passes treat every region of a session. */
        struct ProgramOptions
        {
            /** @brief Decides each node's precision; null leaves every type as traced. */
            const Graph::PrecisionPolicy* precision = nullptr;

            /** @brief Options of the fusion pass; its observed list is replaced per region. */
            Graph::FusionOptions fusion;

            /** @brief Options of the memory planner; its pinned list is replaced per region. */
            Graph::PlanOptions plan;

            /** @brief Whether stored K_IM2COL results are recomputed at their later readers. */
            bool rematerialise = true;
        };

        /**
         * @brief Holds one region after the passes, with the values it shares.
         * @see modules/core/eager/README.md
         */
        struct RegionProgram
        {
            /** @brief The region's nodes behind one K_LEAF per import; value ids are the step's. */
            Graph::Graph graph;

            /** @brief Where each value of graph lives inside the region's own arenas. */
            Graph::MemoryPlan plan;

            /** @brief What the cast pass converted. */
            Graph::CastReport casts;

            /** @brief What the fusion pass folded and what it refused. */
            Graph::FusionReport fusion;

            /** @brief The values the region reads from storage it does not own. */
            std::vector<Graph::ValueID> imports;

            /** @brief The values the region writes into storage it does not own. */
            std::vector<Graph::ValueID> exports;
        };

        /**
         * @brief Cuts region @p key out of @p graph and runs the passes over it.
         * @param boundary The entry find_boundaries() returned for @p key.
         * @return The program, with no export folded or packed, or a pass's refusal.
         * @pre @p partition fits @p graph and @p key names a region; else the process stops.
         * @see modules/core/eager/README.md
         */
        [[nodiscard]] Result<RegionProgram> compile_region(const Graph::Graph& graph,
                                                           const Partition& partition,
                                                           RegionKey key,
                                                           const RegionBoundary& boundary,
                                                           const ProgramOptions& options = {});

    } // namespace Eager
} // namespace SushiAI

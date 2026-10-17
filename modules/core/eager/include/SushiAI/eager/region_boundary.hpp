/**************************************************************************/
/* region_boundary.hpp                                                    */
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
 * @file region_boundary.hpp
 * @brief Declares BoundaryMap, the values each region of a partition shares with the others.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Names one result of one node by its place inside a region. */
        struct ValuePosition
        {
            std::uint32_t node = 0;   /**< Index of the producer among the region's nodes. */
            std::uint32_t result = 0; /**< Index of the value among the producer's results. */

            /** @brief Reports whether both indices agree. */
            bool operator==(const ValuePosition& other) const noexcept
            {
                return node == other.node && result == other.result;
            }

            /** @brief Reports whether an index differs. */
            bool operator!=(const ValuePosition& other) const noexcept
            {
                return !(*this == other);
            }
        };

        /**
         * @brief Lists the values one region reads from outside and writes for others.
         * @see modules/core/eager/README.md
         */
        struct RegionBoundary
        {
            /** @brief Values the region reads and does not produce, each once, in read order. */
            std::vector<Graph::ValueID> imports;

            /** @brief Values the region produces into a shared buffer, in node order. */
            std::vector<Graph::ValueID> exports;

            /** @brief The place of each export inside the region, parallel to exports. */
            std::vector<ValuePosition> export_positions;
        };

        /**
         * @brief Holds the boundary of every region of one partition.
         * @see modules/core/eager/README.md
         */
        struct BoundaryMap
        {
            /** @brief One boundary per region, in the order of Partition::regions. */
            std::vector<RegionBoundary> regions;

            /** @brief The value that names the buffer of each value, by value id. */
            std::vector<Graph::ValueID> buffer_of;
        };

        /**
         * @brief Finds the values that cross a region boundary of @p partition.
         * @param observed Values read from outside the graph; each is exported.
         * @return The imports and exports of every region and the buffer of every value, or
         *         StatusCode::InvalidDescription when an ACCUMULATE writes into a leaf.
         * @see modules/core/eager/README.md
         */
        [[nodiscard]] Result<BoundaryMap> find_boundaries(
            const Graph::Graph& graph, const Partition& partition,
            SushiRuntime::span<const Graph::ValueID> observed);

    } // namespace Eager
} // namespace SushiAI

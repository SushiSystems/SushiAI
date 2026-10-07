/**************************************************************************/
/* partition.hpp                                                          */
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
 * @file partition.hpp
 * @brief Declares Partition, the assignment of a differentiated graph's nodes to regions.
 * @author Mustafa Garip
 * @see include/SushiAI/eager/README.md
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Identifies a region; a lower key produces what a higher key consumes. */
        using RegionKey = std::uint64_t;

        /** @brief Stands for "this node belongs to no region". */
        inline constexpr RegionKey K_NO_REGION = 0;

        /** @brief Says which pass of a segment a region holds. */
        enum class Direction : std::uint8_t
        {
            Forward,  /**< The nodes the caller traced. */
            Backward, /**< The nodes differentiation appended for them. */
        };

        /** @brief Describes one region of a partition. */
        struct RegionInfo
        {
            RegionKey key = K_NO_REGION;              /**< The region's key. */
            std::string name;                         /**< Segment name and occurrence, "head#0". */
            Direction direction = Direction::Forward; /**< The pass the region holds. */
        };

        /**
         * @brief Assigns every node of one differentiated graph to a region.
         * @see include/SushiAI/eager/README.md
         */
        struct Partition
        {
            /** @brief Region of each node, by node id; K_NO_REGION for a LEAF node. */
            std::vector<RegionKey> region_of_node;

            /** @brief The regions that hold at least one node, in ascending key order. */
            std::vector<RegionInfo> regions;

            /** @brief A key above every region, reserved for the optimizer update. */
            RegionKey optimizer_key = K_NO_REGION;
        };

    } // namespace Eager
} // namespace SushiAI

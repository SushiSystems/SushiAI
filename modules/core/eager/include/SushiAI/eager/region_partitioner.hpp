/**************************************************************************/
/* region_partitioner.hpp                                                 */
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
 * @file region_partitioner.hpp
 * @brief Declares RegionPartitioner, the policy that cuts a differentiated graph into regions.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Assigns every node of a differentiated graph a region key.
         * @see modules/core/eager/README.md
         */
        class RegionPartitioner
        {
            public:
                /** @brief Destroys the partitioner. */
                virtual ~RegionPartitioner() = default;

                /**
                 * @brief Partitions @p graph.
                 * @param segments The forward trace's segments; they cover every forward node.
                 * @param provenance The record differentiate() filled for @p graph.
                 * @return A partition in which no node reads a value a higher key produces.
                 */
                [[nodiscard]] virtual Partition assign(
                    const Graph::Graph& graph,
                    SushiRuntime::span<const Segment> segments,
                    const Autograd::Provenance& provenance) const = 0;
        };

    } // namespace Eager
} // namespace SushiAI

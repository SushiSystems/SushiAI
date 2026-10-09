/**************************************************************************/
/* scope_partitioner.hpp                                                  */
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
 * @file scope_partitioner.hpp
 * @brief Declares ScopePartitioner, which makes one region per segment and direction.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <SushiAI/eager/region_partitioner.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Cuts a graph only where the caller opened or closed a scope.
         * @see modules/core/eager/README.md
         */
        class ScopePartitioner final : public RegionPartitioner
        {
            public:
                /**
                 * @brief Gives segment k key k and its backward nodes key 2m - k, for m segments.
                 * @param segments The forward trace's segments; they cover every forward node.
                 * @param provenance The record differentiate() filled for @p graph.
                 */
                [[nodiscard]] Partition assign(
                    const Graph::Graph& graph,
                    SushiRuntime::span<const Segment> segments,
                    const Autograd::Provenance& provenance) const override;
        };

    } // namespace Eager
} // namespace SushiAI

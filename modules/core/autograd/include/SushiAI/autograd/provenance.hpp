/**************************************************************************/
/* provenance.hpp                                                         */
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
 * @file provenance.hpp
 * @brief Declares Provenance, the forward node each backward node was emitted for.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <vector>

#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief Records, for every node differentiate() appended, the forward node it serves.
         * @see modules/core/eager/README.md
         */
        struct Provenance
        {
            /** @brief Node count of the graph before differentiation; the first backward id. */
            std::size_t forward_node_count = 0;

            /** @brief Origins in append order: origin[i] belongs to node forward_node_count + i. */
            std::vector<Graph::NodeId> origin;

            /**
             * @brief Returns the forward node that @p backward_node was emitted for.
             * @param backward_node A node id at or above forward_node_count.
             */
            Graph::NodeId origin_of(Graph::NodeId backward_node) const
            {
                SA_THROW_IF(backward_node < forward_node_count
                                || backward_node - forward_node_count >= origin.size(),
                            "Node {} is not a backward node of this differentiation, which "
                            "appended nodes {} to {}",
                            backward_node, forward_node_count,
                            forward_node_count + origin.size());
                return origin[backward_node - forward_node_count];
            }
        };

    } // namespace Autograd
} // namespace SushiAI

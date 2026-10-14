/**************************************************************************/
/* node_range.hpp                                                         */
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
 * @file node_range.hpp
 * @brief Declares NodeRange, a run of consecutive nodes of an IR graph.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Names the nodes of a graph from @c first up to, and without, @c end. */
        struct NodeRange
        {
            /** @brief The first node of the range. */
            NodeID first = 0;

            /** @brief The node behind the last; K_NO_NODE stands for the graph's node count. */
            NodeID end = K_NO_NODE;
        };

    } // namespace Graph
} // namespace SushiAI

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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file node_range.hpp
 * @brief Declares NodeRange, a run of consecutive nodes of an IR graph.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiAI/failure/status.hpp>
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

        /**
         * @brief Checks that @p nodes lies in a graph of @p node_count nodes.
         * @param nodes The range; an @c end of K_NO_NODE stands for @p node_count.
         * @param node_count The number of nodes of the graph.
         * @return Success, or InvalidDescription naming the range and the node count.
         */
        [[nodiscard]] inline Status check_node_range(NodeRange nodes,
                                                     std::size_t node_count) noexcept
        {
            const NodeID count = static_cast<NodeID>(node_count);
            const NodeID end = nodes.end == K_NO_NODE ? count : nodes.end;
            SA_FAIL_IF(nodes.first > end || end > count, StatusCode::InvalidDescription,
                       "The node range [{}, {}) does not lie in a graph of {} nodes",
                       nodes.first, end, count);

            return Status();
        }

    } // namespace Graph
} // namespace SushiAI

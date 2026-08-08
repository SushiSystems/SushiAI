/**************************************************************************/
/* rematerialise.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <cstddef>
#include <vector>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/rematerialise.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief The name every rematerialised column matrix carries. */
            constexpr const char* REMAT_NAME = "cols_remat";

            /**
             * @brief The results of every @c IM2COL in @p graph, in graph order.
             *
             * Collected as *values* rather than node ids on purpose. The pass
             * inserts nodes, which shifts every node id after the insertion
             * point; value ids never move, and a value's producer is kept in
             * step by @c Graph::insert_node(). Snapshotting first also keeps
             * the pass from rematerialising its own clones.
             */
            std::vector<ValueId> stored_column_matrices(const Graph& graph)
            {
                std::vector<ValueId> stored;
                for (const Node& node : graph.nodes())
                    if (node.op == Ops::IM2COL.id && !node.outputs.empty())
                        stored.push_back(node.outputs.front());
                return stored;
            }

            /** @brief The nodes after @p from that read @p value, in graph order. */
            std::vector<NodeId> consumers_after(const Graph& graph, NodeId from, ValueId value)
            {
                std::vector<NodeId> consumers;
                for (std::size_t i = from + 1; i < graph.node_count(); ++i)
                {
                    for (ValueId in : graph.nodes()[i].inputs)
                    {
                        if (in != value) continue;
                        consumers.push_back(static_cast<NodeId>(i));
                        break;
                    }
                }
                return consumers;
            }
        }

        void rematerialise_im2col(Graph& graph)
        {
            const GraphBuilder builder(graph);

            for (ValueId cols : stored_column_matrices(graph))
            {
                const NodeId gather = graph.value(cols).producer;
                const std::vector<NodeId> consumers = consumers_after(graph, gather, cols);

                // One consumer is the forward GEMM and nothing else reads the
                // matrix, so there is no live range to shorten; none at all is
                // a dead gather, which is not this pass's problem either.
                if (consumers.size() < 2) continue;

                // Both copied rather than referenced: the declaration below
                // grows the value table and the insertion moves the node table,
                // and a reference into either would be dangling by then.
                const Node source = graph.node(gather);
                const Value original = graph.value(cols);

                // declare_like, never a hand-written layout assignment - the
                // clone has to carry the layout of the value it replaces or
                // Graph::finalise() would reject a graph that is correct.
                const ValueId clone = builder.declare_like(cols, original.shape, original.dtype,
                                                           Lifetime::SCRATCH, REMAT_NAME);
                graph.value(clone).requires_grad = original.requires_grad;

                // Attributes carried across wholesale. IM2COL's slots 8 and 9
                // are output extents that cannot be re-derived without assuming
                // symmetric padding, so recomputing them here would be the one
                // mistake this pass can make that produces a plausible answer
                // instead of an error.
                graph.insert_node(consumers[1],
                                  Node{source.op, source.inputs, {clone}, source.attributes});

                // Every consumer from the second onwards reads the clone. Their
                // ids all shifted by the insertion, which landed at the first
                // of them.
                for (std::size_t i = 1; i < consumers.size(); ++i)
                    for (ValueId& in : graph.node(consumers[i] + 1).inputs)
                        if (in == cols) in = clone;

                // The stored copy now dies at the forward GEMM, which is the
                // node right after its gather. That is the whole point.
                graph.value(cols).lifetime = Lifetime::SCRATCH;
            }
        }

    } // namespace Graph
} // namespace SushiAI

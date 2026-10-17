/**************************************************************************/
/* rematerialise.cpp                                                      */
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
 * @file rematerialise.cpp
 * @brief Implements column matrix rematerialisation for convolution backward passes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/rematerialise.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief The name every rematerialised column matrix carries. */
            constexpr const char* K_REMAT_NAME = "cols_remat";

            /**
             * @brief Collects output value ids of every IM2COL node in graph order.
             *
             * Value ids remain stable when nodes are inserted during the pass.
             */
            std::vector<ValueID> stored_column_matrices(const Graph& graph)
            {
                std::vector<ValueID> stored;
                for (const Node& node : graph.nodes())
                    if (node.op == Ops::K_IM2COL.id && !node.outputs.empty())
                        stored.push_back(node.outputs.front());
                return stored;
            }

            /** @brief The nodes after @p from that read @p value, in graph order. */
            std::vector<NodeID> consumers_after(const Graph& graph, NodeID from, ValueID value)
            {
                std::vector<NodeID> consumers;
                for (std::size_t i = from + 1; i < graph.node_count(); ++i)
                {
                    for (ValueID in : graph.nodes()[i].inputs)
                    {
                        if (in != value) continue;
                        consumers.push_back(static_cast<NodeID>(i));
                        break;
                    }
                }
                return consumers;
            }
        }

        void rematerialise_im2col(Graph& graph)
        {
            const GraphBuilder builder(graph);

            for (ValueID cols : stored_column_matrices(graph))
            {
                const NodeID gather = graph.value(cols).producer;
                const std::vector<NodeID> consumers = consumers_after(graph, gather, cols);

                // Rematerialisation targets column matrices consumed across multiple phases.
                if (consumers.size() < 2) continue;

                // Values are copied by value to prevent invalidation during table reallocation.
                const Node source = graph.node(gather);
                const Value original = graph.value(cols);

                // Clones must preserve original tensor layout to satisfy layout proof invariants.
                const ValueID clone = builder.declare_like(cols, original.shape, original.dtype,
                                                           Lifetime::Scratch, K_REMAT_NAME);
                graph.value(clone).requires_grad = original.requires_grad;

                // Geometric attributes are cloned directly to preserve exact padding extents.
                graph.insert_node(consumers[1],
                                  Node{source.op, source.inputs, {clone}, source.attributes});

                // Subsequent consumers are updated to read the cloned value operand.
                for (std::size_t i = 1; i < consumers.size(); ++i)
                    for (ValueID& in : graph.node(consumers[i] + 1).inputs)
                        if (in == cols) in = clone;

                // The original column matrix is demoted to scratch lifetime.
                graph.value(cols).lifetime = Lifetime::Scratch;
            }
        }

    } // namespace Graph
} // namespace SushiAI

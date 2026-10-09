/**************************************************************************/
/* lowering.cpp                                                           */
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
 * @file lowering.cpp
 * @brief Implements the lowering driver, which records each node through its table row.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering.hpp>

#include <SushiAI/core/check.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/lowering/all.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status Lowering::record(const Graph& graph)
        {
            emitted_ = 0;
            selections_.clear();

            graph.prove_layout_uniformity();

            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
            {
                const Node& node = graph.node(id);
                const LoweringRule rule = find_lowering(node.op);
                SA_FATAL_IF(rule == nullptr,
                            "No lowering for operation {} (node {}). Add a row to its domain's "
                            "table under include/SushiAI/graph/lowering/, or list it in "
                            "unlowered.hpp.",
                            Ops::name_of(node.op), id);

                const LoweringContext context(*recorder_, *stream_, *pool_, graph, id, *selector_,
                                              selections_, emitted_);
                const Status recorded = ::SushiAI::Detail::guard(
                    ::SushiAI::Detail::SeamPhase::Record, [&]() { return rule(context); });
                if (!recorded.is_ok())
                {
                    return SA_ANNOTATE(recorded, "{} (node {})", Ops::name_of(node.op), id);
                }
            }

            return Status();
        }

    } // namespace Graph
} // namespace SushiAI

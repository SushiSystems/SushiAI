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

#include <string>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/lowering/all.hpp>

namespace SushiAI
{
    namespace Graph
    {
        std::vector<ScratchRequest> scratch_of(const Graph& graph) noexcept
        {
            std::vector<ScratchRequest> requests;
            for (NodeID id = 0; id < static_cast<NodeID>(graph.node_count()); ++id)
            {
                const ScratchRule rule = find_scratch_rule(graph.node(id).op);
                if (rule != nullptr) rule(graph, id, requests);
            }

            return requests;
        }

        Status Lowering::record(const Graph& graph, NodeRange nodes) noexcept
        {
            emitted_ = 0;
            selections_.clear();
            preludes_.clear();

            const NodeID count = static_cast<NodeID>(graph.node_count());
            const NodeID end = nodes.end == K_NO_NODE ? count : nodes.end;
            SA_FAIL_IF(nodes.first > end || end > count, StatusCode::InvalidDescription,
                       "The node range [{}, {}) does not lie in a graph of {} nodes",
                       nodes.first, end, count);

            SA_TRY(graph.prove_layout_uniformity());

            for (NodeID id = nodes.first; id < end; ++id)
            {
                const Node& node = graph.node(id);
                const LoweringRule rule = find_lowering(node.op);
                SA_FATAL_IF(rule == nullptr,
                            "No lowering for operation {} (node {}). Add a row to its domain's "
                            "table under SushiAI/graph/lowering/, or list it in "
                            "unlowered.hpp.",
                            Ops::name_of(node.op), id);

                // The origin is built once per record() and never on a run.
                const std::string origin = origin_prefix_ + std::to_string(id) + ":"
                                           + std::string(Ops::name_of(node.op));
                const SushiBLAS::TaskRecorder::OriginScope scope = recorder_->origin(origin);
                const LoweringContext context(*recorder_, *stream_, *pool_, graph, id, *selector_,
                                              selections_, emitted_, preludes_);
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

/**************************************************************************/
/* region_program.cpp                                                     */
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
 * @file region_program.cpp
 * @brief Implements compile_region, which turns one region into a graph of its own.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_program.hpp>

#include <cstddef>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/rematerialise.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Views @p values as the span the passes read. */
            [[nodiscard]] SushiRuntime::span<const Graph::ValueID> view_of(
                const std::vector<Graph::ValueID>& values) noexcept
            {
                return SushiRuntime::span<const Graph::ValueID>(values.data(), values.size());
            }

            /**
             * @brief Declares each import of @p boundary as a leaf of @p region.
             * @param graph The step's graph, read for what produces each import there.
             * @see modules/core/eager/README.md
             */
            void declare_imports(const Graph::Graph& graph, const RegionBoundary& boundary,
                                 Graph::Graph& region)
            {
                for (Graph::ValueID value : boundary.imports)
                {
                    const Graph::NodeID producer = graph.value(value).producer;
                    const bool is_leaf = producer == Graph::K_NO_NODE
                                         || graph.node(producer).op == Ops::K_LEAF.id;

                    // A value another region computes is rewritten every step, as an input is.
                    if (!is_leaf)
                        region.value(value).lifetime = Graph::Lifetime::Input;

                    region.add_node(Ops::K_LEAF.id, {}, {value});
                }
            }

            /** @brief Appends the nodes @p partition files under @p key, in node order. */
            void copy_nodes(const Graph::Graph& graph, const Partition& partition, RegionKey key,
                            Graph::Graph& region)
            {
                for (std::size_t i = 0; i < graph.node_count(); ++i)
                {
                    if (partition.region_of_node[i] != key)
                        continue;

                    const Graph::Node& node = graph.node(static_cast<Graph::NodeID>(i));
                    region.add_node(node.op, node.inputs, node.outputs, node.attributes);
                }
            }

        } // namespace

        RegionProgram compile_region(const Graph::Graph& graph, const Partition& partition,
                                     RegionKey key, const RegionBoundary& boundary,
                                     const ProgramOptions& options)
        {
            SA_FATAL_IF(partition.region_of_node.size() != graph.node_count(),
                        "The partition assigns {} nodes but the graph holds {}",
                        partition.region_of_node.size(), graph.node_count());
            SA_FATAL_IF(key == K_NO_REGION, "Region key {} names no region", key);

            RegionProgram program;
            program.imports = boundary.imports;
            program.exports = boundary.exports;

            program.graph = graph.without_nodes();
            declare_imports(graph, boundary, program.graph);
            copy_nodes(graph, partition, key, program.graph);

            std::vector<Graph::ValueID> shared = boundary.imports;
            shared.insert(shared.end(), boundary.exports.begin(), boundary.exports.end());

            Graph::CastOptions cast_options;
            cast_options.policy = options.precision;
            cast_options.observed = view_of(shared);
            program.casts = Graph::insert_casts(program.graph, cast_options);

            Graph::FusionOptions fusion_options = options.fusion;
            fusion_options.observed = view_of(shared);
            program.fusion = Graph::fuse(program.graph, fusion_options);

            if (options.rematerialise)
                Graph::rematerialise_im2col(program.graph);

            Graph::PlanOptions plan_options = options.plan;
            plan_options.pinned = view_of(boundary.exports);
            const std::vector<Graph::ScratchRequest> scratch = Graph::scratch_of(program.graph);
            program.plan = Graph::MemoryPlan::build(
                program.graph, plan_options,
                SushiRuntime::span<const Graph::ScratchRequest>(scratch.data(), scratch.size()));

            return program;
        }

    } // namespace Eager
} // namespace SushiAI

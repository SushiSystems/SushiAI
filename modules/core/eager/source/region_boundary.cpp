/**************************************************************************/
/* region_boundary.cpp                                                    */
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
 * @file region_boundary.cpp
 * @brief Implements find_boundaries, which lists the values each region shares.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_boundary.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Stands for "no region" among positions in Partition::regions. */
            constexpr std::uint32_t K_NO_INDEX = ~std::uint32_t{0};

            /** @brief Reports whether a node writes @p value and that node is not a LEAF. */
            [[nodiscard]] bool is_computed(const Graph::Graph& graph, Graph::ValueID value)
            {
                const Graph::NodeID producer = graph.value(value).producer;
                return producer != Graph::K_NO_NODE
                       && graph.node(producer).op != Ops::K_LEAF.id;
            }

            /** @brief Returns the representative of @p value's set, halving the path to it. */
            [[nodiscard]] Graph::ValueID find_buffer(std::vector<Graph::ValueID>& parent,
                                                     Graph::ValueID value) noexcept
            {
                while (parent[value] != value)
                {
                    parent[value] = parent[parent[value]];
                    value = parent[value];
                }

                return value;
            }

            /**
             * @brief Returns, for each value, the lowest value id that shares its buffer.
             * @see modules/core/eager/README.md
             */
            [[nodiscard]] std::vector<Graph::ValueID> buffers_of(const Graph::Graph& graph)
            {
                std::vector<Graph::ValueID> parent(graph.value_count());
                for (std::size_t value = 0; value < parent.size(); ++value)
                    parent[value] = static_cast<Graph::ValueID>(value);

                for (const Graph::Node& node : graph.nodes())
                {
                    if (node.op != Ops::K_ACCUMULATE.id || node.inputs.empty()
                        || node.outputs.empty())
                        continue;

                    SA_THROW_IF(!is_computed(graph, node.inputs[0]),
                                "ACCUMULATE writes into value {}, which is a leaf; a region "
                                "cannot add into storage the session does not own",
                                node.inputs[0]);

                    const Graph::ValueID a = find_buffer(parent, node.outputs[0]);
                    const Graph::ValueID b = find_buffer(parent, node.inputs[0]);
                    parent[std::max(a, b)] = std::min(a, b);
                }

                for (std::size_t value = 0; value < parent.size(); ++value)
                    parent[value] = find_buffer(parent, static_cast<Graph::ValueID>(value));

                return parent;
            }

            /** @brief Maps a region key to its position in Partition::regions. */
            class RegionIndex final
            {
                public:
                    /** @brief Indexes the regions of @p partition. */
                    explicit RegionIndex(const Partition& partition)
                    {
                        RegionKey highest = K_NO_REGION;
                        for (const RegionInfo& info : partition.regions)
                            highest = std::max(highest, info.key);

                        position_.assign(static_cast<std::size_t>(highest) + 1, K_NO_INDEX);
                        for (std::size_t i = 0; i < partition.regions.size(); ++i)
                            position_[static_cast<std::size_t>(partition.regions[i].key)] =
                                static_cast<std::uint32_t>(i);
                    }

                    /** @brief Returns the position of @p key, or K_NO_INDEX for an unlisted key. */
                    [[nodiscard]] std::uint32_t of(RegionKey key) const noexcept
                    {
                        return key < position_.size() ? position_[static_cast<std::size_t>(key)]
                                                      : K_NO_INDEX;
                    }

                private:
                    /// @brief Position of each key in Partition::regions, by key.
                    std::vector<std::uint32_t> position_;
            };

            /** @brief Records which buffers more than one region touches. */
            class SharedBuffers final
            {
                public:
                    /** @brief Starts with @p value_count buffers, none of them touched. */
                    explicit SharedBuffers(std::size_t value_count)
                        : first_(value_count, K_NO_INDEX), shared_(value_count, false)
                    {

                    }

                    /** @brief Notes that region @p region reads or writes @p buffer. */
                    void touch(Graph::ValueID buffer, std::uint32_t region)
                    {
                        if (first_[buffer] == K_NO_INDEX)
                            first_[buffer] = region;
                        else if (first_[buffer] != region)
                            shared_[buffer] = true;
                    }

                    /** @brief Marks @p buffer as read from outside the graph. */
                    void observe(Graph::ValueID buffer)
                    {
                        shared_[buffer] = true;
                    }

                    /** @brief Reports whether @p buffer is observed or touched by two regions. */
                    [[nodiscard]] bool holds(Graph::ValueID buffer) const
                    {
                        return shared_[buffer];
                    }

                private:
                    /// @brief The first region that touched each buffer, or K_NO_INDEX.
                    std::vector<std::uint32_t> first_;

                    /// @brief Whether each buffer is shared.
                    std::vector<bool> shared_;
            };

            /**
             * @brief Returns the nodes of each region, in node order.
             * @throws Error when a node that is not a LEAF has a key the partition does not list.
             */
            [[nodiscard]] std::vector<std::vector<Graph::NodeID>> nodes_by_region(
                const Graph::Graph& graph, const Partition& partition, const RegionIndex& index)
            {
                std::vector<std::vector<Graph::NodeID>> nodes(partition.regions.size());
                for (std::size_t i = 0; i < graph.node_count(); ++i)
                {
                    const RegionKey key = partition.region_of_node[i];
                    if (key == K_NO_REGION)
                        continue;

                    const std::uint32_t region = index.of(key);
                    SA_THROW_IF(region == K_NO_INDEX,
                                "Node {} has region key {}, which the partition does not list",
                                i, key);
                    nodes[region].push_back(static_cast<Graph::NodeID>(i));
                }

                return nodes;
            }

        } // namespace

        BoundaryMap find_boundaries(const Graph::Graph& graph,
                                    const Partition& partition,
                                    SushiRuntime::span<const Graph::ValueID> observed)
        {
            SA_THROW_IF(partition.region_of_node.size() != graph.node_count(),
                        "The partition assigns {} nodes but the graph holds {}",
                        partition.region_of_node.size(), graph.node_count());

            BoundaryMap map;
            map.buffer_of = buffers_of(graph);
            map.regions.resize(partition.regions.size());

            const RegionIndex index(partition);
            const std::vector<std::vector<Graph::NodeID>> nodes =
                nodes_by_region(graph, partition, index);

            SharedBuffers shared(graph.value_count());
            for (std::size_t region = 0; region < nodes.size(); ++region)
            {
                for (Graph::NodeID id : nodes[region])
                {
                    const Graph::Node& node = graph.node(id);
                    for (Graph::ValueID operand : node.inputs)
                        shared.touch(map.buffer_of[operand], static_cast<std::uint32_t>(region));
                    for (Graph::ValueID result : node.outputs)
                        shared.touch(map.buffer_of[result], static_cast<std::uint32_t>(region));
                }
            }

            for (Graph::ValueID value : observed)
            {
                SA_THROW_IF(value >= graph.value_count(),
                            "Observed value {} does not exist; the graph holds {} values",
                            value, graph.value_count());
                shared.observe(map.buffer_of[value]);
            }

            // imported_by[v] names the last region that listed v, so one region lists it once.
            std::vector<std::uint32_t> imported_by(graph.value_count(), K_NO_INDEX);
            for (std::size_t region = 0; region < nodes.size(); ++region)
            {
                RegionBoundary& boundary = map.regions[region];
                const RegionKey key = partition.regions[region].key;

                for (std::size_t local = 0; local < nodes[region].size(); ++local)
                {
                    const Graph::Node& node = graph.node(nodes[region][local]);
                    for (Graph::ValueID operand : node.inputs)
                    {
                        const Graph::NodeID producer = graph.value(operand).producer;
                        const bool inside = producer != Graph::K_NO_NODE
                                            && partition.region_of_node[producer] == key;
                        if (inside || imported_by[operand] == region)
                            continue;

                        imported_by[operand] = static_cast<std::uint32_t>(region);
                        boundary.imports.push_back(operand);
                    }

                    for (std::size_t result = 0; result < node.outputs.size(); ++result)
                    {
                        if (!shared.holds(map.buffer_of[node.outputs[result]]))
                            continue;

                        boundary.exports.push_back(node.outputs[result]);
                        boundary.export_positions.push_back(
                            ValuePosition{static_cast<std::uint32_t>(local),
                                          static_cast<std::uint32_t>(result)});
                    }
                }
            }

            return map;
        }

    } // namespace Eager
} // namespace SushiAI

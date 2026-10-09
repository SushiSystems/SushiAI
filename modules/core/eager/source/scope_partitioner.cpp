/**************************************************************************/
/* scope_partitioner.cpp                                                  */
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
 * @file scope_partitioner.cpp
 * @brief Implements ScopePartitioner, which makes one region per segment and direction.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/scope_partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Returns whether @p node is a LEAF, which no region holds. */
            bool is_leaf(const Graph::Graph& graph, Graph::NodeId node)
            {
                return graph.node(node).op == Ops::K_LEAF.id;
            }

            /**
             * @brief Throws unless @p provenance records the backward nodes of @p graph.
             * @see modules/core/eager/README.md
             */
            void validate_provenance(const Graph::Graph& graph,
                                     const Autograd::Provenance& provenance)
            {
                const std::size_t forward_count = provenance.forward_node_count;

                SA_THROW_IF(forward_count + provenance.origin.size() != graph.node_count(),
                            "The provenance describes {} forward and {} backward nodes, but the "
                            "graph has {} nodes; pass the record differentiate() filled for this "
                            "graph",
                            forward_count, provenance.origin.size(), graph.node_count());

                for (std::size_t i = 0; i < provenance.origin.size(); ++i)
                    SA_THROW_IF(provenance.origin[i] >= forward_count,
                                "Backward node {} has origin {}, which is not one of the {} "
                                "forward nodes",
                                forward_count + i, provenance.origin[i], forward_count);
            }

            /**
             * @brief Throws unless @p segments cover nodes 0 to @p forward_count without a gap.
             * @see modules/core/eager/README.md
             */
            void validate_segments(SushiRuntime::span<const Segment> segments,
                                   std::size_t forward_count)
            {
                SA_THROW_IF(segments.size() == 0,
                            "A partition needs at least one segment; seal a ScopeTrace over a "
                            "graph that holds a node");

                std::size_t expected_begin = 0;
                for (std::size_t i = 0; i < segments.size(); ++i)
                {
                    const Segment& segment = segments[i];

                    SA_THROW_IF(segment.begin != expected_begin,
                                "Segment {} ('{}') begins at node {}, but the segments before it "
                                "end at node {}; segments must be contiguous from node 0",
                                i, segment.name, segment.begin, expected_begin);
                    SA_THROW_IF(segment.end <= segment.begin,
                                "Segment {} ('{}') spans nodes {} to {} and holds no node",
                                i, segment.name, segment.begin, segment.end);

                    expected_begin = segment.end;
                }

                SA_THROW_IF(expected_begin != forward_count,
                            "The segments cover nodes 0 to {}, but the forward graph has {} "
                            "nodes; trace the whole forward pass before differentiating",
                            expected_begin, forward_count);
            }

            /** @brief Returns the key above every region of @p segment_count segments. */
            RegionKey optimizer_key_of(RegionKey segment_count)
            {
                return 2 * segment_count;
            }

            /**
             * @brief Returns the key of the backward region of segment @p number.
             * @param number Segment number, counted from 1.
             * @see modules/core/eager/README.md
             */
            RegionKey backward_key_of_segment(RegionKey segment_count, RegionKey number)
            {
                return optimizer_key_of(segment_count) - number;
            }

            /**
             * @brief Returns, for each forward node, the number of its segment, counted from 1.
             * @pre validate_segments() accepted @p segments for @p forward_count nodes.
             */
            std::vector<RegionKey> number_forward_nodes(SushiRuntime::span<const Segment> segments,
                                                        std::size_t forward_count)
            {
                std::vector<RegionKey> segment_of_node(forward_count, K_NO_REGION);

                for (std::size_t i = 0; i < segments.size(); ++i)
                    std::fill(segment_of_node.begin() + segments[i].begin,
                              segment_of_node.begin() + segments[i].end,
                              static_cast<RegionKey>(i + 1));

                return segment_of_node;
            }

            /**
             * @brief Returns the highest key among the regions that produce @p node's operands.
             * @param region_of_node Keys of every node below @p node.
             * @return K_NO_REGION when every operand is a leaf or has no producer.
             */
            RegionKey highest_operand_key(const Graph::Graph& graph, Graph::NodeId node,
                                          const std::vector<RegionKey>& region_of_node)
            {
                RegionKey highest = K_NO_REGION;

                for (Graph::ValueId operand : graph.node(node).inputs)
                {
                    const Graph::NodeId producer = graph.value(operand).producer;
                    if (producer != Graph::K_NO_NODE)
                        highest = std::max(highest, region_of_node[producer]);
                }

                return highest;
            }

            /**
             * @brief Returns the key of backward node @p node.
             * @param segment_of_node Segment number of every forward node, from 1.
             * @param region_of_node Keys of every node below @p node.
             * @see modules/core/eager/README.md
             */
            RegionKey backward_key(const Graph::Graph& graph, Graph::NodeId node,
                                   Graph::NodeId origin,
                                   const std::vector<RegionKey>& segment_of_node,
                                   const std::vector<RegionKey>& region_of_node,
                                   RegionKey segment_count)
            {
                if (!is_leaf(graph, origin))
                    return backward_key_of_segment(segment_count, segment_of_node[origin]);

                const RegionKey key = highest_operand_key(graph, node, region_of_node);

                SA_THROW_IF(key == K_NO_REGION,
                            "Backward node {} ({}) has LEAF node {} as its origin and reads only "
                            "leaves, so no region can hold it; a gradient summed for a leaf must "
                            "read at least one value a region produces",
                            node, Ops::name_of(graph.node(node).op), origin);

                return key;
            }

            /**
             * @brief Throws when a node reads a value that a region with a higher key produces.
             * @param region_of_node Key of every node of @p graph.
             * @see modules/core/eager/README.md
             */
            void require_producers_first(const Graph::Graph& graph,
                                         const std::vector<RegionKey>& region_of_node)
            {
                for (std::size_t i = 0; i < graph.node_count(); ++i)
                {
                    const Graph::NodeId node = static_cast<Graph::NodeId>(i);

                    for (Graph::ValueId operand : graph.node(node).inputs)
                    {
                        const Graph::NodeId producer = graph.value(operand).producer;
                        if (producer == Graph::K_NO_NODE)
                            continue;

                        SA_THROW_IF(region_of_node[producer] > region_of_node[i],
                                    "Node {} ({}) has region key {} but reads a value of node {}, "
                                    "whose key is {}; the provenance does not list the backward "
                                    "nodes in the order differentiate() emits them",
                                    node, Ops::name_of(graph.node(node).op), region_of_node[i],
                                    producer, region_of_node[producer]);
                    }
                }
            }

            /** @brief Returns the entry of region @p key, which holds a pass of @p segment. */
            RegionInfo describe_region(RegionKey key, const Segment& segment, Direction direction)
            {
                return RegionInfo{key, segment.name + "#" + std::to_string(segment.occurrence),
                                  direction};
            }

            /**
             * @brief Lists the regions that hold a node, in ascending key order.
             * @param region_of_node Key of every node; K_NO_REGION for a leaf.
             * @see modules/core/eager/README.md
             */
            std::vector<RegionInfo> list_regions(const std::vector<RegionKey>& region_of_node,
                                                 SushiRuntime::span<const Segment> segments)
            {
                const RegionKey segment_count = segments.size();

                std::vector<bool> holds_a_node(
                    static_cast<std::size_t>(optimizer_key_of(segment_count)), false);
                for (RegionKey key : region_of_node)
                    holds_a_node[static_cast<std::size_t>(key)] = true;

                std::vector<RegionInfo> regions;
                for (RegionKey number = 1; number <= segment_count; ++number)
                {
                    if (holds_a_node[static_cast<std::size_t>(number)])
                        regions.push_back(
                            describe_region(number, segments[static_cast<std::size_t>(number - 1)],
                                            Direction::Forward));
                }

                // The last segment's backward nodes share its forward key, listed above.
                for (RegionKey number = segment_count - 1; number >= 1; --number)
                {
                    const RegionKey key = backward_key_of_segment(segment_count, number);
                    if (holds_a_node[static_cast<std::size_t>(key)])
                        regions.push_back(
                            describe_region(key, segments[static_cast<std::size_t>(number - 1)],
                                            Direction::Backward));
                }

                return regions;
            }

        } // namespace

        Partition ScopePartitioner::assign(const Graph::Graph& graph,
                                           SushiRuntime::span<const Segment> segments,
                                           const Autograd::Provenance& provenance) const
        {
            validate_provenance(graph, provenance);
            validate_segments(segments, provenance.forward_node_count);

            const std::size_t forward_count = provenance.forward_node_count;
            const RegionKey segment_count = segments.size();
            const std::vector<RegionKey> segment_of_node =
                number_forward_nodes(segments, forward_count);

            Partition partition;
            partition.optimizer_key = optimizer_key_of(segment_count);
            partition.region_of_node.assign(graph.node_count(), K_NO_REGION);

            // Node order is topological, so every operand's producer already has its key.
            for (std::size_t i = 0; i < graph.node_count(); ++i)
            {
                const Graph::NodeId node = static_cast<Graph::NodeId>(i);
                if (is_leaf(graph, node))
                    continue;

                partition.region_of_node[i] =
                    i < forward_count
                        ? segment_of_node[i]
                        : backward_key(graph, node, provenance.origin[i - forward_count],
                                       segment_of_node, partition.region_of_node, segment_count);
            }

            require_producers_first(graph, partition.region_of_node);

            partition.regions = list_regions(partition.region_of_node, segments);
            return partition;
        }

    } // namespace Eager
} // namespace SushiAI

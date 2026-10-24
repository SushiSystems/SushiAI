/**************************************************************************/
/* region_signature.cpp                                                   */
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
 * @file region_signature.cpp
 * @brief Implements signature_of, which hashes the structure of one region.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_signature.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <unordered_map>

#include <SushiAI/failure/error.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Tags the kind of operand position that follows in the hashed stream. */
            enum class OperandKind : std::uint8_t
            {
                Inside = 1,   /**< Produced by a node of the region being hashed. */
                External = 2, /**< Read by the region being hashed and produced outside it. */
            };

            /** @brief Accumulates a 64-bit FNV-1a hash over the values fed to it. */
            class FNV1a final
            {
                public:
                    /** @brief Feeds the eight bytes of @p word, lowest byte first. */
                    void add_word(std::uint64_t word) noexcept
                    {
                        state_ = Hash::fnv1a_word(state_, word);
                    }

                    /** @brief Feeds the bit pattern of @p real. */
                    void add_real(double real) noexcept
                    {
                        std::uint64_t bits = 0;
                        std::memcpy(&bits, &real, sizeof(bits));
                        add_word(bits);
                    }

                    /** @brief Returns the hash of everything fed so far. */
                    [[nodiscard]] std::uint64_t digest() const noexcept
                    {
                        return state_;
                    }

                private:
                    /// @brief The running hash.
                    std::uint64_t state_ = Hash::K_FNV1A_OFFSET_BASIS;
            };

            /** @brief Records the position of each node of one region among that region's nodes. */
            class RegionLayout final
            {
                public:
                    /**
                     * @brief Gives @p node the next position of the region.
                     * @pre The nodes of the region are appended in node order, each once.
                     */
                    void append(Graph::NodeID node)
                    {
                        position_.emplace(node, static_cast<std::uint32_t>(position_.size()));
                    }

                    /** @brief Returns the position of @p node, or null when it was not appended. */
                    [[nodiscard]] const std::uint32_t* find(Graph::NodeID node) const
                    {
                        const auto it = position_.find(node);
                        return it != position_.end() ? &it->second : nullptr;
                    }

                private:
                    /// @brief Position of each appended node among the region's nodes, by node id.
                    std::unordered_map<Graph::NodeID, std::uint32_t> position_;
            };

            /** @brief Reports whether @p partition lists region @p key. */
            [[nodiscard]] bool is_listed(const Partition& partition, RegionKey key) noexcept
            {
                const std::vector<RegionInfo>& regions = partition.regions;

                // Partition::regions is in ascending key order.
                const auto it = std::lower_bound(
                    regions.begin(), regions.end(), key,
                    [](const RegionInfo& info, RegionKey wanted)
                    {
                        return info.key < wanted;
                    });
                return it != regions.end() && it->key == key;
            }

            /** @brief Returns the index of @p value among the results of @p producer. */
            [[nodiscard]] std::size_t result_index(const Graph::Node& producer,
                                                   Graph::ValueID value) noexcept
            {
                const auto it = std::find(producer.outputs.begin(), producer.outputs.end(), value);
                return static_cast<std::size_t>(it - producer.outputs.begin());
            }

            /** @brief Feeds the operation, the attributes and every result of @p node. */
            void hash_node(FNV1a& hasher, const Graph::Graph& graph, const Graph::Node& node)
            {
                hasher.add_word(node.op.value);

                for (double attribute : node.attributes.values)
                    hasher.add_real(attribute);

                hasher.add_word(node.outputs.size());
                for (Graph::ValueID output : node.outputs)
                {
                    const Graph::Value& result = graph.value(output);
                    const std::int32_t rank = result.shape.rank();

                    hasher.add_word(static_cast<std::uint64_t>(rank));
                    for (std::int32_t axis = 0; axis < rank; ++axis)
                        hasher.add_word(static_cast<std::uint64_t>(result.shape[axis]));

                    hasher.add_word(static_cast<std::uint64_t>(result.dtype));
                    hasher.add_word(static_cast<std::uint64_t>(result.layout));
                }
            }

            /**
             * @brief Feeds the position of @p operand as a node of the signed region reads it.
             * @param layout The nodes of the signed region that precede the reader.
             * @param external_identity One identity per value id; read when no node in @p layout
             *        writes @p operand, or when a LEAF does.
             * @see modules/core/eager/README.md
             */
            void hash_operand(FNV1a& hasher,
                              const Graph::Graph& graph,
                              const RegionLayout& layout,
                              SushiRuntime::span<const std::uint64_t> external_identity,
                              Graph::ValueID operand)
            {
                const Graph::NodeID producer_id = graph.value(operand).producer;
                const bool computed = producer_id != Graph::K_NO_NODE
                                      && graph.node(producer_id).op != Ops::K_LEAF.id;
                const std::uint32_t* position = computed ? layout.find(producer_id) : nullptr;

                if (position != nullptr)
                {
                    hasher.add_word(static_cast<std::uint64_t>(OperandKind::Inside));
                    hasher.add_word(*position);
                    hasher.add_word(result_index(graph.node(producer_id), operand));
                    return;
                }

                SA_THROW_IF(operand >= external_identity.size(),
                            "External value {} has no identity; the caller supplied {} identities",
                            operand, external_identity.size());
                hasher.add_word(static_cast<std::uint64_t>(OperandKind::External));
                hasher.add_word(external_identity[operand]);
            }

        } // namespace

        RegionSignature signature_of(const Graph::Graph& graph,
                                     const Partition& partition,
                                     RegionKey region,
                                     SushiRuntime::span<const std::uint64_t> external_identity)
        {
            SA_THROW_IF(partition.region_of_node.size() != graph.node_count(),
                        "The partition assigns {} nodes but the graph holds {}",
                        partition.region_of_node.size(), graph.node_count());
            SA_THROW_IF(!is_listed(partition, region),
                        "Region {} is not listed in the partition", region);

            RegionSignature signature;
            RegionLayout layout;
            FNV1a hasher;

            const std::vector<Graph::Node>& nodes = graph.nodes();
            for (std::size_t id = 0; id < nodes.size(); ++id)
            {
                if (partition.region_of_node[id] != region)
                    continue;

                const Graph::Node& node = nodes[id];
                signature.ops.push_back(node.op);
                hash_node(hasher, graph, node);

                hasher.add_word(node.inputs.size());
                for (Graph::ValueID operand : node.inputs)
                    hash_operand(hasher, graph, layout, external_identity, operand);

                // A node reads only results of earlier nodes, so it is recorded after its operands.
                layout.append(static_cast<Graph::NodeID>(id));
            }

            signature.hash = hasher.digest();
            return signature;
        }

    } // namespace Eager
} // namespace SushiAI

/**************************************************************************/
/* memory_plan.cpp                                                        */
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
 * @file memory_plan.cpp
 * @brief Implements MemoryPlan allocation layout and transitive reachability analysis.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief @p value rounded up to the next multiple of @p multiple. */
            std::size_t round_up(std::size_t value, std::size_t multiple) noexcept
            {
                const std::size_t remainder = value % multiple;
                return remainder == 0 ? value : value + (multiple - remainder);
            }

            /** @brief How many bytes one value's elements occupy. */
            std::size_t byte_size(const Value& value) noexcept
            {
                return static_cast<std::size_t>(value.shape.num_elements())
                     * dtype_size(value.dtype);
            }

            /**
             * @class Aliases
             * @brief Disjoint sets over ValueIDs for tracking shared buffer aliases.
             *
             * Resolves alias chains like ACCUMULATE to a single representative value.
             */
            class Aliases
            {
                public:
                    /** @brief Every value in its own set. */
                    explicit Aliases(std::size_t count) : parent_(count)
                    {
                        for (std::size_t i = 0; i < count; ++i)
                            parent_[i] = static_cast<ValueID>(i);
                    }

                    /** @brief The representative of @p id's set, with path halving. */
                    ValueID find(ValueID id) noexcept
                    {
                        while (parent_[id] != id)
                        {
                            parent_[id] = parent_[parent_[id]];
                            id = parent_[id];
                        }
                        return id;
                    }

                    /** @brief Merges @p a's and @p b's sets. */
                    void merge(ValueID a, ValueID b) noexcept
                    {
                        const ValueID ra = find(a);
                        const ValueID rb = find(b);
                        if (ra == rb) return;
                        if (ra < rb) parent_[rb] = ra;
                        else         parent_[ra] = rb;
                    }

                private:
                    std::vector<ValueID> parent_;
            };

            /**
             * @class Reachability
             * @brief Transitive closure bitset representing node execution precedence.
             *
             * Computed in a single reverse sweep over topological insertion order.
             */
            class Reachability
            {
                public:
                    /**
                     * @brief Closes @p graph.
                     * @param graph The graph to close over.
                     * @param consumers Which nodes read each value.
                     */
                    Reachability(const Graph& graph,
                                 const std::vector<std::vector<NodeID>>& consumers)
                    {
                        const std::size_t count = graph.node_count();
                        words_ = (count + 63) / 64;
                        bits_.assign(count * words_, 0);

                        for (std::size_t index = count; index-- > 0; )
                        {
                            std::uint64_t* row = bits_.data() + index * words_;

                            for (ValueID out : graph.node(static_cast<NodeID>(index)).outputs)
                                for (NodeID consumer : consumers[out])
                                {
                                    const std::uint64_t* successor =
                                        bits_.data() + static_cast<std::size_t>(consumer) * words_;

                                    for (std::size_t word = 0; word < words_; ++word)
                                        row[word] |= successor[word];

                                    row[consumer / 64] |= std::uint64_t{1} << (consumer % 64);
                                }
                        }
                    }

                    /**
                     * @brief Checks whether a directed path exists between nodes.
                     *
                     * @param from Upstream source node.
                     * @param to Downstream destination node.
                     * @return True if to is strictly downstream of from.
                     */
                    bool reaches(NodeID from, NodeID to) const noexcept
                    {
                        const std::uint64_t* row =
                            bits_.data() + static_cast<std::size_t>(from) * words_;
                        return ((row[to / 64] >> (to % 64)) & std::uint64_t{1}) != 0;
                    }

                private:
                    std::size_t words_ = 0;
                    std::vector<std::uint64_t> bits_;
            };

            /** @brief Marks a buffer that serves no scratch request. */
            constexpr std::size_t K_NO_REQUEST = ~std::size_t{0};

            /**
             * @struct Buffer
             * @brief One run of bytes the plan has to find room for.
             *
             * A buffer is a whole alias class, not a single value: see @c Aliases.
             */
            struct Buffer
            {
                /** @brief The alias class's representative value. */
                ValueID owner = K_NO_VALUE;

                /** @brief Which arena class it belongs to. */
                Lifetime lifetime = Lifetime::Activation;

                /** @brief The largest member's byte count. */
                std::size_t bytes = 0;

                /** @brief The granularity its start must be rounded to. */
                std::size_t alignment = 1;

                /** @brief The earliest node that writes it, or NO_NODE. */
                NodeID first_write = K_NO_NODE;

                /** @brief Every node that writes or reads any member. */
                std::vector<NodeID> uses;

                /** @brief Every value that names these bytes. */
                std::vector<ValueID> members;

                /** @brief The index of the scratch request these bytes serve, or K_NO_REQUEST. */
                std::size_t request = K_NO_REQUEST;

                /**
                 * @brief Whether the planner may overlap this buffer with others.
                 *
                 * False for persistent classes or buffers unwritten by the graph.
                 */
                bool packable = false;
            };

            /**
             * @brief Checks if all uses of earlier precede first write of later.
             *
             * @param earlier Buffer whose storage is candidate for reuse.
             * @param later Candidate consumer buffer.
             * @param reach Transitive reachability closure.
             * @return True if reuse introduces no anti-dependencies.
             */
            bool precedes(const Buffer& earlier, const Buffer& later, const Reachability& reach)
            {
                if (later.first_write == K_NO_NODE) return false;

                for (NodeID use : earlier.uses)
                    if (!reach.reaches(use, later.first_write)) return false;

                return true;
            }

            /** @brief Two buffers conflict when neither is finished before the other starts. */
            bool conflicts(const Buffer& a, const Buffer& b, const Reachability& reach)
            {
                return !precedes(a, b, reach) && !precedes(b, a, reach);
            }

            /** @brief Whether the buffers of @p lifetime are overlapped by the order of the nodes. */
            constexpr bool follows_node_order(Lifetime lifetime) noexcept
            {
                return lifetime == Lifetime::Scratch;
            }

            /** @brief The last node, in the order of the graph, that writes or reads @p buffer. */
            NodeID last_use(const Buffer& buffer) noexcept
            {
                return *std::max_element(buffer.uses.begin(), buffer.uses.end());
            }

            /**
             * @brief Two buffers conflict in node order when neither's last use comes before
             *        the other's first write.
             */
            bool conflicts_in_node_order(const Buffer& a, const Buffer& b) noexcept
            {
                return !(last_use(a) < b.first_write) && !(last_use(b) < a.first_write);
            }

            /** @brief One buffer already given an offset in the arena being filled. */
            struct Slot
            {
                std::size_t offset = 0;
                std::size_t bytes = 0;
                std::size_t buffer = 0; /**< Index into the buffer table. */
            };
        }

        const Placement& MemoryPlan::placement(ValueID id) const
        {
            SA_THROW_IF(id >= placements_.size(),
                        "Value {} has no placement; this plan covers {} values",
                        id, placements_.size());
            return placements_[id];
        }

        const Placement* MemoryPlan::scratch_placement(NodeID node, int slot) const noexcept
        {
            for (const ScratchSlot& held : scratch_)
                if (held.node == node && held.slot == slot) return &held.placement;

            return nullptr;
        }

        std::size_t MemoryPlan::arena_bytes() const noexcept
        {
            std::size_t total = 0;
            for (const Arena& arena : arenas_) total += arena.bytes;
            return total;
        }

        std::size_t MemoryPlan::unpacked_bytes() const noexcept
        {
            std::size_t total = 0;
            for (const Arena& arena : arenas_) total += arena.unpacked_bytes;
            return total;
        }

        MemoryPlan MemoryPlan::build(const Graph& graph, const PlanOptions& options,
                                     SushiRuntime::span<const ScratchRequest> scratch)
        {
            SA_THROW_IF(options.alignment == 0,
                        "PlanOptions::alignment must be positive; an offset cannot be rounded "
                        "to a multiple of zero");

            MemoryPlan plan;

            const std::size_t value_count = graph.value_count();
            const std::size_t node_count = graph.node_count();

            plan.placements_.assign(value_count, Placement{});
            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
            {
                plan.placements_[id].bytes = byte_size(graph.value(id));
                plan.placements_[id].buffer = id;
            }

            plan.reuse_enabled_ = options.reuse;
            if (!options.reuse)
                plan.reuse_note_ = "live-range overlap was switched off by PlanOptions::reuse; "
                                   "every buffer has its own slot";

            if (value_count == 0) return plan;

            // Verifies topological ordering to ensure downstream reachability closure validity.
            std::vector<std::vector<NodeID>> consumers(value_count);

            for (NodeID id = 0; id < static_cast<NodeID>(node_count); ++id)
                for (ValueID in : graph.node(id).inputs)
                {
                    const NodeID producer = graph.value(in).producer;
                    SA_THROW_IF(producer != K_NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The memory "
                                "planner needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                    consumers[in].push_back(id);
                }

            // Groups values that share a common backing buffer.
            Aliases aliases(value_count);
            for (const Node& node : graph.nodes())
                if (node.op == Ops::K_ACCUMULATE.id && !node.inputs.empty() && !node.outputs.empty())
                    aliases.merge(node.outputs[0], node.inputs[0]);

            std::vector<Buffer> buffers;
            std::unordered_map<ValueID, std::size_t> buffer_of_owner;
            buffers.reserve(value_count);

            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
            {
                const Value& value = graph.value(id);
                const ValueID owner = aliases.find(id);

                const auto [entry, inserted] = buffer_of_owner.emplace(owner, buffers.size());
                if (inserted)
                {
                    Buffer fresh;
                    fresh.owner = owner;
                    fresh.lifetime = value.lifetime;
                    fresh.packable = true;
                    buffers.push_back(std::move(fresh));
                }

                Buffer& buffer = buffers[entry->second];
                buffer.members.push_back(id);
                buffer.bytes = std::max(buffer.bytes, byte_size(value));
                buffer.alignment = std::max(buffer.alignment,
                                            std::lcm(options.alignment, dtype_size(value.dtype)));

                if (id == owner) buffer.lifetime = value.lifetime;
                if (is_persistent(value.lifetime)) buffer.packable = false;

                if (value.producer == K_NO_NODE)
                {
                    // External inputs persist across step replays and are marked unpackable.
                    buffer.packable = false;
                }
                else
                {
                    buffer.first_write = std::min(buffer.first_write, value.producer);
                    buffer.uses.push_back(value.producer);
                }

                for (NodeID consumer : consumers[id]) buffer.uses.push_back(consumer);
            }

            // A requested scratch buffer has one user, its node, and names no value.
            plan.scratch_.reserve(scratch.size());
            for (std::size_t index = 0; index < scratch.size(); ++index)
            {
                const ScratchRequest& request = scratch[index];
                SA_THROW_IF(request.node >= node_count,
                            "A scratch request names node {}, which this graph does not have",
                            request.node);

                Buffer fresh;
                fresh.lifetime = Lifetime::Scratch;
                fresh.bytes = static_cast<std::size_t>(request.shape.num_elements())
                            * dtype_size(request.dtype);
                fresh.alignment = std::lcm(options.alignment, dtype_size(request.dtype));
                fresh.first_write = request.node;
                fresh.uses.push_back(request.node);
                fresh.packable = true;
                fresh.request = index;
                buffers.push_back(std::move(fresh));

                Placement placement;
                placement.bytes = buffers.back().bytes;
                plan.scratch_.push_back(ScratchSlot{request.node, request.slot, placement});
            }

            // Caller-pinned buffers bypass packing regardless of lifetime class.
            for (ValueID pinned : options.pinned)
            {
                SA_THROW_IF(pinned >= value_count,
                            "PlanOptions::pinned names value {}, which this graph does not have",
                            pinned);
                buffers[buffer_of_owner.at(aliases.find(pinned))].packable = false;
            }

            // Evaluates reachability matrix when node limit allows reuse.
            const bool overlap = options.reuse && node_count <= options.reuse_node_limit;
            if (options.reuse && !overlap)
            {
                plan.reuse_enabled_ = false;
                plan.reuse_note_ =
                    "the graph has more nodes than PlanOptions::reuse_node_limit, and the "
                    "reachability closure this planner needs is quadratic in the node count. "
                    "Every buffer was given its own slot rather than overlapping live ranges "
                    "against a linear schedule the runtime is free not to follow";
            }

            // Computes reachability closure only when reuse is active.
            std::optional<Reachability> reach;
            if (overlap) reach.emplace(graph, consumers);

            // Allocates arenas per packable lifetime class in fixed enumeration order.
            constexpr Lifetime K_PACKABLE[] = {Lifetime::Activation, Lifetime::Scratch};

            for (Lifetime lifetime : K_PACKABLE)
            {
                std::vector<std::size_t> candidates;
                for (std::size_t index = 0; index < buffers.size(); ++index)
                    if (buffers[index].packable && buffers[index].lifetime == lifetime)
                        candidates.push_back(index);

                if (candidates.empty()) continue;

                // Sorts greedy-by-size descending and by owner for determinism.
                std::sort(candidates.begin(), candidates.end(),
                          [&buffers](std::size_t a, std::size_t b)
                          {
                              if (buffers[a].bytes != buffers[b].bytes)
                                  return buffers[a].bytes > buffers[b].bytes;
                              if (buffers[a].owner != buffers[b].owner)
                                  return buffers[a].owner < buffers[b].owner;
                              return a < b;
                          });

                const ArenaID arena_id = static_cast<ArenaID>(plan.arenas_.size());
                std::vector<Slot> placed;
                placed.reserve(candidates.size());

                Arena arena;
                arena.lifetime = lifetime;

                const bool in_node_order = options.reuse && follows_node_order(lifetime);
                const auto clashes = [&](const Buffer& a, const Buffer& b)
                {
                    if (in_node_order) return conflicts_in_node_order(a, b);
                    return !overlap || conflicts(a, b, *reach);
                };

                for (std::size_t index : candidates)
                {
                    const Buffer& buffer = buffers[index];

                    // Finds the first aligned non-conflicting slot in address order.
                    std::size_t offset = 0;
                    for (const Slot& slot : placed)
                    {
                        if (clashes(buffer, buffers[slot.buffer]))
                        {
                            offset = round_up(offset, buffer.alignment);
                            if (offset + buffer.bytes <= slot.offset) break;
                            offset = std::max(offset, slot.offset + slot.bytes);
                        }
                    }
                    offset = round_up(offset, buffer.alignment);

                    const Slot slot{offset, buffer.bytes, index};
                    placed.insert(std::upper_bound(placed.begin(), placed.end(), slot,
                                                   [](const Slot& a, const Slot& b)
                                                   { return a.offset < b.offset; }),
                                  slot);

                    arena.bytes = std::max(arena.bytes, offset + buffer.bytes);
                    arena.unpacked_bytes += buffer.bytes;
                    ++arena.buffers;

                    for (ValueID member : buffer.members)
                    {
                        plan.placements_[member].arena = arena_id;
                        plan.placements_[member].offset = offset;
                        plan.placements_[member].buffer = buffer.owner;
                    }

                    if (buffer.request != K_NO_REQUEST)
                    {
                        plan.scratch_[buffer.request].placement.arena = arena_id;
                        plan.scratch_[buffer.request].placement.offset = offset;
                    }
                }

                arena.bytes = round_up(arena.bytes, options.alignment);
                plan.arenas_.push_back(arena);
            }

            // Unpacked buffers retain dedicated allocations while recording owner aliases.
            for (const Buffer& buffer : buffers)
                if (!buffer.packable)
                    for (ValueID member : buffer.members)
                        plan.placements_[member].buffer = buffer.owner;

            return plan;
        }

    } // namespace Graph
} // namespace SushiAI

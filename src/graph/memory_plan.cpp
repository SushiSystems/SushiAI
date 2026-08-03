/**************************************************************************/
/* memory_plan.cpp                                                        */
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
             * @brief Disjoint sets over ValueIds, for values that share a buffer.
             *
             * @c ACCUMULATE's result *is* its accumulator's storage, so the two
             * values are two names for one buffer. Planning them separately
             * would be silently wrong rather than merely wasteful: the planner
             * would think the accumulator died at the @c ACCUMULATE node and
             * hand its bytes to something else while the result is still being
             * read under the other name.
             *
             * The lower id always wins the union, so a chain's representative is
             * the value declared first - which, because insertion order is
             * topological, is also the one whose producer writes the buffer
             * first.
             */
            class Aliases
            {
                public:
                    /** @brief Every value in its own set. */
                    explicit Aliases(std::size_t count) : parent_(count)
                    {
                        for (std::size_t i = 0; i < count; ++i)
                            parent_[i] = static_cast<ValueId>(i);
                    }

                    /** @brief The representative of @p id's set, with path halving. */
                    ValueId find(ValueId id) noexcept
                    {
                        while (parent_[id] != id)
                        {
                            parent_[id] = parent_[parent_[id]];
                            id = parent_[id];
                        }
                        return id;
                    }

                    /** @brief Merges @p a's and @p b's sets. */
                    void merge(ValueId a, ValueId b) noexcept
                    {
                        const ValueId ra = find(a);
                        const ValueId rb = find(b);
                        if (ra == rb) return;
                        if (ra < rb) parent_[rb] = ra;
                        else         parent_[ra] = rb;
                    }

                private:
                    std::vector<ValueId> parent_;
            };

            /**
             * @class Reachability
             * @brief "Can node A's work be finished before node B's begins?"
             *
             * The transitive closure of the dataflow DAG, one bitset row per
             * node. This is the whole reason the planner is not an interval
             * overlap test: two nodes with no path between them are unordered,
             * the runtime may run them concurrently, and bytes must not be
             * shared across that boundary - see the header.
             *
             * Built in one reverse sweep because a graph's insertion order is a
             * valid topological order (Graph::nodes()), so a node's successors
             * are all complete by the time it is visited. O(V*E/64) time,
             * O(V^2/64) memory.
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
                                 const std::vector<std::vector<NodeId>>& consumers)
                    {
                        const std::size_t count = graph.node_count();
                        words_ = (count + 63) / 64;
                        bits_.assign(count * words_, 0);

                        for (std::size_t index = count; index-- > 0; )
                        {
                            std::uint64_t* row = bits_.data() + index * words_;

                            for (ValueId out : graph.node(static_cast<NodeId>(index)).outputs)
                                for (NodeId consumer : consumers[out])
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
                     * @brief Whether @p to is strictly downstream of @p from.
                     *
                     * Strict by construction: a DAG has no path from a node to
                     * itself, so a node's own bit is never set in its own row.
                     *
                     * @param from The upstream node.
                     * @param to The node to reach.
                     * @return True when some directed path runs from @p from to @p to.
                     */
                    bool reaches(NodeId from, NodeId to) const noexcept
                    {
                        const std::uint64_t* row =
                            bits_.data() + static_cast<std::size_t>(from) * words_;
                        return ((row[to / 64] >> (to % 64)) & std::uint64_t{1}) != 0;
                    }

                private:
                    std::size_t words_ = 0;
                    std::vector<std::uint64_t> bits_;
            };

            /**
             * @struct Buffer
             * @brief One run of bytes the plan has to find room for.
             *
             * A buffer is a whole alias class, not a single value: see @c Aliases.
             */
            struct Buffer
            {
                /** @brief The alias class's representative value. */
                ValueId owner = NO_VALUE;

                /** @brief Which arena class it belongs to. */
                Lifetime lifetime = Lifetime::ACTIVATION;

                /** @brief The largest member's byte count. */
                std::size_t bytes = 0;

                /** @brief The granularity its start must be rounded to. */
                std::size_t alignment = 1;

                /** @brief The earliest node that writes it, or NO_NODE. */
                NodeId first_write = NO_NODE;

                /** @brief Every node that writes or reads any member. */
                std::vector<NodeId> uses;

                /** @brief Every value that names these bytes. */
                std::vector<ValueId> members;

                /**
                 * @brief Whether the planner may overlap it with anything.
                 *
                 * False for a persistent class, and false for a buffer nothing
                 * in the graph writes. The second case matters as much as the
                 * first: a buffer with no producer is written from outside the
                 * step and is *not* rewritten on the next replay, so handing its
                 * bytes away survives exactly one step and then trains on
                 * whatever overwrote it.
                 */
                bool packable = false;
            };

            /**
             * @brief Whether @p earlier's bytes may be handed to @p later.
             *
             * The contract of the whole pass, in one predicate: every use of
             * @p earlier must be a strict ancestor of the node that first
             * writes @p later. Not "the last use" - checking all of them is
             * equivalent (precedence is transitive, so if the maximal uses
             * precede the write, so do the rest) and needs no separate pass to
             * find which uses are maximal.
             *
             * @param earlier The buffer whose bytes would be reused.
             * @param later The buffer that would receive them.
             * @param reach The graph's transitive closure.
             * @return True when the reuse adds no ordering that was not already implied.
             */
            bool precedes(const Buffer& earlier, const Buffer& later, const Reachability& reach)
            {
                if (later.first_write == NO_NODE) return false;

                for (NodeId use : earlier.uses)
                    if (!reach.reaches(use, later.first_write)) return false;

                return true;
            }

            /** @brief Two buffers conflict when neither is finished before the other starts. */
            bool conflicts(const Buffer& a, const Buffer& b, const Reachability& reach)
            {
                return !precedes(a, b, reach) && !precedes(b, a, reach);
            }

            /** @brief One buffer already given an offset in the arena being filled. */
            struct Slot
            {
                std::size_t offset = 0;
                std::size_t bytes = 0;
                std::size_t buffer = 0; /**< Index into the buffer table. */
            };
        }

        const Placement& MemoryPlan::placement(ValueId id) const
        {
            SA_THROW_IF(id >= placements_.size(),
                        "Value {} has no placement; this plan covers {} values",
                        id, placements_.size());
            return placements_[id];
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

        MemoryPlan MemoryPlan::build(const Graph& graph, const PlanOptions& options)
        {
            SA_THROW_IF(options.alignment == 0,
                        "PlanOptions::alignment must be positive; an offset cannot be rounded "
                        "to a multiple of zero");

            MemoryPlan plan;

            const std::size_t value_count = graph.value_count();
            const std::size_t node_count = graph.node_count();

            plan.placements_.assign(value_count, Placement{});
            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
            {
                plan.placements_[id].bytes = byte_size(graph.value(id));
                plan.placements_[id].buffer = id;
            }

            plan.reuse_enabled_ = options.reuse;
            if (!options.reuse)
                plan.reuse_note_ = "live-range overlap was switched off by PlanOptions::reuse; "
                                   "every buffer has its own slot";

            if (value_count == 0) return plan;

            // --- Who reads what, and is the graph actually topological? -------
            // The whole pass rests on "insertion order is a valid topological
            // order", which Graph documents and add_node makes true for anything
            // built through GraphBuilder. A hand-assembled graph can still break
            // it, and a broken one would produce a closure that silently misses
            // edges, so it is checked rather than assumed.
            std::vector<std::vector<NodeId>> consumers(value_count);

            for (NodeId id = 0; id < static_cast<NodeId>(node_count); ++id)
                for (ValueId in : graph.node(id).inputs)
                {
                    const NodeId producer = graph.value(in).producer;
                    SA_THROW_IF(producer != NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The memory "
                                "planner needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                    consumers[in].push_back(id);
                }

            // --- Group values that share one buffer ---------------------------
            Aliases aliases(value_count);
            for (const Node& node : graph.nodes())
                if (node.op == Ops::ACCUMULATE.id && !node.inputs.empty() && !node.outputs.empty())
                    aliases.merge(node.outputs[0], node.inputs[0]);

            std::vector<Buffer> buffers;
            std::unordered_map<ValueId, std::size_t> buffer_of_owner;
            buffers.reserve(value_count);

            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
            {
                const Value& value = graph.value(id);
                const ValueId owner = aliases.find(id);

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

                if (value.producer == NO_NODE)
                {
                    // Written from outside the step, so never rewritten on a
                    // replay. Its bytes belong to it for the whole run.
                    buffer.packable = false;
                }
                else
                {
                    buffer.first_write = std::min(buffer.first_write, value.producer);
                    buffer.uses.push_back(value.producer);
                }

                for (NodeId consumer : consumers[id]) buffer.uses.push_back(consumer);
            }

            // A caller's veto beats every other consideration, including a
            // lifetime that says the value is packable.
            for (ValueId pinned : options.pinned)
            {
                SA_THROW_IF(pinned >= value_count,
                            "PlanOptions::pinned names value {}, which this graph does not have",
                            pinned);
                buffers[buffer_of_owner.at(aliases.find(pinned))].packable = false;
            }

            // --- Reachability, or an honest refusal to compute it -------------
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

            // Built only when it will be used: the refusal above exists
            // precisely because this object is the expensive part.
            std::optional<Reachability> reach;
            if (overlap) reach.emplace(graph, consumers);

            // --- One arena per non-persistent lifetime class ------------------
            // Enum order, so the arena ids a graph produces do not depend on the
            // order its values happened to be declared in.
            constexpr Lifetime PACKABLE[] = {Lifetime::ACTIVATION, Lifetime::SCRATCH};

            for (Lifetime lifetime : PACKABLE)
            {
                std::vector<std::size_t> candidates;
                for (std::size_t index = 0; index < buffers.size(); ++index)
                    if (buffers[index].packable && buffers[index].lifetime == lifetime)
                        candidates.push_back(index);

                if (candidates.empty()) continue;

                // Largest first, then by owner: the classic greedy-by-size
                // packing, made deterministic so two runs of the same graph
                // produce byte-identical offsets.
                std::sort(candidates.begin(), candidates.end(),
                          [&buffers](std::size_t a, std::size_t b)
                          {
                              if (buffers[a].bytes != buffers[b].bytes)
                                  return buffers[a].bytes > buffers[b].bytes;
                              return buffers[a].owner < buffers[b].owner;
                          });

                const ArenaId arena_id = static_cast<ArenaId>(plan.arenas_.size());
                std::vector<Slot> placed;
                placed.reserve(candidates.size());

                Arena arena;
                arena.lifetime = lifetime;

                for (std::size_t index : candidates)
                {
                    const Buffer& buffer = buffers[index];

                    // Walk the already-placed slots in address order and slide
                    // past every one that is live at the same time as this
                    // buffer. The first aligned gap that fits wins.
                    std::size_t offset = 0;
                    for (const Slot& slot : placed)
                    {
                        if (!overlap || conflicts(buffer, buffers[slot.buffer], *reach))
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

                    for (ValueId member : buffer.members)
                    {
                        plan.placements_[member].arena = arena_id;
                        plan.placements_[member].offset = offset;
                        plan.placements_[member].buffer = buffer.owner;
                    }
                }

                arena.bytes = round_up(arena.bytes, options.alignment);
                plan.arenas_.push_back(arena);
            }

            // Everything else keeps its own allocation, but still records which
            // value owns the bytes so an alias chain resolves to one buffer.
            for (const Buffer& buffer : buffers)
                if (!buffer.packable)
                    for (ValueId member : buffer.members)
                        plan.placements_[member].buffer = buffer.owner;

            return plan;
        }

    } // namespace Graph
} // namespace SushiAI

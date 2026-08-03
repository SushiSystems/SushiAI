/**************************************************************************/
/* memory_plan.hpp                                                        */
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

#pragma once

/**
 * @file memory_plan.hpp
 * @brief Where every non-persistent value's bytes live, decided once, before
 *        anything is lowered.
 *
 * A pure function of the IR. No engine, no allocator, no SYCL - the planner
 * reads a @c Graph and returns a table of (arena, byte offset) assignments,
 * which is exactly why the IR is a host-lane structure and why the whole pass
 * is unit-testable over synthetic graphs with no device present.
 *
 * @section arenas One arena per lifetime class, never one global arena
 *
 * Two independent reasons, and both of them produce code that is *correct and
 * quietly slow* rather than code that fails, which is why they are written
 * down instead of rediscovered:
 *
 * 1. SushiRuntime's dependency tracker shards its access history on the
 *    allocation's base pointer. One arena for everything collapses every shard
 *    into a single chain and makes hazard resolution quadratic in the number
 *    of tensors.
 * 2. Memory reuse *injects* WAR and WAW anti-dependencies. Bytes shared across
 *    two branches that could have run in parallel serialise those branches -
 *    still correct, measurably slower, and with nothing anywhere reporting it.
 *
 * Only the non-persistent classes are packed (see @c Graph::is_persistent).
 * Parameters, gradients, optimizer state and inputs live for the whole run, so
 * packing them would save nothing at all while merging the tracker shards of
 * precisely the tensors that are touched most - by the compiled step and by
 * the optimizer, which runs outside it.
 *
 * @section liveness Liveness is a reachability partial order, not an interval
 *
 * Value @c b1 may hand its bytes to @c b2 only when **every** use of @c b1 is
 * a strict ancestor of the node that first writes @c b2. That is a partial
 * order over the DAG, not an interval overlap test on a linearised schedule.
 *
 * The linear version - sort nodes into one order, overlap disjoint
 * @c [first_use, last_use] intervals - passes every test one would think to
 * write and quietly halves the achievable parallelism, because it assumes an
 * execution order the runtime is free not to take. The reachability version
 * has the property that makes reuse free: if every use of @c b1 already
 * precedes @c b2's producer along the dataflow, the anti-dependency that reuse
 * injects is implied by an edge that was there anyway, so no ordering is added
 * and no parallelism is lost.
 *
 * Transitive reachability costs O(V*E/64) time and O(V^2/64) memory. Above
 * @c PlanOptions::reuse_node_limit nodes the planner **declines to overlap
 * anything** and says so through @c reuse_enabled() / @c reuse_note(), rather
 * than silently substituting the linear approximation. The resulting plan is
 * still a valid plan; it just saves nothing.
 *
 * @section offsets Offsets are elements downstream, bytes here
 *
 * @c SushiBLAS::Tensor::storage_offset counts **elements**, which is what
 * makes a single-dtype view safe by construction. A planner mixing dtypes in
 * one arena does not get that for free: it has to round each value's start up
 * to its own element width itself, because by the time a @c Tensor exists the
 * offset has already been chosen. That rounding is done here - see
 * @c PlanOptions::alignment.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/common.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Identifies an arena in a MemoryPlan; an index, not a pointer. */
        using ArenaId = std::uint32_t;

        /** @brief The "this value is not packed" sentinel; it gets its own allocation. */
        inline constexpr ArenaId NO_ARENA = ~ArenaId{0};

        /**
         * @struct Arena
         * @brief One allocation the plan asks for, and what it holds.
         */
        struct Arena
        {
            /** @brief The lifetime class every value in this arena belongs to. */
            Lifetime lifetime = Lifetime::ACTIVATION;

            /** @brief How many bytes to allocate. */
            std::size_t bytes = 0;

            /** @brief How many distinct buffers were packed into it. */
            std::size_t buffers = 0;

            /** @brief What those buffers would have cost one allocation each. */
            std::size_t unpacked_bytes = 0;
        };

        /**
         * @struct Placement
         * @brief Where one value's elements live.
         *
         * @c buffer is the value that owns the bytes. It differs from the value
         * being looked up exactly when the IR aliases two values onto one
         * buffer - @c ACCUMULATE, whose result *is* its accumulator's storage.
         * Every value in such a chain shares one placement, and the chain's
         * live range is the union of its members', because the bytes are live
         * for as long as any name for them is.
         */
        struct Placement
        {
            /** @brief Which arena holds it, or NO_ARENA for its own allocation. */
            ArenaId arena = NO_ARENA;

            /** @brief Byte offset from the arena's base; 0 when unpacked. */
            std::size_t offset = 0;

            /** @brief How many bytes the value occupies. */
            std::size_t bytes = 0;

            /** @brief The value that owns the bytes; equal to the value unless aliased. */
            ValueId buffer = NO_VALUE;
        };

        /**
         * @struct PlanOptions
         * @brief The three knobs the planner has, and nothing else.
         */
        struct PlanOptions
        {
            /**
             * @brief The granularity every packed value's start is rounded to.
             *
             * Correctness needs only the value's own element width, and the
             * planner enforces that floor whatever this says - a byte offset
             * that is not a whole number of elements cannot be expressed as a
             * @c storage_offset at all.
             *
             * The default is the allocator's own alignment because that is what
             * a standalone @c Storage would have given the same tensor. Packing
             * therefore changes nothing a kernel can observe about alignment,
             * which keeps vectorised paths - and with them bit-for-bit
             * reproducibility - exactly as they were, and stops two tensors
             * written by concurrent kernels from sharing a cache line.
             */
            std::size_t alignment = SushiRuntime::Core::DEFAULT_ALIGNMENT;

            /**
             * @brief Above this many nodes, reuse is declined rather than approximated.
             *
             * The reachability closure is O(V^2/64) bytes; the default keeps it
             * to about 8 MiB. See the file comment.
             */
            std::size_t reuse_node_limit = 8192;

            /** @brief Whether to overlap live ranges at all; false gives every buffer its own slot. */
            bool reuse = true;

            /**
             * @brief Values that must keep their own allocation, whatever their
             *        lifetime says.
             *
             * The planner knows what a value *is* - its shape, its dtype, its
             * class, who reads it - and nothing about what will be *done* with
             * it. A caller that does know, and that knows some operation cannot
             * address the value through an arena, vetoes it here rather than
             * teaching the planner about operations.
             *
             * @c Lowering::values_needing_own_allocation() is the one such
             * caller today, and it names its reason.
             *
             * Read only during @c MemoryPlan::build; it need not outlive the call.
             */
            std::span<const ValueId> pinned{};
        };

        /**
         * @class MemoryPlan
         * @brief The planner's output: one Placement per value, plus arena sizes.
         *
         * A value type. Copyable, and cheap enough to hand around by value; it
         * holds two vectors and nothing that owns memory on a device.
         */
        class MemoryPlan
        {
            public:
                /**
                 * @brief The empty plan: it covers no value and asks for no arena.
                 *
                 * Adopting one is how a caller says "pack nothing" - every value
                 * then falls through to its own allocation, which is what the
                 * library did before this pass existed and what an A/B
                 * measurement of the pass needs as its control.
                 */
                MemoryPlan() = default;

                /**
                 * @brief Plans @p graph.
                 *
                 * @param graph The graph to plan; only read.
                 * @param options How to pack.
                 * @return The plan, covering every value of @p graph.
                 * @throws Error If @p options.alignment is zero, or @p graph is
                 *         not in topological insertion order - a node that reads
                 *         a value written by a later node has no position in the
                 *         partial order the whole pass is built on.
                 */
                static MemoryPlan build(const Graph& graph, const PlanOptions& options = {});

                /** @brief How many values the plan covers. */
                std::size_t value_count() const noexcept { return placements_.size(); }

                /**
                 * @brief Where @p id lives.
                 * @param id The value to look up.
                 * @return Its placement.
                 * @throws Error If @p id is not covered by this plan.
                 */
                const Placement& placement(ValueId id) const;

                /** @brief Whether @p id was packed into an arena. */
                bool is_packed(ValueId id) const noexcept
                {
                    return id < placements_.size() && placements_[id].arena != NO_ARENA;
                }

                /**
                 * @brief Which value owns @p id's bytes.
                 *
                 * The total, non-throwing form of @c placement().buffer, so a
                 * caller can resolve an alias chain against a plan that does not
                 * cover the value at all - which is what the empty plan is.
                 *
                 * @param id The value to resolve.
                 * @return The owning value, or @p id when this plan says nothing about it.
                 */
                ValueId buffer_of(ValueId id) const noexcept
                {
                    if (id >= placements_.size()) return id;
                    const ValueId owner = placements_[id].buffer;
                    return owner == NO_VALUE ? id : owner;
                }

                /** @brief Every arena the plan asks for, in id order. */
                std::span<const Arena> arenas() const noexcept { return arenas_; }

                /** @brief Total bytes across every arena. */
                std::size_t arena_bytes() const noexcept;

                /** @brief What the packed buffers would have cost one allocation each. */
                std::size_t unpacked_bytes() const noexcept;

                /**
                 * @brief Whether live ranges were actually overlapped.
                 *
                 * False means every buffer got its own slot - a correct plan
                 * that saves nothing. @c reuse_note() says why.
                 */
                bool reuse_enabled() const noexcept { return reuse_enabled_; }

                /** @brief Why reuse was declined, or empty when it ran. */
                std::string_view reuse_note() const noexcept { return reuse_note_; }

            private:
                std::vector<Placement> placements_;
                std::vector<Arena> arenas_;
                bool reuse_enabled_ = false;
                std::string_view reuse_note_{};
        };

    } // namespace Graph
} // namespace SushiAI

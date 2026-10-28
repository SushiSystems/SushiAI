/**************************************************************************/
/* memory_plan.hpp                                                        */
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
 * @file memory_plan.hpp
 * @brief Declares the MemoryPlan class and arena placement structures.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/common.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Identifies an arena in a MemoryPlan; an index, not a pointer. */
        using ArenaID = std::uint32_t;

        /** @brief The "this value is not packed" sentinel; it gets its own allocation. */
        inline constexpr ArenaID K_NO_ARENA = ~ArenaID{0};

        /**
         * @struct Arena
         * @brief One allocation the plan asks for, and what it holds.
         */
        struct Arena
        {
            /** @brief The lifetime class every value in this arena belongs to. */
            Lifetime lifetime = Lifetime::Activation;

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
         * The buffer field identifies the owning value, allowing aliased chains
         * such as ACCUMULATE to share storage placements and live ranges.
         */
        struct Placement
        {
            /** @brief Which arena holds it, or NO_ARENA for its own allocation. */
            ArenaID arena = K_NO_ARENA;

            /** @brief Byte offset from the arena's base; 0 when unpacked. */
            std::size_t offset = 0;

            /** @brief How many bytes the value occupies. */
            std::size_t bytes = 0;

            /** @brief The value that owns the bytes; equal to the value unless aliased. */
            ValueID buffer = K_NO_VALUE;
        };

        /**
         * @struct ScratchRequest
         * @brief Names one buffer a node writes on every run before it reads it.
         */
        struct ScratchRequest
        {
            /** @brief The node that asks for the buffer and is its one user. */
            NodeID node = K_NO_NODE;

            /** @brief The scratch slot of the node. */
            int slot = 0;

            /** @brief The extents of the buffer. */
            Shape shape;

            /** @brief The element type of the buffer. */
            DType dtype = DType::Float32;
        };

        /**
         * @struct PlanOptions
         * @brief The three knobs the planner has, and nothing else.
         */
        struct PlanOptions
        {
            /**
             * @brief Granularity each packed value start offset is rounded to.
             *
             * Defaults to allocator alignment to preserve vectorization and reproducibility.
             */
            std::size_t alignment = SushiRuntime::K_DEFAULT_ALIGNMENT;

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
             * @brief Values that must keep their own allocation regardless of lifetime.
             *
             * Read only during build; permits callers to veto arena packing.
             */
            SushiRuntime::span<const ValueID> pinned{};
        };

        /**
         * @class MemoryPlan
         * @brief The planner's output: one Placement per value, plus arena sizes.
         * @see modules/core/graph/README.md
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
                 * @brief Generates an arena memory plan for the given graph.
                 * @param graph The graph to plan.
                 * @param options Packing and alignment options.
                 * @param scratch The buffers to place in the Scratch arena beside the values.
                 * @return The plan, or InvalidDescription for a zero alignment.
                 * @pre @p graph is ordered and holds each pinned value and scratch node.
                 */
                [[nodiscard]] static Result<MemoryPlan> build(
                    const Graph& graph, const PlanOptions& options = {},
                    SushiRuntime::span<const ScratchRequest> scratch = {});

                /** @brief How many values the plan covers. */
                std::size_t value_count() const noexcept { return placements_.size(); }

                /**
                 * @brief Where @p id lives.
                 * @param id The value to look up.
                 * @return Its placement.
                 * @pre @p id is covered by this plan; another id stops the process.
                 */
                const Placement& placement(ValueID id) const noexcept;

                /** @brief Whether @p id was packed into an arena. */
                bool is_packed(ValueID id) const noexcept
                {
                    return id < placements_.size() && placements_[id].arena != K_NO_ARENA;
                }

                /**
                 * @brief Resolves which value owns the specified value's bytes.
                 *
                 * Non-throwing lookup to resolve alias chains even in an empty plan.
                 *
                 * @param id The value to resolve.
                 * @return Owning value id, or id itself if unmapped.
                 */
                ValueID buffer_of(ValueID id) const noexcept
                {
                    if (id >= placements_.size()) return id;
                    const ValueID owner = placements_[id].buffer;
                    return owner == K_NO_VALUE ? id : owner;
                }

                /**
                 * @brief Where the scratch buffer of @p node and @p slot lives.
                 * @return The placement, or nullptr when the plan was given no such request.
                 */
                const Placement* scratch_placement(NodeID node, int slot) const noexcept;

                /** @brief Every arena the plan asks for, in id order. */
                SushiRuntime::span<const Arena> arenas() const noexcept { return arenas_; }

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
                /** @brief Holds the placement of one requested scratch buffer. */
                struct ScratchSlot
                {
                    NodeID node = K_NO_NODE;
                    int slot = 0;
                    Placement placement;
                };

                std::vector<Placement> placements_;
                std::vector<ScratchSlot> scratch_;
                std::vector<Arena> arenas_;
                bool reuse_enabled_ = false;
                std::string_view reuse_note_{};
        };

    } // namespace Graph
} // namespace SushiAI

/**************************************************************************/
/* lowering.hpp                                                           */
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
 * @file lowering.hpp
 * @brief Turns SushiAI IR into recorded SushiBLAS work.
 *
 * This is the only file in the library that calls @c SushiBLAS::Engine. Every
 * other layer - autograd, the layer library, the optimizers' *shapes* - deals
 * in IR, so the mapping from "what the model means" to "which kernels run"
 * lives in exactly one place and can be re-read against the BLAS surface in
 * one sitting.
 *
 * @section pool One Storage per value, except where the planner says otherwise
 *
 * @c TensorPool used to give every @c ValueId its own @c SushiBLAS::Storage,
 * and the reason was a correctness requirement rather than a missed
 * optimisation: almost every SushiBLAS kernel captured
 * @c tensor.storage->data_ptr and ignored @c tensor.storage_offset (only the
 * reductions, @c logic().all/any, @c random().constant and @c io() honoured
 * it), so an offset view handed to gemm, axpy or an elementwise op read from
 * element zero of the whole arena and produced a silently wrong answer.
 *
 * **That reason no longer holds.** Every SushiBLAS op family honours
 * @c storage_offset, and @c Tensor::region() reports a view's own byte
 * interval rather than the allocation base, so two disjoint tensors packed
 * into one arena are correctly seen as not conflicting instead of falsely
 * serialising. The pool therefore asks a @c MemoryPlan where each value goes:
 * a packed value becomes a view into an arena allocated once at bind time, and
 * everything else - leaves, the persistent lifetime classes, the shared
 * constants and the per-node scratch below - still gets its own Storage.
 * *Which* values are packed, and why the arenas are split by lifetime rather
 * than merged into one, is memory_plan.hpp's business and is argued there.
 *
 * @section rewritten What the planner assumes about this file
 *
 * **Every node writes all of its outputs on every replay.** The planner hands
 * a dead value's bytes to a later one, so a buffer that is filled once and
 * then only ever read cannot live in an arena: it would survive exactly one
 * step and then be quietly overwritten. That is why @c ZERO records its fill
 * into the step instead of into the one-time constant queue below, and why
 * that queue is only ever used for buffers the planner does not place.
 *
 * @section constants Constant operands are filled once, not every step
 *
 * Several lowerings need an all-ones vector (bias broadcast, row sums) or an
 * all-zero buffer. The pool owns one per distinct shape and dtype and records
 * the fill into a @e pending list rather than into the step being recorded;
 * @c TensorPool::flush_constants() runs them after the step has been compiled
 * and drained. Recording them inline would re-fill an unchanging buffer on
 * every replay, and filling them eagerly mid-recording would call
 * @c Engine::execute(), which compiles and drains the half-built step.
 *
 * @section logsoftmax Why log-softmax shifts by a device-side global maximum
 *
 * The stable recipe subtracts a maximum before exponentiating. SushiBLAS has
 * no axis-wise reduction, so the shift is the tensor's @e global maximum
 * rather than a per-row one. In exact arithmetic that is not an approximation:
 * log-softmax is invariant to subtracting any single scalar from every
 * element, so the global maximum gives the same answer as the row-wise one
 * while still driving every exponent to at most zero, which is all overflow
 * stability asks for.
 *
 * @warning In floating point the two differ, and the difference has a
 *          measurable edge. A row whose own maximum sits far @e below the
 *          batch maximum has every one of its exponentials rounded to zero, so
 *          its row sum underflows, @c log() of it is @c -inf, and the row's
 *          log-probabilities come back @c +inf. Measured in binary32 on a
 *          [2, 3] input: exact to about 90 below the batch maximum, about 7%
 *          low at 100 (the sum is denormal by then), and @c +inf from about
 *          104 on - which then makes the loss and every gradient below it
 *          non-finite. Row-wise shifting has no such edge; the global shift
 *          trades it for not needing an axis reduction SushiBLAS does not
 *          have. It is the right trade for logits a training run actually
 *          produces, whose rows sit within a few tens of each other, and it is
 *          the wrong one for a model that has already diverged. A fused
 *          SushiBLAS @c log_softmax_rows removes the edge along with the rest
 *          of the recipe; nothing short of a real row-wise maximum does.
 *          Integration_Resilience.LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange
 *          pins the supported side of that boundary.
 *
 * The maximum stays on the device. Reading it back to the host and passing it
 * as an @c axpy alpha would be simpler to write, and it is what a first draft
 * does - but the alpha would be captured when the step is @e recorded, so a
 * compiled-once, replayed-many step would keep shifting by the first step's
 * maximum forever. Broadcasting the 1-element maximum through a gemv against
 * a [C, 1] ones matrix costs one extra tiny kernel and keeps the whole step
 * replayable, which is the property AI-1 exists to prove. A future fused
 * SushiBLAS @c log_softmax_rows would replace the whole seven-op recipe.
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include <unordered_map>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class TensorPool
         * @brief Owns the real memory behind a graph's values, across steps.
         *
         * One @c SushiBLAS::Tensor per @c ValueId, drawn either from its own
         * Storage or from a planned arena (see the file comment). The pool
         * outlives any single step, so a compiled step that replays finds the
         * same buffers every time and a steady-state training loop allocates
         * nothing.
         */
        class TensorPool
        {
            public:
                /**
                 * @brief Constructs an empty pool drawing memory from @p engine.
                 * @param engine The engine whose context owns the allocator; must outlive the pool.
                 */
                explicit TensorPool(SushiBLAS::Engine& engine) noexcept : engine_(&engine) {}

                /**
                 * @brief Allocates @p plan's arenas and packs future values into them.
                 *
                 * This is "bind time": every arena the plan asks for is
                 * allocated here, once, and nothing allocates one again for the
                 * pool's whole life. Call it before any planned value has been
                 * materialised - a value that already has its own Storage keeps
                 * it, so adopting late silently plans nothing.
                 *
                 * @param plan Where each value's bytes go; moved in.
                 * @throws Error If a plan has already been adopted, or an arena
                 *         allocation fails.
                 */
                void adopt(MemoryPlan plan);

                /** @brief Whether a plan has been adopted. */
                bool has_plan() const noexcept { return arenas_.has_value(); }

                /**
                 * @brief The adopted plan and its arenas.
                 * @return The arena set.
                 * @throws Error If no plan has been adopted.
                 */
                const ArenaSet& arenas() const;

                /**
                 * @brief Allocates a standalone tensor of @p shape and @p dtype.
                 *
                 * Its own Storage, zero offset, the engine's default layout.
                 * Used for values, helper constants and any caller-side buffer
                 * that must be safe to hand to a SushiBLAS kernel.
                 *
                 * @param shape The extents.
                 * @param dtype The element type.
                 * @return The new tensor; the pool does not retain it.
                 */
                SushiBLAS::Tensor create(const Shape& shape, DType dtype) const;

                /**
                 * @brief Binds an externally owned tensor to @p id.
                 *
                 * For a leaf whose memory the caller already has - a dataset
                 * batch, a parameter shared with an optimizer.
                 *
                 * @param id The value to bind.
                 * @param tensor The tensor to bind it to; shares its Storage.
                 * @throws Error If @p id is already bound.
                 */
                void bind(ValueId id, SushiBLAS::Tensor tensor);

                /** @brief Whether @p id has a tensor behind it yet. */
                bool is_bound(ValueId id) const noexcept { return tensors_.find(id) != tensors_.end(); }

                /**
                 * @brief The tensor behind @p id.
                 * @param id The value to look up.
                 * @return A reference stable for the pool's lifetime.
                 * @throws Error If @p id is not bound.
                 */
                SushiBLAS::Tensor& tensor(ValueId id);

                /** @brief Const overload of tensor(). */
                const SushiBLAS::Tensor& tensor(ValueId id) const;

                /**
                 * @brief Returns @p id's tensor, allocating it on first request.
                 * @param id The value to materialise.
                 * @param value The value's description, read for shape and dtype.
                 * @return A reference stable for the pool's lifetime.
                 */
                SushiBLAS::Tensor& materialise(ValueId id, const Value& value);

                /**
                 * @brief Allocates every leaf of @p graph that is not bound yet.
                 *
                 * Lowering refuses to record an unbound leaf, because a leaf is
                 * by definition written from outside the step and fresh
                 * uninitialised storage would train on garbage rather than
                 * fail. Call this once the leaves you do want to supply
                 * yourself are bound.
                 *
                 * @param graph The graph whose leaves to allocate.
                 */
                void materialise_leaves(const Graph& graph);

                /**
                 * @brief Points @p target at the same tensor as @p source.
                 *
                 * How @c ACCUMULATE avoids a buffer: its output is its first
                 * input's storage, updated in place.
                 *
                 * Idempotent when @p target already names those exact bytes,
                 * because the planner reaches the same conclusion independently
                 * - it has to, or it would think the accumulator died at the
                 * @c ACCUMULATE node and hand its bytes away while the result
                 * is still being read.
                 *
                 * @param target The value to bind.
                 * @param source The already-bound value to share with.
                 * @throws Error If @p source is unbound, or @p target is already
                 *         bound to a different buffer.
                 */
                void alias(ValueId target, ValueId source);

                /**
                 * @brief The shared all-ones tensor of this shape and dtype.
                 *
                 * Created on first request and queued for a one-time fill; see
                 * flush_constants().
                 *
                 * @param shape The extents.
                 * @param dtype The element type.
                 * @return A reference stable for the pool's lifetime.
                 */
                SushiBLAS::Tensor& ones(const Shape& shape, DType dtype);

                /**
                 * @brief A per-node scratch buffer, reused across replays.
                 *
                 * Keyed by node and slot rather than pooled by shape, so two
                 * nodes never share a temporary and the recorded dependency
                 * edges stay exactly the IR's dataflow.
                 *
                 * @param node The owning node.
                 * @param slot The node's index for this temporary.
                 * @param shape The extents.
                 * @param dtype The element type.
                 * @return A reference stable for the pool's lifetime.
                 */
                SushiBLAS::Tensor& scratch(NodeId node, int slot, const Shape& shape, DType dtype);

                /**
                 * @brief Queues @p tensor to be filled with @p value once.
                 * @param tensor The tensor to fill; must outlive the flush.
                 * @param value The constant.
                 */
                void request_constant(SushiBLAS::Tensor& tensor, double value);

                /**
                 * @brief Runs every queued constant fill and waits for it.
                 *
                 * Records the fills into the engine's (by then empty) task
                 * graph, executes, and waits. Call it after the step has been
                 * compiled - never during recording, because it drains the
                 * builder.
                 */
                void flush_constants();

                /** @brief How many tensors the pool has handed out. */
                std::size_t tensor_count() const noexcept;

                /**
                 * @brief Bytes actually allocated, excluding bound externals.
                 *
                 * Every arena in full, plus the elements of every tensor that
                 * kept its own Storage. This is the number that shrinks when
                 * planning works; compare it against @c unpacked_bytes().
                 */
                std::size_t allocated_bytes() const noexcept
                {
                    return arena_bytes_ + standalone_bytes_;
                }

                /**
                 * @brief What the same tensors would cost one allocation each.
                 *
                 * The counterfactual the saving is measured against: the sum of
                 * every tensor's own elements, packed or not. Equal to
                 * @c allocated_bytes() when no plan has been adopted.
                 */
                std::size_t unpacked_bytes() const noexcept { return unpacked_bytes_; }

                /** @brief The engine this pool draws memory from. */
                SushiBLAS::Engine& engine() const noexcept { return *engine_; }

            private:
                /** @brief Books a tensor that got its own Storage, for both counters. */
                void note_standalone(const Shape& shape, DType dtype) noexcept;

                struct ShapeKey
                {
                    std::uint64_t shape_hash;
                    std::int32_t dtype;

                    bool operator==(const ShapeKey& other) const noexcept
                    {
                        return shape_hash == other.shape_hash && dtype == other.dtype;
                    }
                };

                struct ShapeKeyHash
                {
                    std::size_t operator()(const ShapeKey& key) const noexcept
                    {
                        return static_cast<std::size_t>(key.shape_hash * 31u
                                                        + static_cast<std::uint64_t>(key.dtype));
                    }
                };

                struct ScratchKey
                {
                    NodeId node;
                    int slot;

                    bool operator==(const ScratchKey& other) const noexcept
                    {
                        return node == other.node && slot == other.slot;
                    }
                };

                struct ScratchKeyHash
                {
                    std::size_t operator()(const ScratchKey& key) const noexcept
                    {
                        return static_cast<std::size_t>(key.node) * 131u
                             + static_cast<std::size_t>(key.slot);
                    }
                };

                SushiBLAS::Engine* engine_;
                std::optional<ArenaSet> arenas_;
                std::unordered_map<ValueId, SushiBLAS::Tensor> tensors_;
                std::unordered_map<ShapeKey, SushiBLAS::Tensor, ShapeKeyHash> ones_;
                std::unordered_map<ScratchKey, SushiBLAS::Tensor, ScratchKeyHash> scratch_;
                std::vector<std::pair<SushiBLAS::Tensor*, double>> pending_constants_;
                std::size_t arena_bytes_ = 0;
                std::size_t standalone_bytes_ = 0;
                std::size_t unpacked_bytes_ = 0;
        };

        /**
         * @class Lowering
         * @brief Records a whole graph as SushiBLAS work, in node order.
         *
         * Stateless between calls apart from the pool it writes through, so a
         * caller may lower the same graph into a fresh engine graph again -
         * which is what makes the "compile once" claim testable rather than
         * structural.
         */
        class Lowering
        {
            public:
                /**
                 * @brief Binds a lowering to an engine and a pool.
                 * @param engine Where the work is recorded; must outlive this object.
                 * @param pool Where the memory comes from; must outlive this object.
                 */
                Lowering(SushiBLAS::Engine& engine, TensorPool& pool) noexcept
                    : engine_(&engine), pool_(&pool) {}

                /**
                 * @brief The values this lowering cannot address inside an arena.
                 *
                 * Feed the result to @c PlanOptions::pinned. Every entry is a
                 * workaround for a defect elsewhere rather than a property of
                 * the IR, so each one is named in the implementation with the
                 * fix that retires it - the list is meant to shrink to empty.
                 *
                 * As of today it holds the operands of SushiBLAS's whole-tensor
                 * reductions. @c Dispatch::execute_reduction describes them to
                 * the dependency tracker as bare @c data() pointers instead of
                 * @c Tensor::region(): identical for a tensor at offset zero, a
                 * different resource entirely for a view inside an arena. The
                 * tracker then files the access under a resource nothing else
                 * touches and emits no edge at all, so the reduction races its
                 * own producer - a wrong number, never an error.
                 *
                 * @param graph The graph about to be lowered.
                 * @return The value ids that must keep their own allocation.
                 */
                static std::vector<ValueId> values_needing_own_allocation(const Graph& graph);

                /**
                 * @brief Records every node of @p graph into the engine's task graph.
                 *
                 * Nothing is executed and nothing is compiled: on return the
                 * engine's builder holds the whole step, ready for exactly one
                 * @c compile().
                 *
                 * @param graph The graph to lower.
                 * @throws Error If a leaf is unbound, an operand's shape is
                 *         wrong for its operation, or the graph contains an
                 *         operation this lowering does not implement.
                 */
                void record(const Graph& graph);

                /** @brief How many SushiBLAS calls the last record() emitted. */
                std::size_t emitted_operations() const noexcept { return emitted_; }

            private:
                void record_node(const Graph& graph, NodeId id);

                void record_matmul(const Graph& graph, NodeId id, const Node& node);
                void record_bias_add(const Graph& graph, NodeId id, const Node& node);
                void record_binary(const Graph& graph, const Node& node);
                void record_scale(const Graph& graph, const Node& node);
                void record_unary_inplace(const Graph& graph, const Node& node);
                void record_activation_backward(const Graph& graph, const Node& node);
                void record_log_softmax(const Graph& graph, NodeId id, const Node& node);
                void record_nll_loss(const Graph& graph, NodeId id, const Node& node);
                void record_sum_rows(const Graph& graph, const Node& node);
                void record_accumulate(const Node& node);
                void record_zero(const Graph& graph, const Node& node);

                SushiBLAS::Tensor& output(const Graph& graph, const Node& node, std::size_t index);
                const SushiBLAS::Tensor& input(const Node& node, std::size_t index) const;

                SushiBLAS::Engine* engine_;
                TensorPool* pool_;
                std::size_t emitted_ = 0;
        };

    } // namespace Graph
} // namespace SushiAI

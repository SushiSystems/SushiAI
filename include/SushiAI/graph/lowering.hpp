/****************************************************************/
/* lowering.hpp                                                 */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file lowering.hpp
 * @brief Translates SushiAI IR graphs into executable SushiBLAS task streams.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include <unordered_map>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class TensorPool
         * @brief Owns device memory buffers backing graph values across steps.
         *
         * Maintains persistent tensor storages and arena slices across replays.
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
                 * @brief Allocates arenas from the specified memory plan.
                 *
                 * Must be called before any planned value is materialised.
                 *
                 * @param plan Arena placement plan to adopt.
                 * @throws Error If a plan was already adopted or allocation fails.
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
                 * @brief Allocates an unmanaged standalone tensor.
                 *
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return Newly allocated tensor instance.
                 */
                SushiBLAS::Tensor create(const Shape& shape, DType dtype) const;

                /**
                 * @brief Binds an externally owned tensor to the given value id.
                 *
                 * @param id Value identifier to bind.
                 * @param tensor External tensor to associate.
                 * @throws Error If the value is already bound.
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
                 * @brief Allocates uninitialised storage for unbound graph leaves.
                 *
                 * @param graph Graph whose unbound input leaves are allocated.
                 */
                void materialise_leaves(const Graph& graph);

                /**
                 * @brief Aliases target value storage to an existing source tensor.
                 *
                 * @param target Value identifier to alias.
                 * @param source Already-bound source value identifier.
                 * @throws Error If source is unbound or target is already bound elsewhere.
                 */
                void alias(ValueId target, ValueId source);

                /**
                 * @brief Retrieves or creates a shared constant all-ones tensor.
                 *
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return Stable reference to the allocated all-ones tensor.
                 */
                SushiBLAS::Tensor& ones(const Shape& shape, DType dtype);

                /**
                 * @brief Retrieves or creates a per-node scratch buffer.
                 *
                 * Keyed by node and slot, reused across replays.
                 *
                 * @return Stable reference to scratch tensor.
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
         * @brief Translates IR graph operations into recorded SushiBLAS work.
         *
         * Records device execution tasks in topological node order.
         */
        class Lowering
        {
            public:
                /**
                 * @brief Constructs a lowering coordinator bound to engine and pool.
                 *
                 * @param engine Target SushiBLAS engine for recorded work.
                 * @param pool Memory pool supplying buffer allocations.
                 * @param selector Optional fused kernel selection strategy.
                 */
                Lowering(SushiBLAS::Engine& engine, TensorPool& pool,
                         const IFusionSelector* selector = nullptr) noexcept
                    : engine_(&engine), pool_(&pool), selector_(selector) {}

                /**
                 * @brief Identifies values that cannot be safely packed into arenas.
                 *
                 * @param graph Graph to inspect for unpinnable values.
                 * @return Vector of value ids requiring standalone allocations.
                 */
                static std::vector<ValueId> values_needing_own_allocation(const Graph& graph);

                /**
                 * @brief Records graph operations into the engine task graph.
                 *
                 * @param graph IR graph to lower.
                 * @throws Error If an operand shape is invalid or leaf is unbound.
                 */
                void record(const Graph& graph);

                /**
                 * @brief Returns the count of SushiBLAS operations recorded.
                 *
                 * @return Number of operations emitted during the last lowering.
                 */
                std::size_t emitted_operations() const noexcept { return emitted_; }

                /**
                 * @brief Returns fusion selector decisions recorded during lowering.
                 *
                 * @return Span of chosen fusion configurations in execution order.
                 */
                SushiRuntime::span<const FusionSelection> selections() const noexcept
                {
                    return SushiRuntime::span<const FusionSelection>(selections_.data(), selections_.size());
                }

            private:
                void record_node(const Graph& graph, NodeId id);

                void record_matmul(const Graph& graph, NodeId id, const Node& node);
                void record_bias_add(const Graph& graph, NodeId id, const Node& node);
                void record_binary(const Graph& graph, const Node& node);
                void record_scale(const Graph& graph, const Node& node);
                void record_dynamic_scale(const Graph& graph, NodeId id, const Node& node);
                void record_unary_inplace(const Graph& graph, const Node& node);
                void record_activation_backward(const Graph& graph, const Node& node);
                void record_leaky_relu(const Graph& graph, const Node& node);
                void record_leaky_relu_backward(const Graph& graph, const Node& node);
                void record_log_softmax(const Graph& graph, NodeId id, const Node& node);
                void record_nll_loss(const Graph& graph, NodeId id, const Node& node);
                void record_sum_rows(const Graph& graph, const Node& node);
                void record_accumulate(const Node& node);
                void record_zero(const Graph& graph, const Node& node);
                void record_cast(const Graph& graph, NodeId id, const Node& node);
                void record_reshape(const Graph& graph, NodeId id, const Node& node);
                void record_layout_convert(const Graph& graph, NodeId id, const Node& node);
                void record_im2col(const Graph& graph, NodeId id, const Node& node);
                void record_col2im(const Graph& graph, NodeId id, const Node& node);
                void record_maxpool2d(const Graph& graph, NodeId id, const Node& node);
                void record_maxpool2d_backward(const Graph& graph, NodeId id, const Node& node);
                void record_avgpool2d(const Graph& graph, NodeId id, const Node& node);
                void record_avgpool2d_backward(const Graph& graph, NodeId id, const Node& node);
                void record_upsample2d(const Graph& graph, NodeId id, const Node& node);
                void record_upsample2d_backward(const Graph& graph, NodeId id, const Node& node);
                void record_spatial_mean(const Graph& graph, NodeId id, const Node& node);
                void record_spatial_mean_backward(const Graph& graph, NodeId id, const Node& node);
                void record_concat_channels(const Graph& graph, NodeId id, const Node& node);
                void record_slice_channels(const Graph& graph, NodeId id, const Node& node);
                void record_channel_affine(const Graph& graph, NodeId id, const Node& node);
                void record_batchnorm2d(const Graph& graph, NodeId id, const Node& node);
                void record_batchnorm2d_backward(const Graph& graph, NodeId id, const Node& node);

                void record_fused_gemm(const Graph& graph, NodeId id, const Node& node,
                                       FusedForm matched);
                void record_fused_exp_sub_scale(const Graph& graph, NodeId id, const Node& node,
                                                FusedForm matched);

                /** @brief Asks the selector, recording the answer for selections(). */
                FusionChoice select(const FusionQuery& query);

                SushiBLAS::Tensor& output(const Graph& graph, const Node& node, std::size_t index);
                const SushiBLAS::Tensor& input(const Node& node, std::size_t index) const;

                SushiBLAS::Engine* engine_;
                TensorPool* pool_;
                const IFusionSelector* selector_ = nullptr;
                std::vector<FusionSelection> selections_;
                std::size_t emitted_ = 0;
        };

    } // namespace Graph
} // namespace SushiAI

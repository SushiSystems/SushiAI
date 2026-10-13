/**************************************************************************/
/* tensor_pool.hpp                                                        */
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
 * @file tensor_pool.hpp
 * @brief Declares the TensorPool that owns the tensors backing graph values.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include <unordered_map>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Holds the bytes a pool allocated, by what they back. */
        struct PoolBytes
        {
            std::size_t arenas = 0;  /**< Every arena the pool allocated, in full. */
            std::size_t values = 0;  /**< Graph values that kept a Storage of their own. */
            std::size_t ones = 0;    /**< The shared all-ones tensors. */
            std::size_t scratch = 0; /**< The per-node scratch buffers outside the arenas. */

            /** @brief Returns the sum of the four. */
            std::size_t total() const noexcept { return arenas + values + ones + scratch; }
        };

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
                 * @brief Allocates arenas from the specified memory plan, or borrows them.
                 * @param plan Arena placement plan to adopt.
                 * @param lender A pool whose arenas this one uses where they fit, or nullptr.
                 * @return Success, or the failure of an arena allocation.
                 * @pre No plan has been adopted and no planned value has been materialised; a
                 *      second adoption stops the process.
                 */
                [[nodiscard]] Status adopt(MemoryPlan plan, const TensorPool* lender = nullptr);

                /** @brief Whether a plan has been adopted. */
                bool has_plan() const noexcept { return arenas_.has_value(); }

                /**
                 * @brief The adopted plan and its arenas.
                 * @return The arena set.
                 * @pre A plan has been adopted; asking earlier stops the process.
                 */
                const ArenaSet& arenas() const;

                /**
                 * @brief Allocates an unmanaged standalone tensor.
                 *
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return Newly allocated tensor instance, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor> create(const Shape& shape,
                                                               DType dtype) const;

                /**
                 * @brief Binds an externally owned tensor to the given value id.
                 *
                 * @param id Value identifier to bind.
                 * @param tensor External tensor to associate.
                 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
                 * @pre @p id is not bound yet; a second binding stops the process.
                 */
                [[nodiscard]] Status bind(ValueID id, SushiBLAS::Tensor tensor);

                /** @brief Whether @p id has a tensor behind it yet. */
                bool is_bound(ValueID id) const noexcept
                {
                    return tensors_.find(id) != tensors_.end();
                }

                /**
                 * @brief The tensor behind @p id.
                 * @param id The value to look up.
                 * @return A reference stable for the pool's lifetime.
                 * @pre @p id is bound; asking for an unbound value stops the process.
                 */
                SushiBLAS::Tensor& tensor(ValueID id);

                /** @brief Const overload of tensor(). */
                const SushiBLAS::Tensor& tensor(ValueID id) const;

                /**
                 * @brief Returns @p id's tensor, allocating it on first request.
                 * @param id The value to materialise.
                 * @param value The value's description, read for shape and dtype.
                 * @return A reference stable for the pool's lifetime, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> materialise(ValueID id,
                                                                     const Value& value);

                /**
                 * @brief Allocates uninitialised storage for unbound graph leaves.
                 *
                 * @param graph Graph whose unbound input leaves are allocated.
                 * @return Success, or the allocation failure.
                 */
                [[nodiscard]] Status materialise_leaves(const Graph& graph);

                /**
                 * @brief Aliases target value storage to an existing source tensor.
                 * @param target Value identifier to alias.
                 * @param source Already-bound source value identifier.
                 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
                 * @pre @p source is bound, and @p target is unbound or already aliases it; any
                 *      other case stops the process.
                 */
                [[nodiscard]] Status alias(ValueID target, ValueID source);

                /**
                 * @brief Retrieves or creates a shared constant all-ones tensor.
                 *
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return Stable reference to the allocated all-ones tensor, or the allocation
                 *         failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> ones(const Shape& shape, DType dtype);

                /**
                 * @brief Retrieves or creates a per-node scratch buffer.
                 *
                 * Keyed by node and slot, reused across replays. A buffer the adopted plan
                 * placed is a view of its arena; any other keeps a Storage of its own.
                 *
                 * @return Stable reference to scratch tensor, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> scratch(NodeID node, int slot,
                                                                 const Shape& shape, DType dtype);

                /**
                 * @brief Queues @p tensor to be filled with @p value once.
                 * @param tensor The tensor to fill; must outlive the flush.
                 * @param value The constant.
                 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
                 */
                [[nodiscard]] Status request_constant(SushiBLAS::Tensor& tensor, double value);

                /**
                 * @brief Records every queued constant fill and runs it until it has finished.
                 * @return Success when nothing is queued, otherwise the failure of the recording
                 *         or of the run.
                 * @pre The engine's own graph holds no recorded task; the run executes all of it.
                 */
                [[nodiscard]] Status flush_constants();

                /** @brief How many tensors the pool has handed out. */
                std::size_t tensor_count() const noexcept;

                /**
                 * @brief Bytes actually allocated, excluding bound externals.
                 *
                 * Every arena in full, plus the elements of every tensor that
                 * kept its own Storage. This is the number that shrinks when
                 * planning works; compare it against @c unpacked_bytes().
                 */
                std::size_t allocated_bytes() const noexcept { return bytes_.total(); }

                /** @brief Returns allocated_bytes() split by what the bytes back. */
                const PoolBytes& bytes() const noexcept { return bytes_; }

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
                /**
                 * @brief Books a tensor that got its own Storage.
                 * @param counter The member of bytes_ the tensor belongs to.
                 */
                void note_standalone(std::size_t& counter, const Shape& shape,
                                     DType dtype) noexcept;

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
                    NodeID node;
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

                /**
                 * @brief Returns the tensor that holds the bytes of the unpacked buffer
                 *        @p owner names, or nullptr when none of its values has a tensor yet.
                 */
                const SushiBLAS::Tensor* find_shared(ValueID owner) const noexcept;

                SushiBLAS::Engine* engine_;
                std::optional<ArenaSet> arenas_;
                std::unordered_map<ValueID, SushiBLAS::Tensor> tensors_;

                /** @brief The first tensor of each unpacked buffer whose owner came later. */
                std::unordered_map<ValueID, SushiBLAS::Tensor> shared_;
                std::unordered_map<ShapeKey, SushiBLAS::Tensor, ShapeKeyHash> ones_;
                std::unordered_map<ScratchKey, SushiBLAS::Tensor, ScratchKeyHash> scratch_;
                std::vector<std::pair<SushiBLAS::Tensor*, double>> pending_constants_;
                PoolBytes bytes_;
                std::size_t unpacked_bytes_ = 0;
        };

    } // namespace Graph
} // namespace SushiAI

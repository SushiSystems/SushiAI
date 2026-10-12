/**************************************************************************/
/* tensor_pool.cpp                                                        */
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
 * @file tensor_pool.cpp
 * @brief Implements the TensorPool's allocation, binding and constant-fill bookkeeping.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/tensor_pool.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <SushiBLAS/random/random.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Adds @p tensor to @p map under @p key and returns the stored tensor.
             * @param map A map from Key to SushiBLAS::Tensor; must outlive the reference.
             * @return The tensor the map holds for @p key, or the failure of the allocation seam.
             */
            template<typename Map, typename Key, typename Stored>
            [[nodiscard]] Result<SushiBLAS::Tensor&> store_tensor(Map& map, const Key& key,
                                                                 Stored&& tensor)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor* const stored,
                              ::SushiAI::Detail::guard_value(
                                  ::SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    return &map.emplace(key, std::forward<Stored>(tensor)).first->second;
                }));
                return *stored;
            }
        }

        Result<SushiBLAS::Tensor> TensorPool::create(const Shape& shape, DType dtype) const
        {
            return allocate_tensor(*engine_, shape, dtype);
        }

        Status TensorPool::adopt(MemoryPlan plan)
        {
            SA_FATAL_IF(arenas_.has_value(),
                        "This pool has already adopted a memory plan; its arenas are allocated "
                        "once and a second plan would leave the first one's views dangling");

            SA_TRY_ASSIGN(ArenaSet arenas, ArenaSet::make(*engine_, std::move(plan)));
            arenas_.emplace(std::move(arenas));
            bytes_.arenas = arenas_->bytes();
            return Status();
        }

        const ArenaSet& TensorPool::arenas() const
        {
            SA_FATAL_IF(!arenas_.has_value(),
                        "This pool has no memory plan; call adopt() before asking for its arenas");
            return *arenas_;
        }

        Status TensorPool::bind(ValueID id, SushiBLAS::Tensor tensor)
        {
            SA_FATAL_IF(is_bound(id),
                        "Value {} is already bound to a tensor; a value has one buffer", id);

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                tensors_.emplace(id, std::move(tensor));
                return Status();
            });
        }

        SushiBLAS::Tensor& TensorPool::tensor(ValueID id)
        {
            const auto it = tensors_.find(id);
            SA_FATAL_IF(it == tensors_.end(),
                        "Value {} has no tensor; bind or materialise it before lowering", id);
            return it->second;
        }

        const SushiBLAS::Tensor& TensorPool::tensor(ValueID id) const
        {
            const auto it = tensors_.find(id);
            SA_FATAL_IF(it == tensors_.end(),
                        "Value {} has no tensor; bind or materialise it before lowering", id);
            return it->second;
        }

        void TensorPool::note_standalone(std::size_t& counter, const Shape& shape,
                                         DType dtype) noexcept
        {
            const std::size_t bytes = static_cast<std::size_t>(shape.num_elements())
                                    * dtype_size(dtype);
            unpacked_bytes_ += bytes;
            counter += bytes;
        }

        Result<SushiBLAS::Tensor&> TensorPool::materialise(ValueID id, const Value& value)
        {
            const auto it = tensors_.find(id);
            if (it != tensors_.end()) return it->second;

            if (arenas_.has_value())
            {
                // Materialises alias chains through buffer owner to align with planner liveness.
                const ValueID owner = arenas_->plan().buffer_of(id);
                if (owner != id)
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor& shared, materialise(owner, value));
                    return store_tensor(tensors_, id, shared);
                }

                if (arenas_->holds(id))
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor slice,
                                  ::SushiAI::Detail::guard_value(
                                      ::SushiAI::Detail::SeamPhase::Allocate, [&]()
                    {
                        return arenas_->view(id, value.shape, value.dtype);
                    }));

                    // Tracks counterfactual unpacked byte totals without reallocating storage.
                    unpacked_bytes_ += static_cast<std::size_t>(value.shape.num_elements())
                                     * dtype_size(value.dtype);
                    return store_tensor(tensors_, id, std::move(slice));
                }
            }

            SA_TRY_ASSIGN(SushiBLAS::Tensor own, create(value.shape, value.dtype));
            note_standalone(bytes_.values, value.shape, value.dtype);
            return store_tensor(tensors_, id, std::move(own));
        }

        Status TensorPool::materialise_leaves(const Graph& graph)
        {
            for (const Node& node : graph.nodes())
            {
                if (node.op != Ops::K_LEAF.id) continue;
                for (ValueID out : node.outputs)
                    SA_TRY(materialise(out, graph.value(out)));
            }
            return Status();
        }

        Status TensorPool::alias(ValueID target, ValueID source)
        {
            const auto it = tensors_.find(source);
            SA_FATAL_IF(it == tensors_.end(),
                        "Cannot alias value {} onto unbound value {}", target, source);

            const auto existing = tensors_.find(target);
            if (existing != tensors_.end())
            {
                // Validates that existing alias points to the same storage and offset.
                SA_FATAL_IF(existing->second.storage != it->second.storage
                            || existing->second.storage_offset != it->second.storage_offset,
                            "Value {} is already bound to a different buffer; it cannot also "
                            "alias value {}", target, source);
                return Status();
            }

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                tensors_.emplace(target, it->second);
                return Status();
            });
        }

        Result<SushiBLAS::Tensor&> TensorPool::ones(const Shape& shape, DType dtype)
        {
            const ShapeKey key{shape.hash(), static_cast<std::int32_t>(dtype)};

            const auto it = ones_.find(key);
            if (it != ones_.end()) return it->second;

            SA_TRY_ASSIGN(SushiBLAS::Tensor fill, create(shape, dtype));
            note_standalone(bytes_.ones, shape, dtype);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& slot, store_tensor(ones_, key, std::move(fill)));

            const Status queued = request_constant(slot, 1.0);

            // A ones tensor whose fill was not queued leaves the cache, so no caller reads it.
            if (!queued.is_ok())
            {
                ones_.erase(key);
                return queued;
            }
            return slot;
        }

        Result<SushiBLAS::Tensor&> TensorPool::scratch(NodeID node, int slot,
                                                       const Shape& shape, DType dtype)
        {
            const ScratchKey key{node, slot};

            const auto it = scratch_.find(key);
            if (it != scratch_.end()) return it->second;

            SA_TRY_ASSIGN(SushiBLAS::Tensor buffer, create(shape, dtype));
            note_standalone(bytes_.scratch, shape, dtype);

            return store_tensor(scratch_, key, std::move(buffer));
        }

        Status TensorPool::request_constant(SushiBLAS::Tensor& tensor, double value)
        {
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                pending_constants_.emplace_back(&tensor, value);
                return Status();
            });
        }

        Status TensorPool::flush_constants()
        {
            if (pending_constants_.empty()) return Status();

            return ::SushiAI::Detail::record_and_run(*engine_, [this]()
            {
                SushiBLAS::RandomOps filler(*engine_);
                for (const auto& [tensor, value] : pending_constants_)
                    filler.constant(*tensor, value);

                // The list empties once every fill is recorded, before any of them runs.
                pending_constants_.clear();
                return Status();
            });
        }

        std::size_t TensorPool::tensor_count() const noexcept
        {
            return tensors_.size() + ones_.size() + scratch_.size();
        }

    } // namespace Graph
} // namespace SushiAI

/**************************************************************************/
/* buffer_update_recorder.cpp                                             */
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
 * @file buffer_update_recorder.cpp
 * @brief Implements the binding of buffer updates and the recorder that blends them.
 * @author Mustafa Garip
 */

#include <SushiAI/train/buffer_update_recorder.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/tensor_shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Counts the device tasks one slot records: a scale and an axpy. */
            constexpr std::size_t K_TASKS_PER_SLOT = 2;
        } // namespace

        Result<std::vector<BufferUpdateSlot>> bind_buffer_updates(
            const NN::BufferUpdateList& updates, NN::ModelState& state,
            const Graph::TensorPool& pool)
        {
            std::vector<BufferUpdateSlot> slots;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const SushiRuntime::span<const NN::BufferUpdate> list = updates.updates();
                slots.reserve(list.size());

                for (const NN::BufferUpdate& update : list)
                {
                    SA_FATAL_IF(!pool.is_bound(update.source),
                                "bind_buffer_updates: value {} is not bound; materialise the "
                                "graph before binding its updates",
                                update.source);

                    const Result<SushiBLAS::Tensor*> target = state.tensor(update.target);
                    SA_FAIL_IF(!target.has_value(), StatusCode::Incompatible,
                               "bind_buffer_updates: the state has no entry for state {}",
                               static_cast<std::uint64_t>(update.target));

                    const SushiBLAS::Tensor& source = pool.tensor(update.source);
                    const SushiBLAS::Tensor& buffer = *target.value();
                    const Shape buffer_shape = shape_of(buffer);
                    const Shape source_shape = shape_of(source);
                    SA_FAIL_IF(buffer_shape != source_shape || buffer.dtype != source.dtype,
                               StatusCode::Incompatible,
                               "bind_buffer_updates: the buffer of state {} is {} {}, but its "
                               "source, value {}, is {} {}",
                               static_cast<std::uint64_t>(update.target),
                               dtype_name(buffer.dtype), buffer_shape.to_string(),
                               update.source,
                               dtype_name(source.dtype), source_shape.to_string());

                    slots.push_back(BufferUpdateSlot{buffer, source, update.momentum,
                                                     update.source_scale});
                }
                return Status();
            }));

            return std::move(slots);
        }

        BufferUpdateRecorder::BufferUpdateRecorder(std::vector<BufferUpdateSlot> slots) noexcept
            : slots_(std::move(slots))
        {

        }

        Result<std::size_t> BufferUpdateRecorder::record_into_step(
            SushiBLAS::TaskRecorder& recorder)
        {
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::BLASOps blas(recorder);
                for (BufferUpdateSlot& slot : slots_)
                {
                    blas.scal(1.0 - slot.momentum, slot.target);
                    blas.axpy(slot.momentum * slot.source_scale, slot.source, slot.target);
                }
                return Status();
            }));

            return slots_.size() * K_TASKS_PER_SLOT;
        }

        std::size_t BufferUpdateRecorder::slot_count() const noexcept
        {
            return slots_.size();
        }

    } // namespace Train
} // namespace SushiAI

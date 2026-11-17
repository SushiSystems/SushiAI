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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file buffer_update_recorder.cpp
 * @brief Implements the binding of buffer updates to the tensors they touch.
 * @author Mustafa Garip
 */

#include <SushiAI/train/buffer_update_recorder.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<std::vector<BufferUpdateSlot>> bind_buffer_updates(
            const NN::BufferUpdateList& updates, NN::ModelState& state,
            const Graph::TensorPool& pool) noexcept
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

    } // namespace Train
} // namespace SushiAI

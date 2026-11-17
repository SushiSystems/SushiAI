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
 * @brief Implements the recorder that blends running statistics into their buffers.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/buffer_update_recorder.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Counts the device tasks one slot records: a scale and an axpy. */
            constexpr std::size_t K_TASKS_PER_SLOT = 2;
        } // namespace

        BufferUpdateRecorder::BufferUpdateRecorder(std::vector<BufferUpdateSlot> slots) noexcept
            : slots_(std::move(slots))
        {

        }

        Result<std::size_t> BufferUpdateRecorder::record_into_step(const StepRecording& recording)
        {
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::BLASOps blas(recording.tasks());
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

    } // namespace Graph
} // namespace SushiAI

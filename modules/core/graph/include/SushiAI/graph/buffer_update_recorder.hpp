/**************************************************************************/
/* buffer_update_recorder.hpp                                             */
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
 * @file buffer_update_recorder.hpp
 * @brief Declares BufferUpdateSlot and BufferUpdateRecorder, which blend a trace's running
 *        statistics into their buffers on every replay.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Holds one running-statistic update as the two tensors it touches. */
        struct BufferUpdateSlot
        {
            /** @brief The buffer the update writes; it shares storage with the model state. */
            SushiBLAS::Tensor target;

            /** @brief The batch value the update reads; it shares storage with the pool. */
            SushiBLAS::Tensor source;

            /** @brief The weight of the source against the buffer's current content. */
            double momentum = 0.0;

            /** @brief The factor the source is scaled by before it is blended in. */
            double source_scale = 1.0;
        };

        /**
         * @brief Records target = (1 - momentum) * target + momentum * source_scale * source
         *        for every slot, as device work behind a compiled step.
         */
        class BufferUpdateRecorder final : public StepRecorder
        {
            public:
                /**
                 * @brief Takes the slots that every replay blends.
                 * @param slots The slots to blend; an empty list records nothing.
                 */
                explicit BufferUpdateRecorder(std::vector<BufferUpdateSlot> slots) noexcept;

                /**
                 * @brief Records two tasks per slot: a scale of the target, then an axpy of the
                 *        source into it.
                 * @return Twice slot_count(), or the failure of a recorded call.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const StepRecording& recording) override;

                /** @brief Returns how many slots this recorder blends. */
                [[nodiscard]] std::size_t slot_count() const noexcept;

            private:
                /** @brief The slots, in the order they were given; each replay visits them all. */
                std::vector<BufferUpdateSlot> slots_;
        };

    } // namespace Graph
} // namespace SushiAI

/**************************************************************************/
/* step_scalars.hpp                                                       */
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
 * @file step_scalars.hpp
 * @brief The per-step numbers an optimizer's kernel reads instead of baking in.
 *
 * @section problem The problem this solves
 *
 * A recorded SushiBLAS call captures its scalars **by value**. AdamW's bias
 * corrections change every step by definition (1 - beta^t), so an update
 * spelled as `scal`/`axpy` cannot be compiled once and replayed: the numbers
 * are frozen into the captured lambda at record time. That single fact is why
 * the optimizer used to sit outside the compiled plan and pay its own graph
 * analysis and its own flush every step.
 *
 * A **device pointer**, on the other hand, is capturable by value and stays
 * valid for ever, and the bytes it points at may change between replays without
 * anything being recompiled. So the fused update captures one pointer and reads
 * its scalars from index 0..N-1, and this class owns that buffer.
 *
 * @section why_float Why the buffer is float32 even for a float64 parameter
 *
 * Not an oversight, and not a precision compromise introduced here: the
 * separate-op path narrows every one of these scalars to @c float before it
 * hands it to SushiBLAS (see the @c static_cast<float> block in adamw.cpp).
 * Storing them as float reproduces that bit for bit, which is what lets the
 * fused and separate paths be compared with @c == rather than a tolerance. A
 * wider buffer would be a silent numerical change wearing the clothes of an
 * improvement.
 *
 * @section why_no_edge Why no dependency edge is needed, and what that costs
 *
 * The recorded kernels read this buffer without declaring it as a task input,
 * so SushiRuntime's dependency tracker knows nothing about it. That is sound
 * for exactly one access pattern and no other:
 *
 * - It is written **only from the host**, in @c IStepRecorder::before_replay(),
 *   which runs after the previous replay has been waited on and before the next
 *   is dispatched. No kernel can be reading it at that moment.
 * - It is **never written by a device task**. A kernel that filled it would
 *   race with every consumer, because there is no edge to order them.
 *
 * Keeping the update to it on the host is also simply cheaper than the
 * alternative the fusion work considered - a one-element kernel per step - and
 * costs no round trip that the loop was not already paying: the trainer waits
 * for the device every step to read the loss.
 *
 * Device lane: this header reaches SushiBLAS.
 */

#include <cstddef>
#include <span>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @class StepScalars
         * @brief A tiny device-resident float vector, rewritten from the host per step.
         *
         * Non-copyable: the whole point of the object is the stability of the
         * one address its kernels captured, and a copy would suggest there can
         * be two.
         */
        class StepScalars
        {
            public:
                /**
                 * @brief Allocates @p count floats of shared USM and zeroes them.
                 *
                 * Zeroed here rather than left undefined so a kernel recorded
                 * before the first @c write() reads a defined value instead of
                 * whatever the allocator last held.
                 *
                 * @param engine The engine whose context owns the allocator.
                 * @param count How many scalars; must be positive.
                 * @throws Error If @p count is zero or the allocation fails.
                 */
                StepScalars(SushiBLAS::Engine& engine, std::size_t count);

                StepScalars(const StepScalars&) = delete;
                StepScalars& operator=(const StepScalars&) = delete;

                /**
                 * @brief The address a device kernel captures by value.
                 *
                 * Stable for the life of this object, which is what makes the
                 * capture legal across replays.
                 *
                 * @return A pointer to element 0 of the buffer.
                 */
                const float* device_pointer() const noexcept { return data_; }

                /**
                 * @brief Overwrites the buffer from the host.
                 *
                 * Call only from @c IStepRecorder::before_replay() or somewhere
                 * else that is provably not concurrent with a reader - see the
                 * file comment.
                 *
                 * @param values Exactly @c count() values, in index order.
                 * @throws Error If @p values is not exactly @c count() long.
                 */
                void write(std::span<const float> values);

                /** @brief How many scalars the buffer holds. */
                std::size_t count() const noexcept { return count_; }

            private:
                SushiBLAS::Tensor buffer_;
                float* data_ = nullptr;
                std::size_t count_ = 0;
        };

    } // namespace Optim
} // namespace SushiAI

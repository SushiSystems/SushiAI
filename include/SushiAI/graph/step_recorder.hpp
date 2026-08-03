/**************************************************************************/
/* step_recorder.hpp                                                      */
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
 * @file step_recorder.hpp
 * @brief The seam that lets non-IR device work join the compiled step.
 *
 * @c CompiledStep lowers an IR graph, compiles it once and replays it. Some
 * device work in a training step has no IR node behind it - the optimizer's
 * parameter update is the whole reason this interface exists - and used to pay
 * for that by recording and flushing a graph of its own every step. That is a
 * second (and for AdamW a third) graph analysis per step over work that never
 * changes shape.
 *
 * A recorder is anything that can put its work into the engine's builder
 * *once*, before the single @c compile(), and then keep its per-step variation
 * somewhere the compiled nodes can re-read without being recompiled. The
 * ordering against the lowered graph needs no help: every SushiBLAS call
 * declares the byte ranges it reads and writes, so SushiRuntime's dependency
 * tracker already sees "the update reads the gradient the backward pass wrote"
 * and "the update writes the parameter the forward pass read" and orders them.
 *
 * @section contract The contract, in the order the calls happen
 *
 * 1. @c record_into_step() - exactly once, from CompiledStep's constructor,
 *    after the graph is lowered and before it is compiled.
 * 2. @c before_replay() - once per @c run(), before the plan is dispatched.
 *    This is where anything that varies per step is refreshed. It may not
 *    record device work: the builder is drained and the plan is already fixed.
 * 3. @c after_replay() - once per @c run(), after the device has been waited
 *    on.
 *
 * @section invariance What "replay-invariant" has to mean
 *
 * A recorder may only join the step if the work it records is identical on
 * every replay: same tasks, same buffers, same kernels. Anything that changes
 * has to travel through memory the kernels read at dispatch time rather than
 * through a value baked into a captured lambda - see
 * @c SushiAI::Optim::StepScalars for how the optimizers do it. A recorder that
 * cannot promise that must not be handed to CompiledStep; the promise itself is
 * asked for by the caller (Trainer asks @c Optim::Optimizer::is_replay_invariant())
 * rather than being assumed here.
 *
 * Host lane: this header names no SushiBLAS and no SYCL type, so an
 * implementer that wants to stay off the device compiler can.
 */

#include <cstddef>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class IStepRecorder
         * @brief Device work recorded into the compiled step rather than flushed alone.
         *
         * See the file comment for the call order and for what a recorder has
         * to guarantee about its own replay-invariance.
         */
        class IStepRecorder
        {
            public:
                virtual ~IStepRecorder() = default;

                /**
                 * @brief Records this work into the engine's task-graph builder.
                 *
                 * Called exactly once, before the step's single @c compile().
                 * Nothing is executed here; the tasks simply join the plan.
                 *
                 * @return How many device tasks were recorded, so the step can
                 *         report a truthful operation count.
                 */
                virtual std::size_t record_into_step() = 0;

                /**
                 * @brief Refreshes whatever varies between replays.
                 *
                 * Runs on the host, immediately before the plan is dispatched
                 * and therefore after the previous replay was waited on - which
                 * is what makes writing into a buffer the recorded kernels read
                 * safe without a dependency edge. Records no device work.
                 */
                virtual void before_replay() {}

                /** @brief Runs after the replay has completed and been waited on. */
                virtual void after_replay() {}
        };

    } // namespace Graph
} // namespace SushiAI

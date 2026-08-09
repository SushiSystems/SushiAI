/**************************************************************************/
/* optimizer.hpp                                                          */
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
 * @file optimizer.hpp
 * @brief The parameter-update seam.
 *
 * @section one_launch One launch per parameter, not eleven
 *
 * Spelled as separate SushiBLAS calls, an AdamW update is about eleven task
 * launches per parameter per step - decay the parameter, scale each moment,
 * accumulate each moment, correct, add epsilon, divide, step - and every one of
 * them re-reads and rewrites state the one before it just had in registers. For
 * an eight-tensor model that is ~88 launches for the optimizer against ~31 for
 * the entire forward and backward graph. Measured in SushiBLAS, collapsing a
 * five-tensor update into one kernel ran 16.2x faster at n=4096 and 22.2x at
 * n=65536, with per-task cost essentially unchanged: the win is issuing fewer
 * tasks, not doing cheaper arithmetic.
 *
 * So both optimizers default to @c UpdateKernel::FUSED, which is one
 * @c Dispatch::execute_zip_inplace per parameter - one launch, one pass over
 * memory, and no scratch buffers at all because the temporaries live in
 * registers. @c UpdateKernel::SEPARATE keeps the original op sequence reachable,
 * because a fused kernel that is quietly wrong about the arithmetic is a much
 * more expensive bug than a slow one; the two are compared byte for byte in
 * @c tests/functional/regression/test_fused_optimizer.cpp.
 *
 * @section in_graph Why the update can now be part of the compiled step
 *
 * It used not to be, and the recorded reason was real: a per-step scalar baked
 * into a recorded node is a compile-time constant, AdamW's bias corrections are
 * per-step scalars by definition (1 - beta^t changes every step), and SushiBLAS
 * takes its alphas by value - so a compiled @c axpy could never be handed a
 * number that changes between replays.
 *
 * Fusing removes the obstacle rather than working around it. One functor
 * captures one **device pointer** to a small scalar buffer and reads the step's
 * numbers from it (@c StepScalars). A pointer is capturable by value and the
 * bytes behind it may change between replays, so the recorded work becomes
 * launch-invariant - which is precisely the condition for it to be recorded
 * into the same @c TaskGraph as the forward and backward passes and replayed
 * with them. @c Optimizer therefore implements @c Graph::IStepRecorder, and
 * @c Trainer folds it into the plan by default. @c is_replay_invariant() is the
 * promise; a SEPARATE-kernel optimizer answers false and is stepped on its own.
 *
 * @section zero_grad There is no zero_grad()
 *
 * Gradients are written, not accumulated into: every gradient buffer in the
 * step is produced by an operation with beta = 0, and the tiered accumulation
 * in autograd/tape.cpp overwrites the first contributor before adding the rest. A
 * @c zero_grad() would be a no-op, and a no-op named after a real PyTorch
 * obligation is worse than its absence - it invites the reader to believe
 * something is being cleared.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/optim/loss_scaler.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @enum UpdateKernel
         * @brief How an optimizer spells its per-parameter update on the device.
         */
        enum class UpdateKernel
        {
            /**
             * @brief One fused kernel per parameter; the default.
             *
             * Every temporary stays in a register, so the update is one launch
             * and one pass over memory, and it is replay-invariant - which is
             * what lets it be compiled into the training step.
             */
            FUSED,

            /**
             * @brief The original sequence of separate SushiBLAS calls.
             *
             * Kept reachable as the reference the fused kernel is checked
             * against, and as the fallback if a device ever mis-compiles the
             * fused functor. It bakes the step's scalars into the recorded
             * tasks, so it cannot join the compiled plan.
             */
            SEPARATE,
        };

        /**
         * @struct ParameterSlot
         * @brief One parameter and its gradient, as the optimizer sees them.
         *
         * Tensors by value: a @c SushiBLAS::Tensor is a handle that shares its
         * Storage, so a copy costs a reference-count bump and removes any
         * dependence on where the pool keeps its map entries.
         */
        struct ParameterSlot
        {
            /** @brief The parameter's dotted name, e.g. "0.weight". */
            std::string name;

            /** @brief The parameter itself; updated in place. */
            SushiBLAS::Tensor parameter;

            /** @brief Its gradient, rewritten by every step. */
            SushiBLAS::Tensor gradient;
        };

        /**
         * @class Optimizer
         * @brief What every update rule has in common.
         *
         * Owns the parameter list and the step counter; subclasses own the
         * state buffers and the arithmetic. Non-copyable, because the state
         * buffers a subclass allocates are tied to these exact parameters.
         *
         * It is a @c Graph::IStepRecorder because an update whose scalars live
         * in device memory is replay-invariant work like any other, and the
         * whole benefit of fusing it is only collected once it stops being
         * flushed on its own - see the file comment.
         */
        class Optimizer : public Graph::IStepRecorder
        {
            public:
                /**
                 * @brief Binds an optimizer to a fixed set of parameters.
                 * @param engine The engine the update is recorded on; must outlive this object.
                 * @param parameters The parameters to update, with their gradients.
                 * @throws Error If a slot's parameter and gradient disagree in
                 *         shape or dtype, or the element type is not a real float.
                 */
                Optimizer(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters);

                ~Optimizer() override = default;

                Optimizer(const Optimizer&) = delete;
                Optimizer& operator=(const Optimizer&) = delete;

                /**
                 * @brief Applies one update to every parameter, standalone.
                 *
                 * Refreshes the step's scalars, records the whole update,
                 * executes it, waits, and advances the counter - so on return
                 * the parameters are readable from the host. This is the path
                 * for an optimizer that is *not* part of a compiled step; when
                 * it is, @c Graph::CompiledStep drives the same three phases
                 * around its own replay instead.
                 *
                 * Not virtual: the phase order is the contract, and it is the
                 * same one @c IStepRecorder states. Subclasses supply the
                 * phases, not the sequence.
                 *
                 * @throws Error If this optimizer has already been recorded
                 *         into a compiled step, which applies the update on
                 *         every replay - stepping it again would apply it twice
                 *         per batch, and the loop would happily converge at
                 *         double the learning rate.
                 */
                void step();

                /**
                 * @brief Whether the work this optimizer records survives a replay unchanged.
                 *
                 * True only when every number that changes between steps travels
                 * through device memory rather than through a value captured into
                 * a recorded task - which is what @c UpdateKernel::FUSED
                 * arranges. A caller that wants to fold the update into a
                 * compiled plan must ask this first; folding one that answers
                 * false would freeze step one's bias corrections into every step
                 * that followed, and the loop would still converge.
                 *
                 * @return True when the update may be recorded into a compiled step.
                 */
                virtual bool is_replay_invariant() const noexcept = 0;

                /** @brief How many parameters this optimizer updates. */
                std::size_t parameter_count() const noexcept { return parameters_.size(); }

                /** @brief How many updates have been applied; 0 before the first step(). */
                std::uint64_t step_count() const noexcept { return step_; }

                /** @brief The parameters, for inspection. */
                const std::vector<ParameterSlot>& parameters() const noexcept { return parameters_; }

                /**
                 * @brief Attaches a gradient loss scaler to this optimizer.
                 *
                 * The scaler's screens have to be recorded *before* the updates
                 * and its state machine advanced *after* them, and both are
                 * orderings a caller could get wrong silently - so the optimizer
                 * owns the sequence rather than documenting it. See
                 * @c optim/loss_scaler.hpp.
                 *
                 * @param scaler The scaler, or null to detach; must outlive this
                 *        optimizer. It is not owned: the same scaler backs a
                 *        graph value the training graph already holds.
                 * @throws Error If this optimizer's update is not
                 *         replay-invariant. The skip is a branch inside the
                 *         fused kernel and the separate spelling has no way to
                 *         express one, so a scaled run on the separate path
                 *         would unscale correctly and then apply every
                 *         overflowed step anyway.
                 */
                void set_loss_scaler(LossScaler* scaler);

                /** @brief The scaler attached to this optimizer, or null. */
                LossScaler* loss_scaler() const noexcept { return scaler_; }

                /**
                 * @brief Refreshes this step's numbers into device memory.
                 *
                 * @c final, and it is the *sequence* that is final rather than
                 * the work: the scaler's per-step publication has to happen
                 * alongside the optimizer's own, and a subclass that overrode
                 * this would silently drop it. Subclasses supply
                 * @ref write_step_scalars instead - which is still their whole
                 * job, since this adds one call around it.
                 */
                void before_replay() final;

                /**
                 * @brief Advances the state machine, then the step counter.
                 *
                 * @c final for the same reason, plus one that is sharper: the
                 * counter must **not** advance on a skipped step. SGD's
                 * @c FIRST_STEP flag is derived from it and initialises the
                 * momentum buffer, which is allocated but not zeroed - so a
                 * skipped first step followed by an advanced counter would have
                 * the second step read an uninitialised buffer and blend it into
                 * the momentum. AdamW's bias corrections have the milder version
                 * of the same problem: correcting for a step whose moments never
                 * moved.
                 */
                void after_replay() final;

                /**
                 * @brief Records the update into the engine's builder, for a compiled step.
                 *
                 * @c final, and the only caller is a @c Graph::CompiledStep:
                 * @c step() reaches the subclass hook directly. That is what
                 * makes @c is_in_compiled_step() a fact rather than a guess,
                 * and it is why calling both can be refused.
                 *
                 * @return How many device tasks were recorded.
                 */
                std::size_t record_into_step() final;

                /**
                 * @brief Whether a compiled step has taken ownership of this update.
                 *
                 * True from the moment a @c Graph::CompiledStep recorded it.
                 */
                bool is_in_compiled_step() const noexcept { return in_compiled_step_; }

            protected:
                /**
                 * @brief Records one update for every parameter; the subclass's job.
                 *
                 * Records only - nothing is executed and nothing is waited on.
                 *
                 * @return How many device tasks were recorded.
                 */
                virtual std::size_t record_update() = 0;

                /**
                 * @brief The scaler's screens and then the update, in that order.
                 *
                 * The one place the two are sequenced, so both entry points -
                 * @c step() and @c record_into_step() - get the same order and
                 * neither can be given the other one by a later edit.
                 *
                 * @return How many device tasks were recorded, screens included.
                 */
                std::size_t record_all();

                /**
                 * @brief Writes this step's scalars into device memory; the subclass's job.
                 *
                 * Runs where @c IStepRecorder::before_replay() does and under
                 * the same rules: host only, no device work, after the previous
                 * replay was waited on. Pure rather than defaulted because
                 * inheriting an empty body is a silent failure - the update
                 * would read whatever the scalar buffer last held and apply step
                 * one's bias corrections for ever, converging the whole time.
                 */
                virtual void write_step_scalars() = 0;

                /**
                 * @brief What the gradient must be multiplied by before the update.
                 *
                 * Exactly 1 when no scaler is attached, so the unscale is
                 * `x * 1.0f` - the identity for every float including the signed
                 * zeros, which is what keeps an unscaled run byte-identical to
                 * one built before loss scaling existed.
                 *
                 * @return The reciprocal of the current loss scale.
                 */
                float inverse_loss_scale() const noexcept;

                /**
                 * @brief The count buffer an update kernel gates on, or null.
                 *
                 * Null when no scaler is attached, and a kernel must test it -
                 * a null capture is how "there is nothing to skip for" is
                 * spelled, and it is uniform across the whole launch.
                 *
                 * @return Element @c i is parameter @c i's non-finite count.
                 */
                const std::int64_t* gradient_gate() const noexcept;

                /** @brief Runs the recorded update and waits for the device. */
                void flush();

                /** @brief Allocates a state buffer shaped like @p like, with its own Storage. */
                SushiBLAS::Tensor allocate_state(const SushiBLAS::Tensor& like) const;

                /** @brief The engine the update is recorded on. */
                SushiBLAS::Engine& engine() const noexcept { return *engine_; }

                /** @brief The parameters being updated. */
                std::vector<ParameterSlot>& slots() noexcept { return parameters_; }

                /** @brief Bumps the step counter; call once per step(). */
                void advance() noexcept { ++step_; }

            private:
                SushiBLAS::Engine* engine_;
                std::vector<ParameterSlot> parameters_;
                std::uint64_t step_ = 0;
                bool in_compiled_step_ = false;
                LossScaler* scaler_ = nullptr;
        };

    } // namespace Optim
} // namespace SushiAI

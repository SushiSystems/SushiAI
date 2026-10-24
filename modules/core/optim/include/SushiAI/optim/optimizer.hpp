/**************************************************************************/
/* optimizer.hpp                                                          */
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
 * @file optimizer.hpp
 * @brief Declares abstract base optimizer class and common parameter slot types.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/optim/gradient_clip.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/run_state.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @enum UpdateKernel
         * @brief How an optimizer spells its per-parameter update on the device.
         */
        enum class UpdateKernel : std::uint8_t
        {
            /**
             * @brief One fused kernel per parameter; the default.
             *
             * Every temporary stays in a register, so the update is one launch
             * and one pass over memory, and it is replay-invariant - which is
             * what lets it be compiled into the training step.
             */
            Fused,

            /**
             * @brief The original sequence of separate SushiBLAS calls.
             *
             * Kept reachable as the reference the fused kernel is checked
             * against, and as the fallback if a device ever mis-compiles the
             * fused functor. It bakes the step's scalars into the recorded
             * tasks, so it cannot join the compiled plan.
             */
            Separate,
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

            /** @brief Whether weight decay applies; an optimizer's decay_all overrides false. */
            bool decays = true;
        };

        /**
         * @brief Checks that each slot's parameter and gradient agree and hold a real float.
         * @param slots The slots to check, in the order they will be updated.
         * @return Success, or an InvalidDescription status naming the first failing slot.
         */
        [[nodiscard]] Status validate_slots(SushiRuntime::span<const ParameterSlot> slots);

        /**
         * @class Optimizer
         * @brief Common base class providing parameter update recording and lifecycle hooks.
         */
        class Optimizer : public Graph::StepRecorder
        {
            public:
                ~Optimizer() override = default;

                Optimizer(const Optimizer&) = delete;
                Optimizer& operator=(const Optimizer&) = delete;
                Optimizer(Optimizer&&) = delete;
                Optimizer& operator=(Optimizer&&) = delete;

                /**
                 * @brief Applies one update step standalone to every registered parameter.
                 * @return Success, or the status of the recording or the run that failed.
                 */
                [[nodiscard]] Status step();

                /**
                 * @brief Indicates whether the recorded update survives graph replays unchanged.
                 * @return True if updates can be embedded directly into compiled training steps.
                 */
                virtual bool is_replay_invariant() const noexcept = 0;

                /** @brief How many parameters this optimizer updates. */
                std::size_t parameter_count() const noexcept { return parameters_.size(); }

                /** @brief How many updates have been applied; 0 before the first step(). */
                std::uint64_t step_count() const noexcept { return step_; }

                /** @brief Returns the name a training file and a checkpoint give this type. */
                [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;

                /**
                 * @brief Lists what this optimizer keeps across updates, under stable names.
                 * @return Tensors that share this optimizer's storage and the scalars as they
                 *         are now, or the failure of the allocation of the lists.
                 */
                [[nodiscard]] Result<RunState> run_state();

                /**
                 * @brief Refuses run state that an optimizer of another type wrote.
                 * @return Incompatible, naming @p owner and type_name(), unless they are equal.
                 */
                [[nodiscard]] Status check_state_owner(std::string_view owner) const;

                /**
                 * @brief Restores the scalars of run_state(), strictly in both directions.
                 * @param scalars Exactly the scalars run_state() lists, in any order.
                 * @return Incompatible, naming the scalar, for one that is missing, surplus,
                 *         repeated or of the other kind, and for a value that cannot be held.
                 */
                [[nodiscard]] Status restore_scalars(SushiRuntime::span<const StateScalar> scalars);

                /** @brief Returns the step size the next update applies. */
                [[nodiscard]] virtual double learning_rate() const noexcept = 0;

                /**
                 * @brief Sets the step size of every update from the next one on.
                 * @param rate Read by the next replay, compiled or not; a rate that is
                 *             negative or not finite stops the process.
                 */
                void set_learning_rate(double rate);

                /** @brief The parameters, for inspection. */
                const std::vector<ParameterSlot>& parameters() const noexcept
                {
                    return parameters_;
                }

                /**
                 * @brief Attaches a gradient loss scaler to this optimizer.
                 * @param scaler Loss scaler to attach, or nullptr to detach.
                 * @return Success, or an InvalidDescription status if the scaler cannot be
                 *         attached to this optimizer.
                 */
                [[nodiscard]] Status set_loss_scaler(LossScaler* scaler);

                /** @brief The scaler attached to this optimizer, or null. */
                LossScaler* loss_scaler() const noexcept { return scaler_; }

                /**
                 * @brief Clips every later update to a global gradient norm of @p max_norm.
                 * @pre No step has recorded this optimizer's update yet; a later call is fatal.
                 * @return Success, or InvalidDescription for a norm that is not positive and
                 *         finite, for a separate update kernel, for no parameter, or for
                 *         gradients of more than one element type; or the allocation's failure.
                 */
                [[nodiscard]] Status enable_gradient_clip(double max_norm);

                /**
                 * @brief Sets the norm the next update is clipped to.
                 * @pre enable_gradient_clip() succeeded, and @p max_norm is positive and
                 *      finite; either broken stops the process.
                 */
                void set_clip_norm(double max_norm);

                /** @brief The clip of this optimizer's gradients, or null when it has none. */
                const GradientClip* gradient_clip() const noexcept { return clip_.get(); }

                /**
                 * @brief Refreshes this step's numbers into device memory.
                 */
                void before_replay() final;

                /**
                 * @brief Advances loss scale state machine followed by step counter.
                 */
                void after_replay() final;

                /**
                 * @brief Records parameter update operations into a compiled execution plan.
                 * @return Number of device tasks recorded, or the failure of the recording.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const Graph::StepRecording& recording) final;

                /**
                 * @brief Whether a compiled step has taken ownership of this update.
                 *
                 * True from the moment a @c Graph::CompiledStep recorded it.
                 */
                bool is_in_compiled_step() const noexcept { return in_compiled_step_; }

            protected:
                /**
                 * @brief Stores the engine and the slots; checks nothing.
                 * @pre validate_slots(parameters) has returned success.
                 */
                Optimizer(SushiBLAS::Engine& engine,
                          std::vector<ParameterSlot> parameters) noexcept;

                /**
                 * @brief Records one update for every parameter; the subclass's job.
                 *
                 * Records only: nothing is executed and nothing is waited on. A SushiBLAS
                 * exception may escape; record_all() is the only caller and guards it.
                 *
                 * @return How many device tasks were recorded.
                 */
                virtual std::size_t record_update(SushiBLAS::TaskRecorder& recorder) = 0;

                /**
                 * @brief Records loss scaler screens followed by parameter updates.
                 * @param recorder The recorder the tasks are added to.
                 * @return Total number of device tasks recorded, or the failure of the recording.
                 */
                [[nodiscard]] Result<std::size_t> record_all(SushiBLAS::TaskRecorder& recorder);

                /**
                 * @brief Writes current step scalars into device memory before execution.
                 */
                virtual void write_step_scalars() = 0;

                /** @brief Stores a step size set_learning_rate() has checked. */
                virtual void store_learning_rate(double rate) noexcept = 0;

                /**
                 * @brief Appends this optimizer's state tensors, in slot order.
                 *
                 * May throw when the list cannot grow; run_state() is the only caller and
                 * guards it.
                 */
                virtual void append_state_tensors(std::vector<StateTensor>& tensors) = 0;

                /**
                 * @brief Returns multiplier for unscaling gradients before updates.
                 * @return Reciprocal of current loss scale factor.
                 */
                float inverse_loss_scale() const noexcept;

                /**
                 * @brief Returns pointer to device non-finite count buffer, or nullptr.
                 * @return Pointer to per-parameter non-finite gradient counts.
                 */
                const std::int64_t* gradient_gate() const noexcept;

                /**
                 * @brief Returns the one-element factor an update multiplies its gradient by.
                 * @return The clip's scale, of the gradients' element type, or null when no
                 *         clip is enabled; an update that reads it declares the read.
                 */
                const SushiBLAS::Tensor* clip_scale() const noexcept;

                /** @brief Allocates a state buffer shaped like @p like, with its own Storage. */
                [[nodiscard]] Result<SushiBLAS::Tensor> allocate_state(
                    const SushiBLAS::Tensor& like) const;

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

                /// @brief The global-norm clip; null until enable_gradient_clip() succeeds.
                std::unique_ptr<GradientClip> clip_;
        };

    } // namespace Optim
} // namespace SushiAI

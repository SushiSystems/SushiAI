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
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
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
                 * @brief Refreshes this step's numbers into device memory.
                 */
                void before_replay() final;

                /**
                 * @brief Advances loss scale state machine followed by step counter.
                 */
                void after_replay() final;

                /**
                 * @brief Records parameter update operations into a compiled execution plan.
                 * @param recorder The recorder the step's tasks are added to.
                 * @return Number of device tasks recorded, or the failure of the recording.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    SushiBLAS::TaskRecorder& recorder) final;

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

                /** @brief Runs the recorded update and waits for the device. */
                [[nodiscard]] Status flush();

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
        };

    } // namespace Optim
} // namespace SushiAI

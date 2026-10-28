/**************************************************************************/
/* adamw.hpp                                                              */
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
 * @file adamw.hpp
 * @brief Declares the AdamW optimizer with decoupled weight decay.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/step_scalars.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /** @brief Names an AdamW in a training file and in the run state of a checkpoint. */
        inline constexpr std::string_view K_ADAMW_TYPE = "adamw";

        /**
         * @struct AdamWOptions
         * @brief AdamW's hyperparameters, with the reference defaults.
         */
        struct AdamWOptions
        {
            /** @brief Step size. */
            double learning_rate = 1e-3;

            /** @brief First-moment decay. */
            double beta1 = 0.9;

            /** @brief Second-moment decay. */
            double beta2 = 0.999;

            /** @brief Denominator floor, added after the bias correction. */
            double eps = 1e-8;

            /** @brief Decoupled weight decay, applied multiplicatively before the step. */
            double weight_decay = 0.01;

            /** @brief Decays every slot, also one whose ParameterSlot::decays is false. */
            bool decay_all = false;

            /**
             * @brief One fused kernel per parameter, or the twelve separate ops.
             *
             * FUSED is the default and the shipping path; SEPARATE exists to be
             * compared against. See optimizer.hpp.
             */
            UpdateKernel kernel = UpdateKernel::Fused;
        };

        /**
         * @class AdamW
         * @brief The update above, applied to a fixed parameter set.
         */
        class AdamW final : public Optimizer
        {
            public:
                /**
                 * @brief Validates the slots and options, then allocates and zeroes the state.
                 * @return The optimizer, or the first failure of validation, of allocation or of
                 *         the run that zeroes the state.
                 */
                [[nodiscard]] static Result<std::unique_ptr<AdamW>> make(
                    SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                    AdamWOptions options = {}) noexcept;

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::Fused;
                }

                /** @brief Returns @ref K_ADAMW_TYPE. */
                [[nodiscard]] std::string_view type_name() const noexcept override
                {
                    return K_ADAMW_TYPE;
                }

                /** @brief The hyperparameters in force. */
                const AdamWOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                AdamWOptions& options() noexcept { return options_; }

                /** @brief Returns the step size the next update applies. */
                [[nodiscard]] double learning_rate() const noexcept override
                {
                    return options_.learning_rate;
                }

                /**
                 * @brief Returns parameter's first moment tensor for inspection.
                 * @param index Parameter slot index in registration order.
                 * @return First moment tensor reference.
                 * @pre index is less than parameter_count(); a larger index stops the process.
                 */
                const SushiBLAS::Tensor& first_moment(std::size_t index) const noexcept;

                /**
                 * @brief Parameter @p index's second moment, v.
                 * @param index Which parameter, in slot order.
                 * @return Its second moment.
                 * @pre index is less than parameter_count(); a larger index stops the process.
                 */
                const SushiBLAS::Tensor& second_moment(std::size_t index) const noexcept;

            protected:
                /** @brief Stores @p rate as the step size of the options. */
                void store_learning_rate(double rate) noexcept override
                {
                    options_.learning_rate = rate;
                }

                /** @brief Appends "<parameter>.first_moment" and ".second_moment" per slot. */
                void append_state_tensors(std::vector<StateTensor>& tensors) override;

                /**
                 * @brief Records the update for every parameter into @p recorder.
                 * @param recorder The recorder the tasks are added to; nothing is executed.
                 * @return One task per parameter when fused; twelve with decay
                 *         and eleven without, when not.
                 */
                std::size_t record_update(SushiBLAS::TaskRecorder& recorder) override;

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 */
                void write_step_scalars() override;

            private:
                struct State
                {
                    SushiBLAS::Tensor first;    /**< m, the first moment. */
                    SushiBLAS::Tensor second;   /**< v, the second moment. */

                    // SEPARATE only: registers in the fused kernel.
                    SushiBLAS::Tensor denom;    /**< The denominator, and g^2 before it. */
                    SushiBLAS::Tensor update;   /**< m / denom. */
                    SushiBLAS::Tensor ones;     /**< All ones, so eps can be added elementwise. */
                };

                /**
                 * @brief Takes ownership of state that make() has already allocated.
                 * @pre validate_slots(parameters) has returned success.
                 */
                AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                      AdamWOptions options, std::vector<State> state, StepScalars scalars) noexcept;

                /** @brief Records one fused kernel per parameter; returns the task count. */
                std::size_t record_fused(SushiBLAS::TaskRecorder& recorder);

                /** @brief Records the twelve separate ops per parameter; returns the task count. */
                std::size_t record_separate(SushiBLAS::TaskRecorder& recorder);

                AdamWOptions options_;
                std::vector<State> state_;
                StepScalars scalars_;
        };

    } // namespace Optim
} // namespace SushiAI

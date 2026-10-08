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

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/step_scalars.hpp>

namespace SushiAI
{
    namespace Optim
    {
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
                 * @brief Binds AdamW to @p parameters, allocates and zeroes its state.
                 * @param engine The engine the update is recorded on.
                 * @param parameters The parameters to update.
                 * @param options The hyperparameters.
                 * @throws Error If a beta is outside [0, 1) or eps is not positive.
                 */
                AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                      AdamWOptions options = {});

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::Fused;
                }

                /** @brief The hyperparameters in force. */
                const AdamWOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                AdamWOptions& options() noexcept { return options_; }

            protected:
                /**
                 * @brief Records the update for every parameter into the engine's builder.
                 * @return One task per parameter when fused; twelve with decay
                 *         and eleven without, when not.
                 */
                std::size_t record_update() override;

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 */
                void write_step_scalars() override;

            public:
                /**
                 * @brief Returns parameter's first moment tensor for inspection.
                 * @param index Parameter slot index in registration order.
                 * @return First moment tensor reference.
                 * @throws Error If index is out of range.
                 */
                const SushiBLAS::Tensor& first_moment(std::size_t index) const;

                /**
                 * @brief Parameter @p index's second moment, v.
                 * @param index Which parameter, in slot order.
                 * @return Its second moment.
                 * @throws Error If @p index is out of range.
                 */
                const SushiBLAS::Tensor& second_moment(std::size_t index) const;

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

                std::size_t record_fused();
                std::size_t record_separate();

                AdamWOptions options_;
                std::vector<State> state_;
                StepScalars scalars_;
        };

    } // namespace Optim
} // namespace SushiAI

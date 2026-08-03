/**************************************************************************/
/* sgd.hpp                                                                */
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
 * @file sgd.hpp
 * @brief Stochastic gradient descent, in PyTorch's formulation.
 *
 * @section formulation The formulation, stated because they differ
 *
 * "SGD with momentum" names at least three different update rules in common
 * use. This is PyTorch's, chosen so a learning rate transfers between the two
 * without a conversion:
 *
 * @code
 * g_t = grad + weight_decay * theta_{t-1}
 * b_t = g_t                              (t = 1)
 * b_t = momentum * b_{t-1} + (1 - dampening) * g_t   (t > 1)
 * g_t = g_t + momentum * b_t             (nesterov)
 * g_t = b_t                              (otherwise)
 * theta_t = theta_{t-1} - lr * g_t
 * @endcode
 *
 * Two details are easy to get wrong and both change the trajectory rather than
 * breaking anything:
 *
 * - The first step is @c b_1 = g_1, with no dampening applied. Applying it
 *   uniformly makes the first update (1 - dampening) times too small, which
 *   looks like a slightly slow start and nothing more.
 * - The decay is @e coupled: it is added to the gradient, so momentum carries
 *   it. That is what distinguishes this from AdamW's decoupled decay, and it
 *   is why the two need different weight-decay values for comparable
 *   regularisation.
 *
 * @section fusion One kernel, and where g_t went
 *
 * The default @c UpdateKernel::FUSED runs the whole recipe above as one
 * @c execute_zip_inplace per parameter over (theta, g) or (theta, g, b). The
 * staging buffer is gone from that path: @c g_t only ever exists between the
 * first line and the last, so it is a register, not an allocation.
 *
 * The two per-step decisions - whether this is step 1, and whether Nesterov is
 * on - are read from the device scalar buffer rather than branched on the host,
 * because a compiled task is recorded once and cannot be re-recorded with a
 * different shape. They are uniform across every work-item, so the branch is
 * free, and it is a branch rather than arithmetic on purpose: `b = 0*b + 1*g`
 * would turn a `-0.0` gradient into `+0.0` and a poisoned buffer into a NaN,
 * where `b = g` is exactly what the separate path's `copy` does.
 *
 * Whether momentum is used at all is structural, not per-step: it decides
 * whether the momentum buffer exists, and therefore how many operands the
 * kernel has. Turning it on after construction is refused rather than silently
 * ignored.
 */

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/step_scalars.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @struct SGDOptions
         * @brief SGD's hyperparameters, with PyTorch's defaults where they have one.
         */
        struct SGDOptions
        {
            /** @brief Step size. */
            double learning_rate = 1e-2;

            /** @brief Momentum coefficient; 0 disables the buffer entirely. */
            double momentum = 0.0;

            /** @brief Damping applied to the gradient from step 2 onward. */
            double dampening = 0.0;

            /** @brief Coupled L2 penalty, added to the gradient. */
            double weight_decay = 0.0;

            /** @brief Use Nesterov's look-ahead; requires momentum > 0 and dampening == 0. */
            bool nesterov = false;

            /**
             * @brief One fused kernel per parameter, or the separate ops.
             *
             * FUSED is the default and the shipping path; SEPARATE exists to be
             * compared against. See optimizer.hpp.
             */
            UpdateKernel kernel = UpdateKernel::FUSED;
        };

        /**
         * @class SGD
         * @brief The update rule above, applied to a fixed parameter set.
         */
        class SGD final : public Optimizer
        {
            public:
                /**
                 * @brief Binds SGD to @p parameters and allocates its state.
                 *
                 * The momentum buffer is allocated once here, not lazily on the
                 * first step, so the steady state allocates nothing and a
                 * failure to allocate happens at construction where it can be
                 * handled.
                 *
                 * @param engine The engine the update is recorded on.
                 * @param parameters The parameters to update.
                 * @param options The hyperparameters.
                 * @throws Error If nesterov is set without momentum, or with dampening.
                 */
                SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                    SGDOptions options = {});

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 * @throws Error If momentum was switched on after construction,
                 *         when there is no momentum buffer to switch it on for.
                 */
                void before_replay() override;

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::FUSED;
                }

                /** @brief The hyperparameters in force. */
                const SGDOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                SGDOptions& options() noexcept { return options_; }

            protected:
                /**
                 * @brief Records the update for every parameter into the engine's builder.
                 * @return One task per parameter when fused, four or five each when not.
                 */
                std::size_t record_update() override;

            private:
                struct State
                {
                    SushiBLAS::Tensor buffer;   /**< The momentum buffer b; absent when momentum is off. */
                    SushiBLAS::Tensor staged;   /**< SEPARATE only: g_t, kept off the graph's own gradient buffer. */
                };

                std::size_t record_fused();
                std::size_t record_separate();

                SGDOptions options_;
                std::vector<State> state_;
                StepScalars scalars_;

                /** @brief Whether a momentum buffer was allocated; fixed at construction. */
                bool has_momentum_ = false;
        };

    } // namespace Optim
} // namespace SushiAI

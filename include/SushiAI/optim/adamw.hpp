/**************************************************************************/
/* adamw.hpp                                                              */
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
 * @file adamw.hpp
 * @brief AdamW, with the two details that are usually wrong spelled out.
 *
 * @section rule The update
 *
 * With t counting from 1 on the first step:
 *
 * @code
 * theta <- theta * (1 - lr * weight_decay)          // decoupled, and FIRST
 * m     <- beta1 * m + (1 - beta1) * g
 * v     <- beta2 * v + (1 - beta2) * g^2
 * bc1   =  1 - beta1^t
 * bc2   =  1 - beta2^t
 * theta <- theta - (lr / bc1) * m / (sqrt(v / bc2) + eps)
 * @endcode
 *
 * @section trap1 The decay comes first, and it decays theta_{t-1}
 *
 * "Decoupled" means the penalty never enters the gradient and therefore never
 * enters m or v. Applying it after the gradient step decays theta_t instead of
 * theta_{t-1}, which is a different (and lr-dependent) regulariser. Ordering
 * it first is not a preference; it is what the AdamW paper specifies and what
 * makes the decay behave like the multiplicative shrinkage it is meant to be.
 *
 * @section trap2 eps is added AFTER dividing by sqrt(bc2)
 *
 * The denominator is @c sqrt(v / bc2) + eps, not @c sqrt(v + eps) / sqrt(bc2)
 * and not @c sqrt(v) / sqrt(bc2) + eps applied to an uncorrected v. Folding
 * eps inside the correction scales it by 1 / sqrt(bc2), and on step 1
 * bc2 = 1 - 0.999 = 1e-3, so sqrt(bc2) is about 0.0316 and the effective
 * epsilon is 31.6x too large. The first few updates are then damped by a
 * factor nobody asked for. It converges anyway, which is exactly why the bug
 * survives in real codebases - so the ordering is written out here and
 * asserted in the test suite rather than left to a reader's care.
 *
 * @section state State buffers
 *
 * m and v are allocated once, at construction, each with its own Storage - see
 * optimizer.cpp for why an arena would be silently wrong today. Their element
 * type follows the parameter's, which on the training path is FLOAT32
 * throughout; a mixed-type axpy is rejected by SushiBLAS, so matching is the
 * only option that type-checks, and half precision is refused outright by the
 * base class.
 *
 * The fused kernel needs nothing else: g^2, the denominator and m/denom exist
 * only inside one element's arithmetic and stay in registers.
 * @c UpdateKernel::SEPARATE spells the same recipe as twelve SushiBLAS calls
 * that have to hand those temporaries to each other through memory, so it -
 * and only it - also allocates the three scratch buffers, including a buffer of
 * ones whose sole purpose is to let an @c axpy add a constant.
 *
 * @section fusion The fused form, expression by expression
 *
 * The fused functor is written to reproduce the separate path's rounding
 * exactly, not merely its algebra:
 *
 * @code
 * theta = theta * decay                        // scal
 * m     = m * b1 + (1 - b1) * g                // scal then axpy
 * v     = v * b2 + (1 - b2) * (g * g)          // mul, scal, axpy
 * d     = sqrt(v * (1 / bc2)) + eps            // copy, scal, sqrt, axpy
 * theta = theta + (-lr / bc1) * (m / d)        // div then axpy
 * @endcode
 *
 * Each line's operand order matches the kernel it replaces (`x *= alpha` is
 * `x * alpha`, `y += alpha * x` is `y + (alpha * x)`), and the whole stack
 * builds with `-ffp-contract=off` through SushiRuntime's determinism contract,
 * so no `a * b + c` collapses into an FMA on one side of the comparison and not
 * the other. The `theta * decay` multiply is guarded by `decay != 1` for the
 * same reason the separate path guards it: to be the same operation, not an
 * equivalent one.
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
            UpdateKernel kernel = UpdateKernel::FUSED;
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

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 *
                 * The bias corrections are a function of the step counter, and
                 * everything else is a function of @c options(), so this also
                 * picks up a learning-rate schedule between replays without any
                 * recompilation.
                 */
                void before_replay() override;

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::FUSED;
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

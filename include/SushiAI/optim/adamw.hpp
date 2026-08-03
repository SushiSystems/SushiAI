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
 * m, v and three scratch buffers are allocated once, at construction, each
 * with its own Storage - see optimizer.cpp for why an arena would be silently
 * wrong today. Their element type follows the parameter's, which on the
 * training path is FLOAT32 throughout; a mixed-type axpy is rejected by
 * SushiBLAS, so matching is the only option that type-checks, and half
 * precision is refused outright by the base class.
 */

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/optim/optimizer.hpp>

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

                /** @brief Applies one update to every parameter, then waits. */
                void step() override;

                /** @brief The hyperparameters in force. */
                const AdamWOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                AdamWOptions& options() noexcept { return options_; }

            private:
                struct State
                {
                    SushiBLAS::Tensor first;    /**< m, the first moment. */
                    SushiBLAS::Tensor second;   /**< v, the second moment. */
                    SushiBLAS::Tensor denom;    /**< The denominator, and g^2 before it. */
                    SushiBLAS::Tensor update;   /**< m / denom. */
                    SushiBLAS::Tensor ones;     /**< All ones, so eps can be added elementwise. */
                };

                AdamWOptions options_;
                std::vector<State> state_;
        };

    } // namespace Optim
} // namespace SushiAI

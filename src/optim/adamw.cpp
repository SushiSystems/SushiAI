/**************************************************************************/
/* adamw.cpp                                                              */
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

#include <cmath>
#include <utility>
#include <SushiAI/optim/adamw.hpp>

namespace SushiAI
{
    namespace Optim
    {
        AdamW::AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                     AdamWOptions options)
            : Optimizer(engine, std::move(parameters)), options_(options)
        {
            SA_THROW_IF(options_.beta1 < 0.0 || options_.beta1 >= 1.0,
                        "AdamW beta1 must lie in [0, 1); got {}", options_.beta1);
            SA_THROW_IF(options_.beta2 < 0.0 || options_.beta2 >= 1.0,
                        "AdamW beta2 must lie in [0, 1); got {}", options_.beta2);
            SA_THROW_IF(options_.eps <= 0.0,
                        "AdamW eps must be positive; got {}", options_.eps);

            state_.reserve(slots().size());
            for (const ParameterSlot& slot : slots())
            {
                State state;
                state.first = allocate_state(slot.parameter);
                state.second = allocate_state(slot.parameter);
                state.denom = allocate_state(slot.parameter);
                state.update = allocate_state(slot.parameter);
                state.ones = allocate_state(slot.parameter);

                engine.random().constant(state.first, 0.0);
                engine.random().constant(state.second, 0.0);
                engine.random().constant(state.ones, 1.0);

                state_.push_back(std::move(state));
            }

            // The moments must be zero before the first step reads them, and
            // the ones buffer must be ones before it is used to add eps, so
            // these fills are run and waited on here rather than folded into
            // the first update.
            flush();
        }

        void AdamW::step()
        {
            const double t = static_cast<double>(step_count() + 1);
            const double bias_correction1 = 1.0 - std::pow(options_.beta1, t);
            const double bias_correction2 = 1.0 - std::pow(options_.beta2, t);

            const float beta1 = static_cast<float>(options_.beta1);
            const float beta2 = static_cast<float>(options_.beta2);
            const float one_minus_beta1 = static_cast<float>(1.0 - options_.beta1);
            const float one_minus_beta2 = static_cast<float>(1.0 - options_.beta2);
            const float eps = static_cast<float>(options_.eps);
            const float decay_factor =
                static_cast<float>(1.0 - options_.learning_rate * options_.weight_decay);
            const float inverse_correction2 = static_cast<float>(1.0 / bias_correction2);
            const float step_size =
                static_cast<float>(-options_.learning_rate / bias_correction1);

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                // 1. Decoupled decay, on theta_{t-1}, before anything else.
                if (decay_factor != 1.0f) engine().blas().scal(decay_factor, slot.parameter);

                // 2. m <- beta1 * m + (1 - beta1) * g
                engine().blas().scal(beta1, state.first);
                engine().blas().axpy(one_minus_beta1, slot.gradient, state.first);

                // 3. v <- beta2 * v + (1 - beta2) * g^2, borrowing denom for g^2.
                engine().elementwise().mul(slot.gradient, slot.gradient, state.denom);
                engine().blas().scal(beta2, state.second);
                engine().blas().axpy(one_minus_beta2, state.denom, state.second);

                // 4. denom <- sqrt(v / bc2) + eps. The correction is applied to
                //    v BEFORE the square root, and eps is added AFTER - see the
                //    header: folding eps inside inflates it 31.6x on step one.
                engine().blas().copy(state.second, state.denom);
                engine().blas().scal(inverse_correction2, state.denom);
                engine().elementwise().sqrt(state.denom);
                engine().blas().axpy(eps, state.ones, state.denom);

                // 5. theta <- theta - (lr / bc1) * m / denom
                engine().elementwise().div(state.first, state.denom, state.update);
                engine().blas().axpy(step_size, state.update, slot.parameter);
            }

            flush();
            advance();
        }

    } // namespace Optim
} // namespace SushiAI

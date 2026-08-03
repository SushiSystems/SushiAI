/**************************************************************************/
/* sgd.cpp                                                                */
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

#include <utility>
#include <SushiAI/optim/sgd.hpp>

namespace SushiAI
{
    namespace Optim
    {
        SGD::SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                 SGDOptions options)
            : Optimizer(engine, std::move(parameters)), options_(options)
        {
            SA_THROW_IF(options_.nesterov && options_.momentum <= 0.0,
                        "Nesterov momentum needs a positive momentum coefficient; got {}",
                        options_.momentum);
            SA_THROW_IF(options_.nesterov && options_.dampening != 0.0,
                        "Nesterov momentum is undefined with dampening; got {}",
                        options_.dampening);

            const bool needs_momentum = options_.momentum != 0.0;

            state_.reserve(slots().size());
            for (const ParameterSlot& slot : slots())
            {
                State state;
                state.staged = allocate_state(slot.parameter);
                if (needs_momentum) state.buffer = allocate_state(slot.parameter);
                state_.push_back(std::move(state));
            }
        }

        void SGD::step()
        {
            const float learning_rate = static_cast<float>(options_.learning_rate);
            const float momentum = static_cast<float>(options_.momentum);
            const float decay = static_cast<float>(options_.weight_decay);
            const float retained = static_cast<float>(1.0 - options_.dampening);
            const bool first = step_count() == 0;

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                // The gradient buffer belongs to the compiled step and is
                // rewritten by the next forward pass; staging keeps the decay
                // term out of it.
                engine().blas().copy(slot.gradient, state.staged);
                if (decay != 0.0f) engine().blas().axpy(decay, slot.parameter, state.staged);

                if (momentum != 0.0f)
                {
                    if (first)
                    {
                        // b_1 = g_1: no dampening on the first step, matching
                        // PyTorch. Damping it here makes the first update
                        // quietly too small.
                        engine().blas().copy(state.staged, state.buffer);
                    }
                    else
                    {
                        engine().blas().scal(momentum, state.buffer);
                        engine().blas().axpy(retained, state.staged, state.buffer);
                    }

                    if (options_.nesterov)
                        engine().blas().axpy(momentum, state.buffer, state.staged);
                    else
                        engine().blas().copy(state.buffer, state.staged);
                }

                engine().blas().axpy(-learning_rate, state.staged, slot.parameter);
            }

            flush();
            advance();
        }

    } // namespace Optim
} // namespace SushiAI

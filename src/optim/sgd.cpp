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

#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/optim/sgd.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /**
             * @enum Scalar
             * @brief Where each per-step number lives in the device buffer.
             *
             * The last two are conditions rather than coefficients. They live
             * here rather than in the functor's captures because a task
             * recorded into a compiled step is recorded once, and "is this step
             * one" is exactly the kind of thing that must not be frozen at
             * record time.
             */
            enum Scalar : std::size_t
            {
                NEG_LEARNING_RATE = 0,  /**< -lr, as the separate path's axpy takes it. */
                MOMENTUM,               /**< momentum. */
                WEIGHT_DECAY,           /**< Coupled L2 penalty. */
                RETAINED,               /**< 1 - dampening. */
                FIRST_STEP,             /**< 1 on the first step, 0 afterwards. */
                NESTEROV,               /**< 1 when Nesterov's look-ahead is on. */
                SCALAR_COUNT,
            };

            /** @brief The task name; proved below to be the OpID's own string. */
            constexpr const char* TASK_NAME = "sushiai.optim.sgd";

            static_assert(Ops::SGD_UPDATE.name == std::string_view(TASK_NAME),
                          "the fused update's task name must be the string its OpID hashes");
        }

        SGD::SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                 SGDOptions options)
            : Optimizer(engine, std::move(parameters)), options_(options),
              scalars_(engine, SCALAR_COUNT)
        {
            SA_THROW_IF(options_.nesterov && options_.momentum <= 0.0,
                        "Nesterov momentum needs a positive momentum coefficient; got {}",
                        options_.momentum);
            SA_THROW_IF(options_.nesterov && options_.dampening != 0.0,
                        "Nesterov momentum is undefined with dampening; got {}",
                        options_.dampening);

            has_momentum_ = options_.momentum != 0.0;
            const bool separate = options_.kernel == UpdateKernel::SEPARATE;

            state_.reserve(slots().size());
            for (const ParameterSlot& slot : slots())
            {
                State state;

                // g_t is a register in the fused kernel; only the separate
                // spelling has to hand it between launches through memory.
                if (separate) state.staged = allocate_state(slot.parameter);
                if (has_momentum_) state.buffer = allocate_state(slot.parameter);

                state_.push_back(std::move(state));
            }
        }

        void SGD::before_replay()
        {
            // Momentum decides how many operands the recorded kernel has, so it
            // is structural rather than per-step. Turning it on afterwards used
            // to dereference a momentum buffer that was never allocated; saying
            // so is better than either crashing or - in the fused path, which
            // simply would not have the operand - quietly training without it.
            SA_THROW_IF(!has_momentum_ && options_.momentum != 0.0,
                        "SGD was constructed with momentum 0, so it has no momentum buffer; "
                        "setting momentum to {} afterwards cannot take effect. Construct the "
                        "optimizer with the momentum you intend to use.",
                        options_.momentum);
            SA_THROW_IF(options_.nesterov && !has_momentum_,
                        "Nesterov momentum needs a momentum buffer, and this SGD was "
                        "constructed without one.");

            const float learning_rate = static_cast<float>(options_.learning_rate);

            std::array<float, SCALAR_COUNT> values{};
            values[NEG_LEARNING_RATE] = -learning_rate;
            values[MOMENTUM] = static_cast<float>(options_.momentum);
            values[WEIGHT_DECAY] = static_cast<float>(options_.weight_decay);
            values[RETAINED] = static_cast<float>(1.0 - options_.dampening);
            values[FIRST_STEP] = step_count() == 0 ? 1.0f : 0.0f;
            values[NESTEROV] = options_.nesterov ? 1.0f : 0.0f;

            scalars_.write(SushiRuntime::span<const float>(values.data(), values.size()));
        }

        std::size_t SGD::record_update()
        {
            return options_.kernel == UpdateKernel::FUSED ? record_fused()
                                                          : record_separate();
        }

        std::size_t SGD::record_fused()
        {
            const float* const s = scalars_.device_pointer();

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                if (!has_momentum_)
                {
                    std::array<SushiBLAS::Tensor*, 2> operands{&slot.parameter, &slot.gradient};

                    SushiBLAS::Dispatch::execute_zip_inplace(engine(), operands,
                        TASK_NAME, Ops::SGD_UPDATE.id,
                        [s](auto& theta, auto& g)
                        {
                            using T = std::decay_t<decltype(theta)>;

                            // The gradient buffer belongs to the compiled step
                            // and is rewritten by the next forward pass, so the
                            // decay term goes into a register instead of it.
                            T staged = g;

                            const T decay = static_cast<T>(s[WEIGHT_DECAY]);
                            if (decay != static_cast<T>(0)) staged = staged + decay * theta;

                            theta = theta + static_cast<T>(s[NEG_LEARNING_RATE]) * staged;
                        });
                    continue;
                }

                std::array<SushiBLAS::Tensor*, 3> operands{
                    &slot.parameter, &slot.gradient, &state.buffer,
                };

                SushiBLAS::Dispatch::execute_zip_inplace(engine(), operands,
                    TASK_NAME, Ops::SGD_UPDATE.id,
                    [s](auto& theta, auto& g, auto& b)
                    {
                        using T = std::decay_t<decltype(theta)>;

                        T staged = g;

                        const T decay = static_cast<T>(s[WEIGHT_DECAY]);
                        if (decay != static_cast<T>(0)) staged = staged + decay * theta;

                        const T momentum = static_cast<T>(s[MOMENTUM]);

                        // b_1 = g_1: no dampening on the first step, matching
                        // PyTorch. Damping it here makes the first update
                        // quietly too small. An assignment rather than a
                        // 0*b + 1*staged blend, so a -0.0 gradient stays -0.0.
                        if (static_cast<T>(s[FIRST_STEP]) != static_cast<T>(0))
                            b = staged;
                        else
                            b = b * momentum + static_cast<T>(s[RETAINED]) * staged;

                        if (static_cast<T>(s[NESTEROV]) != static_cast<T>(0))
                            staged = staged + momentum * b;
                        else
                            staged = b;

                        theta = theta + static_cast<T>(s[NEG_LEARNING_RATE]) * staged;
                    });
            }

            return slots().size();
        }

        std::size_t SGD::record_separate()
        {
            // The reference spelling. Every scalar is read back out of the same
            // device buffer the fused kernel reads, so neither path can drift
            // from the other by picking up a differently rounded constant.
            const float* const s = scalars_.device_pointer();

            const float neg_learning_rate = s[NEG_LEARNING_RATE];
            const float momentum = s[MOMENTUM];
            const float decay = s[WEIGHT_DECAY];
            const float retained = s[RETAINED];
            const bool first = s[FIRST_STEP] != 0.0f;
            const bool nesterov = s[NESTEROV] != 0.0f;

            std::size_t recorded = 0;

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                // The gradient buffer belongs to the compiled step and is
                // rewritten by the next forward pass; staging keeps the decay
                // term out of it.
                engine().blas().copy(slot.gradient, state.staged);
                ++recorded;

                if (decay != 0.0f)
                {
                    engine().blas().axpy(decay, slot.parameter, state.staged);
                    ++recorded;
                }

                if (has_momentum_)
                {
                    if (first)
                    {
                        // b_1 = g_1: no dampening on the first step, matching
                        // PyTorch. Damping it here makes the first update
                        // quietly too small.
                        engine().blas().copy(state.staged, state.buffer);
                        ++recorded;
                    }
                    else
                    {
                        engine().blas().scal(momentum, state.buffer);
                        engine().blas().axpy(retained, state.staged, state.buffer);
                        recorded += 2;
                    }

                    if (nesterov)
                        engine().blas().axpy(momentum, state.buffer, state.staged);
                    else
                        engine().blas().copy(state.buffer, state.staged);
                    ++recorded;
                }

                engine().blas().axpy(neg_learning_rate, state.staged, slot.parameter);
                ++recorded;
            }

            return recorded;
        }

    } // namespace Optim
} // namespace SushiAI

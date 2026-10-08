/**************************************************************************/
/* sgd.cpp                                                                */
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
 * @file sgd.cpp
 * @brief Implements stochastic gradient descent parameter updates with momentum.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/sgd.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /**
             * @enum Scalar
             * @brief Defines device buffer scalar layout for per-step parameters.
             */
            enum Scalar : std::size_t
            {
                NEG_LEARNING_RATE = 0,  /**< -lr, as the separate path's axpy takes it. */
                MOMENTUM,               /**< momentum. */
                WEIGHT_DECAY,           /**< Coupled L2 penalty. */
                RETAINED,               /**< 1 - dampening. */
                FIRST_STEP,             /**< 1 on the first step, 0 afterwards. */
                NESTEROV,               /**< 1 when Nesterov's look-ahead is on. */
                INVERSE_LOSS_SCALE,     /**< 1 / the gradient loss scale; exactly 1 unscaled. */
                SCALAR_COUNT,
            };

            /** @brief The task name; proved below to be the OpID's own string. */
            constexpr const char* K_TASK_NAME = "sushiai.optim.sgd";

            static_assert(Ops::K_SGD_UPDATE.name == std::string_view(K_TASK_NAME),
                          "the fused update's task name must be the string its OpID hashes");
        }

        Result<std::unique_ptr<SGD>> SGD::make(SushiBLAS::Engine& engine,
                                               std::vector<ParameterSlot> parameters,
                                               SGDOptions options)
        {
            SA_TRY(validate_slots(parameters));

            SA_FAIL_IF(options.nesterov && options.momentum <= 0.0,
                       StatusCode::InvalidDescription,
                       "Nesterov momentum needs a positive momentum coefficient; got {}",
                       options.momentum);
            SA_FAIL_IF(options.nesterov && options.dampening != 0.0,
                       StatusCode::InvalidDescription,
                       "Nesterov momentum is undefined with dampening; got {}",
                       options.dampening);

            const bool has_momentum = options.momentum != 0.0;
            const bool separate = options.kernel == UpdateKernel::Separate;

            SA_TRY_ASSIGN(StepScalars scalars, StepScalars::make(engine, SCALAR_COUNT));

            std::vector<State> state;
            state.reserve(parameters.size());
            for (const ParameterSlot& slot : parameters)
            {
                State entry;

                // Allocate staging buffer for separate kernel launches.
                if (separate)
                {
                    SA_TRY_ASSIGN(entry.staged, allocate_like(engine, slot.parameter));
                }
                if (has_momentum)
                {
                    SA_TRY_ASSIGN(entry.buffer, allocate_like(engine, slot.parameter));
                }

                state.push_back(std::move(entry));
            }

            std::unique_ptr<SGD> made(new SGD(engine, std::move(parameters), options,
                                              std::move(state), std::move(scalars),
                                              has_momentum));
            return std::move(made);
        }

        SGD::SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                 SGDOptions options, std::vector<State> state, StepScalars scalars,
                 bool has_momentum) noexcept
            : Optimizer(engine, std::move(parameters)), options_(options),
              state_(std::move(state)), scalars_(std::move(scalars)),
              has_momentum_(has_momentum)
        {
        }

        void SGD::write_step_scalars()
        {
            // Validate momentum buffer configuration is structural at initialization.
            SA_FATAL_IF(!has_momentum_ && options_.momentum != 0.0,
                        "SGD was constructed with momentum 0, so it has no momentum buffer; "
                        "setting momentum to {} afterwards cannot take effect. Construct the "
                        "optimizer with the momentum you intend to use.",
                        options_.momentum);
            SA_FATAL_IF(options_.nesterov && !has_momentum_,
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

            // Write inverse loss scale factor for exact gradient descaling.
            values[INVERSE_LOSS_SCALE] = inverse_loss_scale();

            scalars_.write(SushiRuntime::span<const float>(values.data(), values.size()));
        }

        std::size_t SGD::record_update(SushiBLAS::TaskRecorder& recorder)
        {
            return options_.kernel == UpdateKernel::Fused ? record_fused(recorder)
                                                          : record_separate(recorder);
        }

        std::size_t SGD::record_fused(SushiBLAS::TaskRecorder& recorder)
        {
            const float* const s = scalars_.device_pointer();

            // Retrieve gradient gate pointer for branching on overflowed steps.
            const std::int64_t* const gate = gradient_gate();

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                if (!has_momentum_)
                {
                    std::array<SushiBLAS::Tensor*, 2> operands{&slot.parameter, &slot.gradient};

                    SushiBLAS::Dispatch::execute_zip_inplace(recorder, operands,
                        K_TASK_NAME, Ops::K_SGD_UPDATE.id,
                        [s, gate, i](auto& theta, auto& g)
                        {
                            using T = std::decay_t<decltype(theta)>;

                            // Skip parameter update if gradient gate signaled overflow.
                            if (gate != nullptr && gate[i] != 0) return;

                            // Compute unscaled gradient into register to preserve gradient buffer.
                            T staged = g * static_cast<T>(s[INVERSE_LOSS_SCALE]);

                            const T decay = static_cast<T>(s[WEIGHT_DECAY]);
                            if (decay != static_cast<T>(0)) staged = staged + decay * theta;

                            theta = theta + static_cast<T>(s[NEG_LEARNING_RATE]) * staged;
                        });
                    continue;
                }

                std::array<SushiBLAS::Tensor*, 3> operands{
                    &slot.parameter, &slot.gradient, &state.buffer,
                };

                SushiBLAS::Dispatch::execute_zip_inplace(recorder, operands,
                    K_TASK_NAME, Ops::K_SGD_UPDATE.id,
                    [s, gate, i](auto& theta, auto& g, auto& b)
                    {
                        using T = std::decay_t<decltype(theta)>;

                        if (gate != nullptr && gate[i] != 0) return;

                        T staged = g * static_cast<T>(s[INVERSE_LOSS_SCALE]);

                        const T decay = static_cast<T>(s[WEIGHT_DECAY]);
                        if (decay != static_cast<T>(0)) staged = staged + decay * theta;

                        const T momentum = static_cast<T>(s[MOMENTUM]);

                        // Assign staged gradient directly on first step to omit dampening.
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

        std::size_t SGD::record_separate(SushiBLAS::TaskRecorder& recorder)
        {
            // Read scalar parameters from common device buffer for separate kernel launches.
            const float* const s = scalars_.device_pointer();

            SushiBLAS::BLASOps blas(recorder);

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

                // Copy gradient into staging buffer to avoid modifying forward graph memory.
                blas.copy(slot.gradient, state.staged);
                ++recorded;

                if (decay != 0.0f)
                {
                    blas.axpy(decay, slot.parameter, state.staged);
                    ++recorded;
                }

                if (has_momentum_)
                {
                    if (first)
                    {
                        // Initialize momentum buffer directly on first step to omit dampening.
                        blas.copy(state.staged, state.buffer);
                        ++recorded;
                    }
                    else
                    {
                        blas.scal(momentum, state.buffer);
                        blas.axpy(retained, state.staged, state.buffer);
                        recorded += 2;
                    }

                    if (nesterov)
                        blas.axpy(momentum, state.buffer, state.staged);
                    else
                        blas.copy(state.buffer, state.staged);
                    ++recorded;
                }

                blas.axpy(neg_learning_rate, state.staged, slot.parameter);
                ++recorded;
            }

            return recorded;
        }

    } // namespace Optim
} // namespace SushiAI

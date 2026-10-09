/**************************************************************************/
/* adamw.cpp                                                              */
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
 * @file adamw.cpp
 * @brief Implements AdamW parameter updates for fused and unfused kernels.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/adamw.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief 1 - lr * weight_decay. */
            constexpr std::size_t K_DECAY_FACTOR = 0;

            /** @brief beta1. */
            constexpr std::size_t K_BETA1 = 1;

            /** @brief 1 - beta1. */
            constexpr std::size_t K_ONE_MINUS_BETA1 = 2;

            /** @brief beta2. */
            constexpr std::size_t K_BETA2 = 3;

            /** @brief 1 - beta2. */
            constexpr std::size_t K_ONE_MINUS_BETA2 = 4;

            /** @brief 1 / (1 - beta2^t). */
            constexpr std::size_t K_INVERSE_BC2 = 5;

            /** @brief eps, added after the correction. */
            constexpr std::size_t K_EPSILON = 6;

            /** @brief -lr / (1 - beta1^t). */
            constexpr std::size_t K_STEP_SIZE = 7;

            /** @brief 1 / the gradient loss scale; exactly 1 when unscaled. */
            constexpr std::size_t K_INVERSE_LOSS_SCALE = 8;

            /** @brief Holds the number of per-step scalars in the device buffer. */
            constexpr std::size_t K_SCALAR_COUNT = 9;

            /**
             * @brief The task's human-readable name, in the form the seam wants.
             *
             * @c execute_zip_inplace takes a @c const @c char*, and the profiler
             * shows this string next to the row @c Ops::K_ADAMW_UPDATE's hash
             * selects. The assertion below makes it impossible for the two to
             * name different things.
             */
            constexpr const char* K_TASK_NAME = "sushiai.optim.adamw";

            static_assert(Ops::K_ADAMW_UPDATE.name == std::string_view(K_TASK_NAME),
                          "the fused update's task name must be the string its OpID hashes");
        }

        Result<std::unique_ptr<AdamW>> AdamW::make(SushiBLAS::Engine& engine,
                                                   std::vector<ParameterSlot> parameters,
                                                   AdamWOptions options)
        {
            SA_TRY(validate_slots(parameters));

            SA_FAIL_IF(options.beta1 < 0.0 || options.beta1 >= 1.0,
                       StatusCode::InvalidDescription,
                       "AdamW beta1 must lie in [0, 1); got {}", options.beta1);
            SA_FAIL_IF(options.beta2 < 0.0 || options.beta2 >= 1.0,
                       StatusCode::InvalidDescription,
                       "AdamW beta2 must lie in [0, 1); got {}", options.beta2);
            SA_FAIL_IF(options.eps <= 0.0, StatusCode::InvalidDescription,
                       "AdamW eps must be positive; got {}", options.eps);

            SA_TRY_ASSIGN(StepScalars scalars, StepScalars::make(engine, K_SCALAR_COUNT));

            const bool separate = options.kernel == UpdateKernel::Separate;

            std::vector<State> state;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                state.reserve(parameters.size());
                for (const ParameterSlot& slot : parameters)
                {
                    State entry;
                    SA_TRY_ASSIGN(entry.first, allocate_like(engine, slot.parameter));
                    SA_TRY_ASSIGN(entry.second, allocate_like(engine, slot.parameter));

                    // Scratch buffers are allocated only when separate kernels execute.
                    if (separate)
                    {
                        SA_TRY_ASSIGN(entry.denom, allocate_like(engine, slot.parameter));
                        SA_TRY_ASSIGN(entry.update, allocate_like(engine, slot.parameter));
                        SA_TRY_ASSIGN(entry.ones, allocate_like(engine, slot.parameter));
                    }

                    state.push_back(std::move(entry));
                }
                return Status();
            }));

            // Invariant: Moments and constant buffers are initialized eagerly before first step.
            SA_TRY(Detail::record_and_run(engine, [&]()
            {
                SushiBLAS::RandomOps random(engine);
                for (State& entry : state)
                {
                    random.constant(entry.first, 0.0);
                    random.constant(entry.second, 0.0);
                    if (separate)
                    {
                        random.constant(entry.ones, 1.0);
                    }
                }
                return Status();
            }));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<AdamW>(new AdamW(engine, std::move(parameters), options,
                                                        std::move(state), std::move(scalars)));
            });
        }

        AdamW::AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                     AdamWOptions options, std::vector<State> state, StepScalars scalars) noexcept
            : Optimizer(engine, std::move(parameters)), options_(options),
              state_(std::move(state)), scalars_(std::move(scalars))
        {
        }

        void AdamW::write_step_scalars()
        {
            const double t = static_cast<double>(step_count() + 1);
            const double bias_correction1 = 1.0 - std::pow(options_.beta1, t);
            const double bias_correction2 = 1.0 - std::pow(options_.beta2, t);

            // Invariant: Floats match separate path precision for bit-identical comparison.
            std::array<float, K_SCALAR_COUNT> values{};
            values[K_DECAY_FACTOR] =
                static_cast<float>(1.0 - options_.learning_rate * options_.weight_decay);
            values[K_BETA1] = static_cast<float>(options_.beta1);
            values[K_ONE_MINUS_BETA1] = static_cast<float>(1.0 - options_.beta1);
            values[K_BETA2] = static_cast<float>(options_.beta2);
            values[K_ONE_MINUS_BETA2] = static_cast<float>(1.0 - options_.beta2);
            values[K_INVERSE_BC2] = static_cast<float>(1.0 / bias_correction2);
            values[K_EPSILON] = static_cast<float>(options_.eps);
            values[K_STEP_SIZE] =
                static_cast<float>(-options_.learning_rate / bias_correction1);

            // Invariant: Inverse loss scale is an exact power of two or unit identity.
            values[K_INVERSE_LOSS_SCALE] = inverse_loss_scale();

            scalars_.write(SushiRuntime::span<const float>(values.data(), values.size()));
        }

        const SushiBLAS::Tensor& AdamW::first_moment(std::size_t index) const
        {
            SA_FATAL_IF(index >= state_.size(),
                        "This AdamW updates {} parameters; there is no moment {}",
                        state_.size(), index);
            return state_[index].first;
        }

        const SushiBLAS::Tensor& AdamW::second_moment(std::size_t index) const
        {
            SA_FATAL_IF(index >= state_.size(),
                        "This AdamW updates {} parameters; there is no moment {}",
                        state_.size(), index);
            return state_[index].second;
        }

        std::size_t AdamW::record_update(SushiBLAS::TaskRecorder& recorder)
        {
            return options_.kernel == UpdateKernel::Fused ? record_fused(recorder)
                                                          : record_separate(recorder);
        }

        std::size_t AdamW::record_fused(SushiBLAS::TaskRecorder& recorder)
        {
            // Device pointer captures step scalars by reference across replays.
            const float* const s = scalars_.device_pointer();

            // Gate pointer evaluates gradient overflow flags uniformly for step skip.
            const std::int64_t* const gate = gradient_gate();

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                std::array<SushiBLAS::Tensor*, 4> operands{
                    &slot.parameter, &slot.gradient, &state.first, &state.second,
                };

                SushiBLAS::Dispatch::execute_zip_inplace(recorder, operands,
                    K_TASK_NAME, Ops::K_ADAMW_UPDATE.id,
                    [s, gate, i](auto& theta, auto& g_scaled, auto& m, auto& v)
                    {
                        using T = std::decay_t<decltype(theta)>;

                        // Invariant: Early exit preserves theta, m, and v exactly when gradients overflow.
                        if (gate != nullptr && gate[i] != 0) return;

                        // Invariant: Unscaling occurs prior to moment and decay updates.
                        const T g = g_scaled * static_cast<T>(s[K_INVERSE_LOSS_SCALE]);

                        // Invariant: Decoupled weight decay applies to previous theta value.
                        const T decay = static_cast<T>(s[K_DECAY_FACTOR]);
                        if (decay != static_cast<T>(1)) theta = theta * decay;

                        // 2. m <- beta1 * m + (1 - beta1) * g
                        m = m * static_cast<T>(s[K_BETA1])
                          + static_cast<T>(s[K_ONE_MINUS_BETA1]) * g;

                        // 3. v <- beta2 * v + (1 - beta2) * g^2
                        v = v * static_cast<T>(s[K_BETA2])
                          + static_cast<T>(s[K_ONE_MINUS_BETA2]) * (g * g);

                        // Invariant: Bias correction scales v before square root and epsilon addition.
                        T denom = v * static_cast<T>(s[K_INVERSE_BC2]);
                        denom = SushiBLAS::Dispatch::safe_sqrt(denom);
                        denom = denom + static_cast<T>(s[K_EPSILON]);

                        // 5. theta <- theta - (lr / bc1) * m / denom
                        theta = theta + static_cast<T>(s[K_STEP_SIZE]) * (m / denom);
                    });
            }

            return slots().size();
        }

        std::size_t AdamW::record_separate(SushiBLAS::TaskRecorder& recorder)
        {
            SushiBLAS::BLASOps blas(recorder);
            SushiBLAS::ElementwiseOps elementwise(recorder);

            // Invariant: Separate reference path reads identical device scalar constants.
            const float* const s = scalars_.device_pointer();

            const float decay_factor = s[K_DECAY_FACTOR];
            const float beta1 = s[K_BETA1];
            const float one_minus_beta1 = s[K_ONE_MINUS_BETA1];
            const float beta2 = s[K_BETA2];
            const float one_minus_beta2 = s[K_ONE_MINUS_BETA2];
            const float inverse_correction2 = s[K_INVERSE_BC2];
            const float eps = s[K_EPSILON];
            const float step_size = s[K_STEP_SIZE];

            std::size_t recorded = 0;

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                // 1. Decoupled decay, on theta_{t-1}, before anything else.
                if (decay_factor != 1.0f)
                {
                    blas.scal(decay_factor, slot.parameter);
                    ++recorded;
                }

                // 2. m <- beta1 * m + (1 - beta1) * g
                blas.scal(beta1, state.first);
                blas.axpy(one_minus_beta1, slot.gradient, state.first);

                // 3. v <- beta2 * v + (1 - beta2) * g^2, borrowing denom for g^2.
                elementwise.mul(slot.gradient, slot.gradient, state.denom);
                blas.scal(beta2, state.second);
                blas.axpy(one_minus_beta2, state.denom, state.second);

                // Invariant: Bias correction scales v before square root and epsilon addition.
                blas.copy(state.second, state.denom);
                blas.scal(inverse_correction2, state.denom);
                elementwise.sqrt(state.denom);
                blas.axpy(eps, state.ones, state.denom);

                // 5. theta <- theta - (lr / bc1) * m / denom
                elementwise.div(state.first, state.denom, state.update);
                blas.axpy(step_size, state.update, slot.parameter);

                recorded += 11;
            }

            return recorded;
        }

    } // namespace Optim
} // namespace SushiAI

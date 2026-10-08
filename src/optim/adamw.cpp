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
#include <string_view>
#include <type_traits>
#include <utility>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/optim/adamw.hpp>

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
             * An enum rather than bare indices because the writer and the
             * kernel are two hundred lines apart and a transposed pair of
             * literals would be a plausible, converging, wrong optimizer.
             */
            enum Scalar : std::size_t
            {
                DECAY_FACTOR = 0,   /**< 1 - lr * weight_decay. */
                BETA1,              /**< beta1. */
                ONE_MINUS_BETA1,    /**< 1 - beta1. */
                BETA2,              /**< beta2. */
                ONE_MINUS_BETA2,    /**< 1 - beta2. */
                INVERSE_BC2,        /**< 1 / (1 - beta2^t). */
                EPSILON,            /**< eps, added after the correction. */
                STEP_SIZE,          /**< -lr / (1 - beta1^t). */
                INVERSE_LOSS_SCALE, /**< 1 / the gradient loss scale; exactly 1 when unscaled. */
                SCALAR_COUNT,
            };

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

        AdamW::AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                     AdamWOptions options)
            : Optimizer(engine, std::move(parameters)), options_(options),
              scalars_(engine, SCALAR_COUNT)
        {
            SA_THROW_IF(options_.beta1 < 0.0 || options_.beta1 >= 1.0,
                        "AdamW beta1 must lie in [0, 1); got {}", options_.beta1);
            SA_THROW_IF(options_.beta2 < 0.0 || options_.beta2 >= 1.0,
                        "AdamW beta2 must lie in [0, 1); got {}", options_.beta2);
            SA_THROW_IF(options_.eps <= 0.0,
                        "AdamW eps must be positive; got {}", options_.eps);

            const bool separate = options_.kernel == UpdateKernel::Separate;

            state_.reserve(slots().size());
            for (const ParameterSlot& slot : slots())
            {
                State state;
                state.first = allocate_state(slot.parameter);
                state.second = allocate_state(slot.parameter);

                engine.random().constant(state.first, 0.0);
                engine.random().constant(state.second, 0.0);

                // Scratch buffers are allocated only when separate kernels execute.
                if (separate)
                {
                    state.denom = allocate_state(slot.parameter);
                    state.update = allocate_state(slot.parameter);
                    state.ones = allocate_state(slot.parameter);
                    engine.random().constant(state.ones, 1.0);
                }

                state_.push_back(std::move(state));
            }

            // Invariant: Moments and constant buffers are initialized eagerly before first step.
            flush();
        }

        void AdamW::write_step_scalars()
        {
            const double t = static_cast<double>(step_count() + 1);
            const double bias_correction1 = 1.0 - std::pow(options_.beta1, t);
            const double bias_correction2 = 1.0 - std::pow(options_.beta2, t);

            // Invariant: Floats match separate path precision for bit-identical comparison.
            std::array<float, SCALAR_COUNT> values{};
            values[DECAY_FACTOR] =
                static_cast<float>(1.0 - options_.learning_rate * options_.weight_decay);
            values[BETA1] = static_cast<float>(options_.beta1);
            values[ONE_MINUS_BETA1] = static_cast<float>(1.0 - options_.beta1);
            values[BETA2] = static_cast<float>(options_.beta2);
            values[ONE_MINUS_BETA2] = static_cast<float>(1.0 - options_.beta2);
            values[INVERSE_BC2] = static_cast<float>(1.0 / bias_correction2);
            values[EPSILON] = static_cast<float>(options_.eps);
            values[STEP_SIZE] =
                static_cast<float>(-options_.learning_rate / bias_correction1);

            // Invariant: Inverse loss scale is an exact power of two or unit identity.
            values[INVERSE_LOSS_SCALE] = inverse_loss_scale();

            scalars_.write(SushiRuntime::span<const float>(values.data(), values.size()));
        }

        const SushiBLAS::Tensor& AdamW::first_moment(std::size_t index) const
        {
            SA_THROW_IF(index >= state_.size(),
                        "This AdamW updates {} parameters; there is no moment {}",
                        state_.size(), index);
            return state_[index].first;
        }

        const SushiBLAS::Tensor& AdamW::second_moment(std::size_t index) const
        {
            SA_THROW_IF(index >= state_.size(),
                        "This AdamW updates {} parameters; there is no moment {}",
                        state_.size(), index);
            return state_[index].second;
        }

        std::size_t AdamW::record_update()
        {
            return options_.kernel == UpdateKernel::Fused ? record_fused()
                                                          : record_separate();
        }

        std::size_t AdamW::record_fused()
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

                SushiBLAS::Dispatch::execute_zip_inplace(engine(), operands,
                    K_TASK_NAME, Ops::K_ADAMW_UPDATE.id,
                    [s, gate, i](auto& theta, auto& g_scaled, auto& m, auto& v)
                    {
                        using T = std::decay_t<decltype(theta)>;

                        // Invariant: Early exit preserves theta, m, and v exactly when gradients overflow.
                        if (gate != nullptr && gate[i] != 0) return;

                        // Invariant: Unscaling occurs prior to moment and decay updates.
                        const T g = g_scaled * static_cast<T>(s[INVERSE_LOSS_SCALE]);

                        // Invariant: Decoupled weight decay applies to previous theta value.
                        const T decay = static_cast<T>(s[DECAY_FACTOR]);
                        if (decay != static_cast<T>(1)) theta = theta * decay;

                        // 2. m <- beta1 * m + (1 - beta1) * g
                        m = m * static_cast<T>(s[BETA1])
                          + static_cast<T>(s[ONE_MINUS_BETA1]) * g;

                        // 3. v <- beta2 * v + (1 - beta2) * g^2
                        v = v * static_cast<T>(s[BETA2])
                          + static_cast<T>(s[ONE_MINUS_BETA2]) * (g * g);

                        // Invariant: Bias correction scales v before square root and epsilon addition.
                        T denom = v * static_cast<T>(s[INVERSE_BC2]);
                        denom = SushiBLAS::Dispatch::safe_sqrt(denom);
                        denom = denom + static_cast<T>(s[EPSILON]);

                        // 5. theta <- theta - (lr / bc1) * m / denom
                        theta = theta + static_cast<T>(s[STEP_SIZE]) * (m / denom);
                    });
            }

            return slots().size();
        }

        std::size_t AdamW::record_separate()
        {
            // Invariant: Separate reference path reads identical device scalar constants.
            const float* const s = scalars_.device_pointer();

            const float decay_factor = s[DECAY_FACTOR];
            const float beta1 = s[BETA1];
            const float one_minus_beta1 = s[ONE_MINUS_BETA1];
            const float beta2 = s[BETA2];
            const float one_minus_beta2 = s[ONE_MINUS_BETA2];
            const float inverse_correction2 = s[INVERSE_BC2];
            const float eps = s[EPSILON];
            const float step_size = s[STEP_SIZE];

            std::size_t recorded = 0;

            for (std::size_t i = 0; i < slots().size(); ++i)
            {
                ParameterSlot& slot = slots()[i];
                State& state = state_[i];

                // 1. Decoupled decay, on theta_{t-1}, before anything else.
                if (decay_factor != 1.0f)
                {
                    engine().blas().scal(decay_factor, slot.parameter);
                    ++recorded;
                }

                // 2. m <- beta1 * m + (1 - beta1) * g
                engine().blas().scal(beta1, state.first);
                engine().blas().axpy(one_minus_beta1, slot.gradient, state.first);

                // 3. v <- beta2 * v + (1 - beta2) * g^2, borrowing denom for g^2.
                engine().elementwise().mul(slot.gradient, slot.gradient, state.denom);
                engine().blas().scal(beta2, state.second);
                engine().blas().axpy(one_minus_beta2, state.denom, state.second);

                // Invariant: Bias correction scales v before square root and epsilon addition.
                engine().blas().copy(state.second, state.denom);
                engine().blas().scal(inverse_correction2, state.denom);
                engine().elementwise().sqrt(state.denom);
                engine().blas().axpy(eps, state.ones, state.denom);

                // 5. theta <- theta - (lr / bc1) * m / denom
                engine().elementwise().div(state.first, state.denom, state.update);
                engine().blas().axpy(step_size, state.update, slot.parameter);

                recorded += 11;
            }

            return recorded;
        }

    } // namespace Optim
} // namespace SushiAI

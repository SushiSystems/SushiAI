/**************************************************************************/
/* optimizer.hpp                                                          */
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
 * @file optimizer.hpp
 * @brief The parameter-update seam.
 *
 * @section why_not_in_graph Why the update is not part of the compiled step
 *
 * Everything else about a step is lowered into the IR, compiled once and
 * replayed. The update is not, and the reason is written into ir.hpp's
 * @c Attributes: a per-step scalar baked into a node becomes a compile-time
 * constant, and changing it forces a recompile - which is exactly what the
 * compile-once guarantee exists to prevent. AdamW's bias corrections are
 * per-step scalars by definition (1 - beta^t changes every step), and
 * SushiBLAS takes its alphas by value, so there is no way to hand a compiled
 * @c axpy a number that changes between replays.
 *
 * So the optimizers record their own short op sequence per step and flush it.
 * The cost is one extra graph analysis per step over a handful of nodes per
 * parameter, which is small next to the forward and backward passes and is
 * measurable rather than hidden. The fix, when SushiRuntime's late-binding
 * block reaches the BLAS layer, is to make the scalars live bindings and fold
 * the update into the same compiled plan.
 *
 * @section zero_grad There is no zero_grad()
 *
 * Gradients are written, not accumulated into: every gradient buffer in the
 * step is produced by an operation with beta = 0, and the tiered accumulation
 * in backward.cpp overwrites the first contributor before adding the rest. A
 * @c zero_grad() would be a no-op, and a no-op named after a real PyTorch
 * obligation is worse than its absence - it invites the reader to believe
 * something is being cleared.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @struct ParameterSlot
         * @brief One parameter and its gradient, as the optimizer sees them.
         *
         * Tensors by value: a @c SushiBLAS::Tensor is a handle that shares its
         * Storage, so a copy costs a reference-count bump and removes any
         * dependence on where the pool keeps its map entries.
         */
        struct ParameterSlot
        {
            /** @brief The parameter's dotted name, e.g. "0.weight". */
            std::string name;

            /** @brief The parameter itself; updated in place. */
            SushiBLAS::Tensor parameter;

            /** @brief Its gradient, rewritten by every step. */
            SushiBLAS::Tensor gradient;
        };

        /**
         * @class Optimizer
         * @brief What every update rule has in common.
         *
         * Owns the parameter list and the step counter; subclasses own the
         * state buffers and the arithmetic. Non-copyable, because the state
         * buffers a subclass allocates are tied to these exact parameters.
         */
        class Optimizer
        {
            public:
                /**
                 * @brief Binds an optimizer to a fixed set of parameters.
                 * @param engine The engine the update is recorded on; must outlive this object.
                 * @param parameters The parameters to update, with their gradients.
                 * @throws Error If a slot's parameter and gradient disagree in
                 *         shape or dtype, or the element type is not a real float.
                 */
                Optimizer(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters);

                virtual ~Optimizer() = default;

                Optimizer(const Optimizer&) = delete;
                Optimizer& operator=(const Optimizer&) = delete;

                /**
                 * @brief Applies one update to every parameter.
                 *
                 * Records the whole update, executes it, and waits, so on
                 * return the parameters are readable from the host.
                 */
                virtual void step() = 0;

                /** @brief How many parameters this optimizer updates. */
                std::size_t parameter_count() const noexcept { return parameters_.size(); }

                /** @brief How many updates have been applied; 0 before the first step(). */
                std::uint64_t step_count() const noexcept { return step_; }

                /** @brief The parameters, for inspection. */
                const std::vector<ParameterSlot>& parameters() const noexcept { return parameters_; }

            protected:
                /** @brief Runs the recorded update and waits for the device. */
                void flush();

                /** @brief Allocates a state buffer shaped like @p like, with its own Storage. */
                SushiBLAS::Tensor allocate_state(const SushiBLAS::Tensor& like) const;

                /** @brief The engine the update is recorded on. */
                SushiBLAS::Engine& engine() const noexcept { return *engine_; }

                /** @brief The parameters being updated. */
                std::vector<ParameterSlot>& slots() noexcept { return parameters_; }

                /** @brief Bumps the step counter; call once per step(). */
                void advance() noexcept { ++step_; }

            private:
                SushiBLAS::Engine* engine_;
                std::vector<ParameterSlot> parameters_;
                std::uint64_t step_ = 0;
        };

    } // namespace Optim
} // namespace SushiAI

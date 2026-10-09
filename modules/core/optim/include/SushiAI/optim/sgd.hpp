/**************************************************************************/
/* sgd.hpp                                                                */
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
 * @file sgd.hpp
 * @brief Declares the stochastic gradient descent optimizer with momentum.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/core/result.hpp>
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
            UpdateKernel kernel = UpdateKernel::Fused;
        };

        /**
         * @class SGD
         * @brief The update rule above, applied to a fixed parameter set.
         */
        class SGD final : public Optimizer
        {
            public:
                /**
                 * @brief Validates the slots and options and allocates the state.
                 * @return The optimizer, or the first failure of validation or allocation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<SGD>> make(
                    SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                    SGDOptions options = {});

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::Fused;
                }

                /** @brief The hyperparameters in force. */
                const SGDOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                SGDOptions& options() noexcept { return options_; }

            protected:
                /**
                 * @brief Records the update for every parameter onto @p recorder.
                 * @param recorder The recorder the update tasks are added to.
                 * @return One task per parameter when fused, four or five each when not.
                 */
                std::size_t record_update(SushiBLAS::TaskRecorder& recorder) override;

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 * @pre options() asks for momentum or nesterov only if make() allocated the
                 *      momentum buffer; otherwise the process stops.
                 */
                void write_step_scalars() override;

            private:
                struct State
                {
                    /** @brief The momentum buffer b; absent when momentum is off. */
                    SushiBLAS::Tensor buffer;

                    /** @brief SEPARATE only: g_t, kept off the graph's own gradient buffer. */
                    SushiBLAS::Tensor staged;
                };

                /**
                 * @brief Takes ownership of state that make() has already allocated.
                 * @pre validate_slots(parameters) has returned success.
                 */
                SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                    SGDOptions options, std::vector<State> state, StepScalars scalars,
                    bool has_momentum) noexcept;

                /** @brief Records one fused kernel per parameter; returns the task count. */
                std::size_t record_fused(SushiBLAS::TaskRecorder& recorder);

                /** @brief Records the separate ops of every parameter; returns the task count. */
                std::size_t record_separate(SushiBLAS::TaskRecorder& recorder);

                SGDOptions options_;
                std::vector<State> state_;
                StepScalars scalars_;

                /** @brief Whether a momentum buffer was allocated; fixed at construction. */
                bool has_momentum_ = false;
        };

    } // namespace Optim
} // namespace SushiAI

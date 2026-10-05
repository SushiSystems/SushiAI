/****************************************************************/
/* sgd.hpp                                                      */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file sgd.hpp
 * @brief Declares the stochastic gradient descent optimizer with momentum.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiBLAS/engine.hpp>
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
            UpdateKernel kernel = UpdateKernel::FUSED;
        };

        /**
         * @class SGD
         * @brief The update rule above, applied to a fixed parameter set.
         */
        class SGD final : public Optimizer
        {
            public:
                /**
                 * @brief Binds SGD to parameters and allocates momentum state buffers.
                 * @param engine Engine used to record update kernels.
                 * @param parameters Parameter and gradient slots to update.
                 * @param options Hyperparameters configuring learning rate and momentum.
                 * @throws Error If nesterov is set without momentum, or with dampening.
                 */
                SGD(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
                    SGDOptions options = {});

                /** @brief True when the update is fused, and therefore replayable. */
                bool is_replay_invariant() const noexcept override
                {
                    return options_.kernel == UpdateKernel::FUSED;
                }

                /** @brief The hyperparameters in force. */
                const SGDOptions& options() const noexcept { return options_; }

                /** @brief Mutable access, for a learning-rate schedule. */
                SGDOptions& options() noexcept { return options_; }

            protected:
                /**
                 * @brief Records the update for every parameter into the engine's builder.
                 * @return One task per parameter when fused, four or five each when not.
                 */
                std::size_t record_update() override;

                /**
                 * @brief Recomputes this step's scalars into the device buffer.
                 * @throws Error If momentum is enabled without allocated buffer.
                 */
                void write_step_scalars() override;

            private:
                struct State
                {
                    SushiBLAS::Tensor buffer;   /**< The momentum buffer b; absent when momentum is off. */
                    SushiBLAS::Tensor staged;   /**< SEPARATE only: g_t, kept off the graph's own gradient buffer. */
                };

                std::size_t record_fused();
                std::size_t record_separate();

                SGDOptions options_;
                std::vector<State> state_;
                StepScalars scalars_;

                /** @brief Whether a momentum buffer was allocated; fixed at construction. */
                bool has_momentum_ = false;
        };

    } // namespace Optim
} // namespace SushiAI

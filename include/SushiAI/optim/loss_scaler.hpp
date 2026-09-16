/**************************************************************************/
/* loss_scaler.hpp                                                        */
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

/**
 * @file loss_scaler.hpp
 * @brief Declares dynamic gradient loss scaling utilities for mixed precision.
 * @author Mustafa Garip
 */

#pragma once

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
         * @struct LossScaleOptions
         * @brief Configuration constants governing dynamic loss scaling.
         */
        struct LossScaleOptions
        {
            /**
             * @brief The scale the first step runs at; 2^16, as PyTorch starts.
             *
             * High on purpose. Starting low and growing wastes the early steps
             * at a scale that underflows, and the cost of starting too high is
             * bounded and self-correcting: a handful of skipped steps while the
             * backoff finds the ceiling.
             */
            double initial = 65536.0;

            /** @brief What the scale is multiplied by after a clean run; must be a power of two >= 1. */
            double growth = 2.0;

            /** @brief What it is multiplied by on overflow; must be a power of two in (0, 1]. */
            double backoff = 0.5;

            /**
             * @brief How many consecutive clean steps buy one growth.
             *
             * PyTorch's default is 2000. It is large because growing costs a
             * skipped step when it overshoots and gains only headroom that was
             * not being used, so the trade is asymmetric in the same direction
             * the backoff is.
             */
            std::uint64_t growth_interval = 2000;

            /** @brief The floor; the scale never backs off below this. */
            double minimum = 1.0;

            /** @brief The ceiling; the scale never grows above this. */
            double maximum = 1073741824.0; // 2^30
        };

        /**
         * @class LossScaleState
         * @brief Host-side state machine tracking dynamic loss scale adjustments.
         */
        class LossScaleState
        {
            public:
                /**
                 * @brief Builds a state machine over @p options.
                 * @param options The four constants; all must be powers of two.
                 * @throws Error If a constant is not a power of two, if the
                 *         bounds do not bracket the initial scale, if growth is
                 *         below one or backoff is not in (0, 1].
                 */
                explicit LossScaleState(LossScaleOptions options = {});

                /**
                 * @brief Folds one step's verdict into the state.
                 *
                 * @param overflow Whether any gradient came back non-finite.
                 */
                void observe(bool overflow) noexcept;

                /** @brief The scale the next step will run at. */
                double scale() const noexcept { return scale_; }

                /**
                 * @brief 1 / @ref scale, exactly.
                 *
                 * Exact because the scale is a power of two, which is the
                 * property the whole design is arranged around: the unscale is
                 * a change of exponent and nothing else.
                 */
                double inverse_scale() const noexcept { return 1.0 / scale_; }

                /** @brief Whether the step just observed was thrown away. */
                bool last_step_skipped() const noexcept { return skipped_; }

                /** @brief How many clean steps have run since the last overflow or growth. */
                std::uint64_t consecutive_good_steps() const noexcept { return good_; }

                /** @brief How many steps have been thrown away in total. */
                std::uint64_t skipped_steps() const noexcept { return skips_; }

                /** @brief How many times the scale has halved. */
                std::uint64_t backoffs() const noexcept { return backoffs_; }

                /** @brief How many times the scale has doubled. */
                std::uint64_t growths() const noexcept { return growths_; }

                /** @brief The options in force. */
                const LossScaleOptions& options() const noexcept { return options_; }

            private:
                LossScaleOptions options_;
                double scale_ = 0.0;
                std::uint64_t good_ = 0;
                std::uint64_t skips_ = 0;
                std::uint64_t backoffs_ = 0;
                std::uint64_t growths_ = 0;
                bool skipped_ = false;
        };

        /**
         * @class LossScaler
         * @brief The state machine, its device-resident scale, and the screen.
         *
         * Non-copyable: kernels capture the addresses of both its buffers, so
         * the object must be the only thing that owns them.
         */
        class LossScaler
        {
            public:
                /**
                 * @brief Constructs a loss scaler bound to device factor and gradient counts.
                 * @param engine Engine executing screen kernels and holding count buffer.
                 * @param factor Single-element float tensor scaled for backward seeding.
                 * @param parameter_count Number of parameter gradients screened.
                 * @param options Configuration options for scale adjustments.
                 */
                LossScaler(SushiBLAS::Engine& engine,
                           SushiBLAS::Tensor factor,
                           std::size_t parameter_count,
                           LossScaleOptions options = {});

                LossScaler(const LossScaler&) = delete;
                LossScaler& operator=(const LossScaler&) = delete;

                /**
                 * @brief Records non-finite gradient count kernels for each parameter.
                 * @param gradients Pointers to gradient tensors in slot order.
                 * @return Number of device tasks recorded into the execution plan.
                 * @throws Error If gradient count does not match parameter count.
                 */
                std::size_t record_screens(const std::vector<SushiBLAS::Tensor*>& gradients);

                /**
                 * @brief Publishes current step scale into device memory before replay.
                 */
                void before_replay();

                /**
                 * @brief Reads the screens back and advances the state machine.
                 *
                 * One host read of @c parameter_count 8-byte counts, at the step
                 * boundary, after the device has been waited on - so it adds no
                 * synchronisation the loop was not already paying.
                 */
                void after_replay();

                /**
                 * @brief Returns pointer to device non-finite count buffer for gating.
                 * @return Pointer to base element of count buffer.
                 */
                const std::int64_t* gate() const noexcept { return counts_; }

                /** @brief How many gradients this scaler screens. */
                std::size_t parameter_count() const noexcept { return count_; }

                /** @brief The scale the next step will run at. */
                double scale() const noexcept { return state_.scale(); }

                /** @brief 1 / @ref scale, exactly; what the update multiplies the gradient by. */
                double inverse_scale() const noexcept { return state_.inverse_scale(); }

                /** @brief Whether the step just completed was thrown away. */
                bool last_step_skipped() const noexcept { return state_.last_step_skipped(); }

                /** @brief How many steps have been thrown away in total. */
                std::uint64_t skipped_steps() const noexcept { return state_.skipped_steps(); }

                /** @brief How many times the scale has halved. */
                std::uint64_t backoffs() const noexcept { return state_.backoffs(); }

                /** @brief How many times the scale has doubled. */
                std::uint64_t growths() const noexcept { return state_.growths(); }

                /** @brief The state machine, for a test or a report that wants the detail. */
                const LossScaleState& state() const noexcept { return state_; }

                /** @brief A one-line summary for a run report. */
                std::string to_string() const;

            private:
                SushiBLAS::Engine* engine_;
                SushiBLAS::Tensor factor_;
                SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena_;
                std::vector<SushiBLAS::Tensor> count_views_;
                std::int64_t* counts_ = nullptr;
                std::size_t count_ = 0;
                LossScaleState state_;
        };

    } // namespace Optim
} // namespace SushiAI

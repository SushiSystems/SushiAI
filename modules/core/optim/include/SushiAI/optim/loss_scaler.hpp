/**************************************************************************/
/* loss_scaler.hpp                                                        */
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
 * @file loss_scaler.hpp
 * @brief Declares dynamic gradient loss scaling utilities for mixed precision.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

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

            /** @brief What the scale is multiplied by after a clean run; a power of two >= 1. */
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
                 * @brief Builds a state machine over @p options, or refuses the options.
                 * @param options The constants; each must satisfy its documented bound.
                 * @return The state machine, or InvalidDescription when a constant is not a
                 *         power of two, when the bounds do not bracket the initial scale, when
                 *         growth is below one or backoff is not in (0, 1], or when the growth
                 *         interval is zero.
                 */
                [[nodiscard]] static Result<LossScaleState> make(LossScaleOptions options = {});

                /**
                 * @brief Folds one step's verdict into the state.
                 *
                 * @param overflow Whether any gradient came back non-finite.
                 */
                void observe(bool overflow) noexcept;

                /**
                 * @brief Sets the scale and the clean-step count a saved run left off at.
                 * @return Incompatible for a scale that is no power of two within the bounds
                 *         of the options, and for a count at or above the growth interval.
                 */
                [[nodiscard]] Status restore(double scale, std::uint64_t good_steps);

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
                /**
                 * @brief Holds @p options as given and starts at their initial scale.
                 * @param options Constants already checked by make().
                 */
                explicit LossScaleState(LossScaleOptions options) noexcept;

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
                 * @brief Builds a loss scaler bound to a device factor and a count of gradients.
                 * @param factor Single-element float32 tensor scaled for backward seeding.
                 * @param parameter_count Number of parameter gradients screened; must be positive.
                 * @return The scaler, or InvalidDescription for a factor that is not one float32
                 *         element, a zero parameter count, or options that LossScaleState::make
                 *         refuses; OutOfMemory when the count buffer cannot be allocated.
                 */
                [[nodiscard]] static Result<std::unique_ptr<LossScaler>> make(
                    SushiBLAS::Engine& engine,
                    SushiBLAS::Tensor factor,
                    std::size_t parameter_count,
                    LossScaleOptions options = {});

                ~LossScaler() = default;

                LossScaler(const LossScaler&) = delete;
                LossScaler& operator=(const LossScaler&) = delete;
                LossScaler(LossScaler&&) = delete;
                LossScaler& operator=(LossScaler&&) = delete;

                /**
                 * @brief Records non-finite gradient count kernels for each parameter.
                 * @param recorder Recorder the count tasks are added to; must outlive the call.
                 * @param gradients Pointers to gradient tensors in slot order.
                 * @return The number of gradients screened, or a failure from the seam.
                 */
                [[nodiscard]] Result<std::size_t> record_screens(
                    SushiBLAS::TaskRecorder& recorder,
                    const std::vector<SushiBLAS::Tensor*>& gradients);

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
                 * @brief Sets the scale and the clean-step count a saved run left off at.
                 * @return What LossScaleState::restore returns.
                 */
                [[nodiscard]] Status restore(double scale, std::uint64_t good_steps);

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
                /**
                 * @brief Takes ownership of the device buffers and the state machine.
                 * @param factor The scale tensor; moved in.
                 * @param arena The storage the count views carve from; moved in.
                 * @param count_views One int64 view per gradient, in slot order; moved in.
                 * @param state The state machine; moved in.
                 * @pre count_views is not empty; make() refuses a zero count first.
                 */
                LossScaler(SushiBLAS::Tensor factor,
                           SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                           std::vector<SushiBLAS::Tensor> count_views,
                           LossScaleState state) noexcept;

                SushiBLAS::Tensor factor_;
                SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena_;
                std::vector<SushiBLAS::Tensor> count_views_;
                std::int64_t* counts_ = nullptr;
                std::size_t count_ = 0;
                LossScaleState state_;
        };

    } // namespace Optim
} // namespace SushiAI

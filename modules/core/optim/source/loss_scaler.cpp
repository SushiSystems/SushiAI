/**************************************************************************/
/* loss_scaler.cpp                                                        */
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
 * @file loss_scaler.cpp
 * @brief Implements dynamic gradient loss scaling utilities for mixed precision.
 * @author Mustafa Garip
 */

#include <cmath>
#include <utility>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/optim/loss_scaler.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /**
             * @brief Determines whether @p value is an exact positive power of two.
             *
             * Splits double into significand and exponent via std::frexp.
             * Returns true only when significand equals 0.5.
             */
            bool is_power_of_two(double value) noexcept
            {
                if (!(value > 0.0) || !std::isfinite(value)) return false;
                int exponent = 0;
                return std::frexp(value, &exponent) == 0.5;
            }

            /**
             * @brief Carves one int64 view per parameter out of @p arena, in slot order.
             * @param engine The engine whose default layout the views take.
             * @param arena Storage the views carve from; must outlive the views.
             * @param parameter_count Number of views to carve.
             * @return The views, or the failure of the allocation seam.
             */
            Result<std::vector<SushiBLAS::Tensor>> carve_count_views(
                SushiBLAS::Engine& engine,
                const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
                std::size_t parameter_count)
            {
                return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
                {
                    std::vector<SushiBLAS::Tensor> views;
                    views.reserve(parameter_count);
                    for (std::size_t i = 0; i < parameter_count; ++i)
                        views.push_back(view_tensor(engine, arena, i * dtype_size(DType::Int64),
                                                    Shape{1}, DType::Int64));
                    return views;
                });
            }
        }

        Result<LossScaleState> LossScaleState::make(LossScaleOptions options) noexcept
        {
            // Validate all scale bounds and factors are powers of two.
            SA_FAIL_IF(!is_power_of_two(options.initial), StatusCode::InvalidDescription,
                       "The initial loss scale must be a power of two; got {}. Only a dyadic "
                       "scale is exact in binary floating point, and exactness is what lets "
                       "the unscale recover the gradient bit for bit instead of adding a "
                       "rounding error nothing downstream could tell from a real one.",
                       options.initial);
            SA_FAIL_IF(!is_power_of_two(options.growth) || options.growth < 1.0,
                       StatusCode::InvalidDescription,
                       "The loss-scale growth factor must be a power of two of at least 1; got {}",
                       options.growth);
            SA_FAIL_IF(!is_power_of_two(options.backoff) || options.backoff > 1.0,
                       StatusCode::InvalidDescription,
                       "The loss-scale backoff factor must be a power of two in (0, 1]; got {}",
                       options.backoff);
            SA_FAIL_IF(!is_power_of_two(options.minimum) || !is_power_of_two(options.maximum),
                       StatusCode::InvalidDescription,
                       "The loss-scale bounds must both be powers of two; got [{}, {}]",
                       options.minimum, options.maximum);
            SA_FAIL_IF(options.minimum > options.maximum, StatusCode::InvalidDescription,
                       "The loss-scale floor {} is above its ceiling {}",
                       options.minimum, options.maximum);
            SA_FAIL_IF(options.initial < options.minimum || options.initial > options.maximum,
                       StatusCode::InvalidDescription,
                       "The initial loss scale {} lies outside the bounds [{}, {}]",
                       options.initial, options.minimum, options.maximum);
            SA_FAIL_IF(options.growth_interval == 0, StatusCode::InvalidDescription,
                       "The loss-scale growth interval must be positive; a zero interval would "
                       "double the scale after every clean step and spend the run oscillating "
                       "between overflow and backoff.");

            return LossScaleState(options);
        }

        LossScaleState::LossScaleState(LossScaleOptions options) noexcept
            : options_(options), scale_(options.initial)
        {

        }

        Status LossScaleState::restore(double scale, std::uint64_t good_steps) noexcept
        {
            SA_FAIL_IF(!is_power_of_two(scale) || scale < options_.minimum ||
                           scale > options_.maximum,
                       StatusCode::Incompatible,
                       "a loss scale of {} is no power of two within the bounds [{}, {}] of "
                       "this loss scaler", scale, options_.minimum, options_.maximum);
            SA_FAIL_IF(good_steps >= options_.growth_interval, StatusCode::Incompatible,
                       "{} clean steps since the loss scale last moved is not below the growth "
                       "interval of {}", good_steps, options_.growth_interval);

            scale_ = scale;
            good_ = good_steps;
            skipped_ = false;
            return Status();
        }

        void LossScaleState::observe(bool overflow) noexcept
        {
            skipped_ = overflow;

            if (overflow)
            {
                // Reset clean step counter immediately on detected gradient overflow.
                ++skips_;
                good_ = 0;

                const double backed_off = scale_ * options_.backoff;
                if (backed_off >= options_.minimum)
                {
                    scale_ = backed_off;
                    ++backoffs_;
                }
                return;
            }

            ++good_;
            if (good_ < options_.growth_interval) return;

            good_ = 0;

            const double grown = scale_ * options_.growth;
            if (grown <= options_.maximum)
            {
                scale_ = grown;
                ++growths_;
            }
        }

        Result<std::unique_ptr<LossScaler>> LossScaler::make(SushiBLAS::Engine& engine,
                                                             SushiBLAS::Tensor factor,
                                                             std::size_t parameter_count,
                                                             LossScaleOptions options) noexcept
        {
            SA_FAIL_IF(factor.num_elements != 1, StatusCode::InvalidDescription,
                       "A loss scaler's factor must be a single element; got {}",
                       factor.num_elements);
            SA_FAIL_IF(factor.dtype != DType::Float32, StatusCode::InvalidDescription,
                       "A loss scaler's factor must be float32; got {}. The graph value it "
                       "backs is multiplied into the cross-entropy tail, which the precision "
                       "policy keeps in the master type for exactly the reason the scale is "
                       "applied there.",
                       dtype_name(factor.dtype));
            SA_FAIL_IF(parameter_count == 0, StatusCode::InvalidDescription,
                       "A loss scaler with no parameters to screen would report every step "
                       "clean and never back off.");

            SA_TRY_ASSIGN(auto state, LossScaleState::make(options));

            // Allocate contiguous int64 arena partitioned into 1-element parameter views.
            SA_TRY_ASSIGN(auto arena,
                          allocate_arena(engine, parameter_count * dtype_size(DType::Int64)));
            SA_TRY_ASSIGN(auto count_views, carve_count_views(engine, arena, parameter_count));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                std::unique_ptr<LossScaler> scaler(new LossScaler(std::move(factor),
                                                                  std::move(arena),
                                                                  std::move(count_views),
                                                                  std::move(state)));

                // Initialize count buffer to zero before initial execution replay.
                for (std::size_t i = 0; i < scaler->count_; ++i) scaler->counts_[i] = 0;

                // Set initial loss scale factor before recording execution steps.
                scaler->factor_.data_as<float>()[0] = static_cast<float>(scaler->state_.scale());
                return scaler;
            });
        }

        LossScaler::LossScaler(SushiBLAS::Tensor factor,
                               SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                               std::vector<SushiBLAS::Tensor> count_views,
                               LossScaleState state) noexcept
            : factor_(std::move(factor)), arena_(std::move(arena)),
              count_views_(std::move(count_views)), count_(count_views_.size()),
              state_(std::move(state))
        {
            counts_ = count_views_.front().data_as<std::int64_t>();
        }

        Result<std::size_t> LossScaler::record_screens(
            SushiBLAS::TaskRecorder& recorder,
            const std::vector<SushiBLAS::Tensor*>& gradients) noexcept
        {
            SA_FATAL_IF(gradients.size() != count_,
                        "This loss scaler screens {} gradients but was handed {}; the gate's "
                        "slot index is a position in this list, so the two must agree",
                        count_, gradients.size());

            return Detail::guard_value(Detail::SeamPhase::Record, [&]()
            {
                SushiBLAS::ReductionOps reductions(recorder);
                for (std::size_t i = 0; i < count_; ++i)
                {
                    SA_FATAL_IF(gradients[i] == nullptr,
                                "Gradient {} is null; every parameter must have one to screen", i);

                    // Screen gradient non-finite values into parameter count view.
                    reductions.count_nonfinite(*gradients[i], count_views_[i]);
                }

                return count_;
            });
        }

        void LossScaler::before_replay() noexcept
        {
            factor_.data_as<float>()[0] = static_cast<float>(state_.scale());
        }

        Status LossScaler::restore(double scale, std::uint64_t good_steps) noexcept
        {
            return state_.restore(scale, good_steps);
        }

        void LossScaler::after_replay() noexcept
        {
            // Check global non-finite count across all screened parameter buffers.
            bool overflow = false;
            for (std::size_t i = 0; i < count_; ++i)
                if (counts_[i] != 0) { overflow = true; break; }

            state_.observe(overflow);
        }

        std::string LossScaler::to_string() const noexcept
        {
            // Format loss scale metrics reporting overflowed steps, backoffs, and growths.
            return SushiLog::format_text(
                "loss scale     : {} (steps overflowed: {}, backoffs: {}, growths: {})\n",
                SushiLog::fixed(state_.scale(), 0), state_.skipped_steps(), state_.backoffs(),
                state_.growths());
        }

    } // namespace Optim
} // namespace SushiAI

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
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/allocation.hpp>
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
        }

        LossScaleState::LossScaleState(LossScaleOptions options)
            : options_(options), scale_(options.initial)
        {
            // Validate all scale bounds and factors are powers of two.
            SA_THROW_IF(!is_power_of_two(options_.initial),
                        "The initial loss scale must be a power of two; got {}. Only a dyadic "
                        "scale is exact in binary floating point, and exactness is what lets "
                        "the unscale recover the gradient bit for bit instead of adding a "
                        "rounding error nothing downstream could tell from a real one.",
                        options_.initial);
            SA_THROW_IF(!is_power_of_two(options_.growth) || options_.growth < 1.0,
                        "The loss-scale growth factor must be a power of two of at least 1; got {}",
                        options_.growth);
            SA_THROW_IF(!is_power_of_two(options_.backoff) || options_.backoff > 1.0,
                        "The loss-scale backoff factor must be a power of two in (0, 1]; got {}",
                        options_.backoff);
            SA_THROW_IF(!is_power_of_two(options_.minimum) || !is_power_of_two(options_.maximum),
                        "The loss-scale bounds must both be powers of two; got [{}, {}]",
                        options_.minimum, options_.maximum);
            SA_THROW_IF(options_.minimum > options_.maximum,
                        "The loss-scale floor {} is above its ceiling {}",
                        options_.minimum, options_.maximum);
            SA_THROW_IF(options_.initial < options_.minimum || options_.initial > options_.maximum,
                        "The initial loss scale {} lies outside the bounds [{}, {}]",
                        options_.initial, options_.minimum, options_.maximum);
            SA_THROW_IF(options_.growth_interval == 0,
                        "The loss-scale growth interval must be positive; a zero interval would "
                        "double the scale after every clean step and spend the run oscillating "
                        "between overflow and backoff.");
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

        LossScaler::LossScaler(SushiBLAS::Engine& engine,
                               SushiBLAS::Tensor factor,
                               std::size_t parameter_count,
                               LossScaleOptions options)
            : engine_(&engine), factor_(std::move(factor)), count_(parameter_count),
              state_(options)
        {
            SA_THROW_IF(factor_.num_elements != 1,
                        "A loss scaler's factor must be a single element; got {}",
                        factor_.num_elements);
            SA_THROW_IF(factor_.dtype != DType::FLOAT32,
                        "A loss scaler's factor must be float32; got {}. The graph value it "
                        "backs is multiplied into the cross-entropy tail, which the precision "
                        "policy keeps in the master type for exactly the reason the scale is "
                        "applied there.",
                        dtype_name(factor_.dtype));
            SA_THROW_IF(parameter_count == 0,
                        "A loss scaler with no parameters to screen would report every step "
                        "clean and never back off.");

            // Allocate contiguous int64 arena partitioned into 1-element parameter views.
            arena_ = allocate_arena(engine, parameter_count * dtype_size(DType::INT64));

            count_views_.reserve(parameter_count);
            for (std::size_t i = 0; i < parameter_count; ++i)
                count_views_.push_back(view_tensor(engine, arena_,
                                                   i * dtype_size(DType::INT64),
                                                   Shape{1}, DType::INT64));

            counts_ = count_views_.front().data_as<std::int64_t>();

            // Initialize count buffer to zero before initial execution replay.
            for (std::size_t i = 0; i < parameter_count; ++i) counts_[i] = 0;

            // Set initial loss scale factor before recording execution steps.
            factor_.data_as<float>()[0] = static_cast<float>(state_.scale());
        }

        std::size_t LossScaler::record_screens(const std::vector<SushiBLAS::Tensor*>& gradients)
        {
            SA_THROW_IF(gradients.size() != count_,
                        "This loss scaler screens {} gradients but was handed {}; the gate's "
                        "slot index is a position in this list, so the two must agree",
                        count_, gradients.size());

            for (std::size_t i = 0; i < count_; ++i)
            {
                SA_THROW_IF(gradients[i] == nullptr,
                            "Gradient {} is null; every parameter must have one to screen", i);

                // Screen gradient non-finite values into parameter count view.
                engine_->reductions().count_nonfinite(*gradients[i], count_views_[i]);
            }

            return count_;
        }

        void LossScaler::before_replay()
        {
            factor_.data_as<float>()[0] = static_cast<float>(state_.scale());
        }

        void LossScaler::after_replay()
        {
            // Check global non-finite count across all screened parameter buffers.
            bool overflow = false;
            for (std::size_t i = 0; i < count_; ++i)
                if (counts_[i] != 0) { overflow = true; break; }

            state_.observe(overflow);
        }

        std::string LossScaler::to_string() const
        {
            // Format loss scale metrics reporting overflowed steps, backoffs, and growths.
            return SushiBLAS::Core::format(
                "loss scale     : {:.0f} (steps overflowed: {}, backoffs: {}, growths: {})\n",
                state_.scale(), state_.skipped_steps(), state_.backoffs(), state_.growths());
        }

    } // namespace Optim
} // namespace SushiAI

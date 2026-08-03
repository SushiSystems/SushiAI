/**************************************************************************/
/* loss_scaler.cpp                                                        */
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
             * @brief Whether @p value is exactly 2^k for some integer k.
             *
             * Via the significand rather than by bit-twiddling: @c std::frexp
             * splits a finite positive double into a significand in [0.5, 1) and
             * an exponent, so the significand is exactly 0.5 for a power of two
             * and nothing else. It also answers false for zero, infinities and
             * NaN without a special case, which is what the validation below
             * wants.
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
            // Every constant is checked for being a power of two, and the
            // message says why rather than only what: a scale of 3 works, it
            // simply makes the unscale inexact, so this would otherwise look
            // like arbitrary pedantry to whoever hits it.
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
                // Immediately, and the counter restarts: the run has just proved
                // the current scale is too high, so the evidence that earned the
                // pending growth is stale.
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

            // One allocation for every count, taken here and never again - the
            // steady state of a training loop allocates nothing, and a screen
            // that allocated per step would be the one thing in the plan that
            // did. `count_nonfinite` insists on a single-element INT64 result,
            // so each parameter gets a 1-element view rather than an index into
            // one wide tensor.
            arena_ = allocate_arena(engine, parameter_count * dtype_size(DType::INT64));

            count_views_.reserve(parameter_count);
            for (std::size_t i = 0; i < parameter_count; ++i)
                count_views_.push_back(view_tensor(engine, arena_,
                                                   i * dtype_size(DType::INT64),
                                                   Shape{1}, DType::INT64));

            counts_ = count_views_.front().data_as<std::int64_t>();

            // Defined before anything can read them. The screens overwrite these
            // on every replay, so the only window in which the initial value is
            // visible is a host read taken before the first step - but a buffer
            // whose contents are "whatever the allocator last held" is a hazard
            // that costs one loop to remove.
            for (std::size_t i = 0; i < parameter_count; ++i) counts_[i] = 0;

            // The same argument for the factor: the first replay's DYNAMIC_SCALE
            // reads it, and before_replay() will have written it, but a defined
            // starting value means a caller who records before stepping does not
            // multiply the gradient by garbage.
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

                // Recorded here and read from a kernel later in the same plan.
                // The edge that orders the two is not declared by anyone: it is
                // the write-after-read hazard on the gradient itself, which this
                // call reads and the fused update declares as a write. See the
                // header's "The gate is per parameter".
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
            // Global, unlike the per-parameter gate the kernels read: one
            // non-finite element anywhere means the scale is too high for this
            // model, whichever tensor found out.
            bool overflow = false;
            for (std::size_t i = 0; i < count_; ++i)
                if (counts_[i] != 0) { overflow = true; break; }

            state_.observe(overflow);
        }

        std::string LossScaler::to_string() const
        {
            // "overflowed", not "skipped": the gate is per parameter, so on such
            // a step the parameters whose own gradients stayed finite did
            // update. Printing it as a skip would read as "nothing moved",
            // which is not what happened - see the header's "The gate is per
            // parameter, and that is a real limitation".
            return SushiBLAS::Core::format(
                "loss scale     : {:.0f} ({} overflowed steps, {} backoffs, {} growths)\n",
                state_.scale(), state_.skipped_steps(), state_.backoffs(), state_.growths());
        }

    } // namespace Optim
} // namespace SushiAI

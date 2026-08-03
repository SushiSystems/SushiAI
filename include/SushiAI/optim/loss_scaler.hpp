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

#pragma once

/**
 * @file loss_scaler.hpp
 * @brief Gradient loss scaling: the numerics half of mixed precision.
 *
 * @c graph/precision.hpp and @c graph/cast_insertion.hpp put fp16 where the
 * FLOPs are and fp32 everywhere it matters. What they do not address is
 * fp16's **exponent range**, and that is what this file is for.
 *
 * @section why_scaling What loss scaling is actually for
 *
 * binary16's smallest normal is 6.1e-5 and its smallest subnormal 6.0e-8;
 * below that a value is exactly zero. A gradient that would have been 1e-8 in
 * fp32 does not arrive small, it arrives *gone*, and a weight whose gradient is
 * zero does not move. Multiplying the loss gradient by a large constant S
 * shifts the whole backward pass up by S - every gradient below the seed is
 * linear in it - and dividing by S before the update puts it back. The
 * arithmetic in between happens in a part of fp16's range that exists.
 *
 * @section honest What this does not do on this machine
 *
 * Nothing observable. On the demo workload fp16 without loss scaling neither
 * stalls nor diverges: both arms converge to the same loss, and at
 * `--hidden 512 --batch-size 64` both reach 0.000003 after 48 steps. This is
 * justified by theory and by models deeper than the demo's two layers, and it
 * is written down here rather than dressed up, because the tests that matter
 * are the ones that construct an overflow deliberately - there is no natural
 * one to wait for. See @c tests/functional/regression/test_loss_scaling.cpp.
 *
 * @section powers Why every constant is a power of two
 *
 * Scaling by a power of two changes only a float's exponent field, so it is
 * **exact** - the unscale recovers the original gradient bit for bit, with no
 * rounding anywhere in the round trip. That is the whole reason the growth
 * factor is 2 and the backoff 0.5 rather than, say, 3 and 1/3: a non-dyadic
 * scale would make every gradient in the model carry a rounding error that
 * exists for no reason and that nothing downstream could distinguish from a
 * real one. @c Regression_LossScaling.UnscalingByAPowerOfTwoIsExact asserts the
 * round trip with @c EXPECT_EQ, not @c EXPECT_NEAR, because approximately is
 * the wrong answer here and asserting it approximately would hide a defect.
 *
 * @section machine The state machine
 *
 * Standard dynamic scaling, and deliberately the same shape as PyTorch's
 * @c GradScaler so that its behaviour is the reference rather than a local
 * invention: start high, **halve and skip the step** whenever any gradient came
 * back non-finite, and double after @c LossScaleOptions::growth_interval
 * consecutive clean steps. Growth is slow and backoff is immediate because the
 * costs are asymmetric - an overshoot costs one thrown-away step, an
 * undershoot costs underflow nobody reports.
 *
 * @section split Where the decision is taken, and why it is split
 *
 * Two decisions, and they have different deadlines, so they live in different
 * places.
 *
 * - **Skipping the update** must be decided *inside* the replay, because the
 *   overflow is detected by @c reductions().count_nonfinite() after the backward
 *   pass and the update runs in the same compiled plan a few tasks later. So the
 *   gate is a **device** read: the optimizer's fused kernel loads this object's
 *   count for its parameter and returns before touching anything. @ref gate is
 *   that pointer.
 * - **Choosing the next scale** has until the *next* step's forward pass, so it
 *   is taken on the **host**, in @ref after_replay, from a buffer the device
 *   has already been waited on. @c docs/slop/make_sushiai_great_again.md §21
 *   forbids a synchronous host read inside forward/backward; this is the step
 *   boundary, which is outside it and where @c Train::Trainer::run_step already
 *   syncs to read the loss. Keeping the state machine on the host also keeps it
 *   *readable*: @ref scale and @ref skipped_steps are facts a test or a report
 *   can assert without adding a synchronisation that the run would not otherwise
 *   pay. The all-device alternative - a one-element kernel per step advancing
 *   the scale - costs one more task per step and makes every one of those
 *   accessors a device round trip.
 *
 * @section skip Why the skip has to be a branch, and not a zero
 *
 * The tempting cheap skip is to neutralise the update through the scalars it
 * already reads: @c STEP_SIZE = 0, @c BETA1 = 1, @c ONE_MINUS_BETA1 = 0, so
 * every line becomes an identity and no kernel changes. **It is wrong, and it
 * fails exactly on the step it exists for.** The gradient on a skipped step is
 * an infinity or a NaN by definition, and `0 * inf` is NaN in IEEE-754 - so
 * `m = 1*m + 0*g` writes NaN into the first moment, the optimizer never
 * recovers, nothing throws, and the loss becomes NaN a step later with the
 * cause three layers away. The gate is therefore a real early return, and
 * @c Regression_LossScaling.ASkippedStepLeavesTheMomentsBitIdentical asserts
 * the *moments*, not only the parameters, because a test that checked the
 * parameters alone would pass while the moments were quietly NaN.
 *
 * @section granularity The gate is per parameter, and that is a real limitation
 *
 * One @c count_nonfinite runs per parameter gradient and the update for that
 * parameter reads its own count. The ordering that makes that sound is not an
 * assumption: @c count_nonfinite *reads* the gradient and the fused update
 * declares the gradient as a **write** (@c execute_zip_inplace declares every
 * operand both ways), so SushiRuntime's tracker sees a write-after-read hazard
 * and orders the update after the screen. That is also why @ref record_screens
 * must run before the updates are recorded, and why it is called from
 * @c Optimizer rather than left to a caller's discipline.
 *
 * What it does **not** give is PyTorch's whole-step skip, where one overflowed
 * parameter freezes every parameter. Reaching that would need each update to
 * read *every* count, and there is no hazard between the update of parameter A
 * and the screen of parameter B to order it - the two touch no common bytes.
 * Getting one would mean a task that declares the whole count buffer as a read
 * and a gradient as a write, which the @c execute_zip_inplace seam cannot
 * express and which would therefore be a hand-written task. The host half of
 * the decision *is* global - any non-zero count halves the scale and suppresses
 * the step counter - so the two differ only in the mixed case.
 *
 * **And the mixed case is not hypothetical.** `sa demo mlp --precision mixed
 * --loss-scale 16777216` reports 8 overflowed steps out of 8 and the loss still
 * moves, because on each of them some gradients were finite and their
 * parameters updated. Nothing unsafe happens - no infinity and no NaN reaches a
 * parameter or a moment, which is what the regression suite asserts - but it is
 * a *partial* step, not the empty one PyTorch would take. That is why the
 * report says "overflowed steps" rather than "skipped": calling it a skip would
 * claim something stronger than what happened.
 *
 * Device lane: this header reaches SushiBLAS.
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
         * @struct LossScaleOptions
         * @brief The four numbers dynamic scaling has, and their bounds.
         *
         * Every one of them is a power of two, and @c LossScaleState refuses
         * anything else - see the header's "Why every constant is a power of
         * two". They are doubles rather than ints because they are magnitudes
         * and a reader thinks in 65536, not in 16.
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
         * @brief The dynamic-scaling state machine, with no device in it.
         *
         * Split out from @c LossScaler so the *rule* can be tested without an
         * engine, a graph or a step: growth after exactly N clean steps, the
         * control at N-1, and the clamps are properties of this class alone and
         * a unit test asserts them directly. Everything device-shaped lives one
         * class up.
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
                 * @brief Binds a scaler to the graph value the backward pass multiplies by.
                 *
                 * @param engine The engine the screen is recorded on and the
                 *        count buffer allocated from; must outlive this object.
                 * @param factor The 1-element FLOAT32 tensor behind the value
                 *        handed to @c Autograd::differentiate() as its seed
                 *        factor. Written from the host before every replay; a
                 *        copy of the handle, so it shares the pool's Storage.
                 * @param parameter_count How many parameter gradients will be
                 *        screened; sizes the count buffer once, here, so the
                 *        steady state allocates nothing.
                 * @param options The state machine's constants.
                 * @throws Error If @p factor is not a single FLOAT32 element, if
                 *         @p parameter_count is zero, or if @p options is not
                 *         well formed.
                 */
                LossScaler(SushiBLAS::Engine& engine,
                           SushiBLAS::Tensor factor,
                           std::size_t parameter_count,
                           LossScaleOptions options = {});

                LossScaler(const LossScaler&) = delete;
                LossScaler& operator=(const LossScaler&) = delete;

                /**
                 * @brief Records one @c count_nonfinite per gradient into the plan.
                 *
                 * **Must be recorded before the updates are.** The write-after-read
                 * hazard on the gradient is what orders each update after its own
                 * screen, and a hazard only orders the later task after the earlier
                 * one - recording these afterwards would order them after the
                 * updates instead, and every gate would read the previous step's
                 * verdict while still converging. @c Optimizer owns the sequence
                 * for that reason; see the header.
                 *
                 * @param gradients One gradient per parameter, in the order the
                 *        optimizer's slots are in - the gate's slot index is a
                 *        position in this list.
                 * @return How many device tasks were recorded.
                 * @throws Error If @p gradients is not exactly @c parameter_count long.
                 */
                std::size_t record_screens(const std::vector<SushiBLAS::Tensor*>& gradients);

                /**
                 * @brief Publishes this step's scale into device memory.
                 *
                 * Host-only, and called where @c IStepRecorder::before_replay()
                 * is: after the previous replay was waited on and before the
                 * next is dispatched, which is what makes writing a buffer the
                 * recorded kernels read safe with no dependency edge. See
                 * @c optim/step_scalars.hpp, which makes the same argument at
                 * greater length.
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
                 * @brief The address an optimizer's kernel captures to gate on.
                 *
                 * Element @c i is the number of non-finite elements in
                 * parameter @c i's gradient, which is zero on a healthy step.
                 * Stable for the life of this object.
                 *
                 * @return A pointer to element 0 of the count buffer.
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

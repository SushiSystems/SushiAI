/**************************************************************************/
/* gradcheck.hpp                                                          */
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
 * @file gradcheck.hpp
 * @brief Finite-difference verification of an analytic gradient.
 *
 * A backward rule that is subtly wrong - a transpose the wrong way round on a
 * square matrix, a factor of N, the forward output where the forward input
 * belonged - produces a model that still trains, just worse. Nothing throws
 * and no test fails unless something compares the analytic gradient against
 * one derived from the function's own values. That is what this does.
 *
 * @section fp64 Double precision is required, not preferred
 *
 * A central difference has two error terms: truncation, which falls as h^2,
 * and cancellation, which rises as eps/h. Minimising their sum puts the best
 * achievable relative accuracy at roughly eps^(2/3). In binary32 that is about
 * 2.4e-5 - the same order as the discrepancy a genuinely wrong gradient
 * produces, so an fp32 check cannot distinguish "correct" from "wrong by a
 * little". In binary64 the floor is about 4e-11, five orders of magnitude
 * below the 1e-5 tolerance used here, so a failure means a real disagreement.
 * Every function passed in must therefore evaluate in FLOAT64 end to end -
 * including the device work behind it.
 *
 * @section mixed This checker and mixed precision are incompatible, and that is fine
 *
 * Stated plainly rather than left as a gap somebody trips over. A graph built
 * under @c Graph::MixedPrecisionPolicy evaluates its matrix products in fp16,
 * whose eps is 9.8e-4; the same derivation puts the best achievable relative
 * accuracy of a difference quotient there at about eps^(2/3) = 1e-2, which is
 * three orders of magnitude *worse* than the disagreement a wrong backward
 * rule produces. A gradcheck over such a graph would not be a weak test, it
 * would be noise - and every way of "fixing" it lowers the bar until the check
 * stops being able to fail for the right reason.
 *
 * The right response is not to weaken this file. What a gradient rule states
 * is a fact about the *mathematics*, and mixed precision is a fact about the
 * arithmetic underneath it: the rule for @c MATMUL is the same rule whichever
 * element type evaluates it, so checking it once in fp64 checks it for every
 * precision. What mixed precision can then get wrong is not the rule but the
 * plumbing - which operand carries which type, and where the conversions sit -
 * and that is a graph-shape question, answered by
 * @c Graph::find_dtype_mismatches and the tests around it rather than by a
 * finite difference.
 *
 * @section kinks Kinks
 *
 * ReLU-family activations are not differentiable at zero, and a difference
 * quotient straddling the kink reports the average of the two one-sided
 * derivatives, which no correct backward rule will ever match. Sample away
 * from it: push_off_kinks() moves any coordinate closer to zero than a few
 * step sizes out of the way before the check runs.
 *
 * @section modes Two modes
 *
 * The full Jacobian probes every coordinate, which is 2n function evaluations
 * and is the right thing for a single operation with a handful of inputs. For
 * a whole model it is not affordable, so the directional mode instead compares
 * u^T g against the difference quotient along a random unit direction u - two
 * evaluations per direction, independent of n, and a wrong gradient survives k
 * random directions with vanishing probability.
 *
 * Host lane: no SushiBLAS, no SYCL. The caller supplies the evaluations, so
 * the same checker serves a device model and a closed-form host function.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @struct GradCheckOptions
         * @brief The knobs, with the defaults the derivation above justifies.
         */
        struct GradCheckOptions
        {
            /** @brief Relative-error ceiling for a coordinate to count as matching. */
            double tolerance = 1e-5;

            /**
             * @brief Step size relative to the coordinate's magnitude.
             *
             * Near eps^(1/3) for binary64, which is where the truncation and
             * cancellation terms of a central difference cross.
             */
            double step_scale = 6.1e-6;

            /** @brief Absolute floor on the step, for coordinates at or near zero. */
            double min_step = 1e-8;

            /**
             * @brief Denominator floor in the relative-error ratio.
             *
             * Below this the comparison falls back to an absolute one, so a
             * gradient that is legitimately zero does not fail on a 1e-12
             * difference divided by itself.
             */
            double error_floor = 1e-6;

            /** @brief How many step sizes a coordinate must clear a kink by. */
            double kink_margin = 10.0;

            /** @brief Seed for the directional mode's random directions. */
            std::uint64_t seed = 0x9E3779B97F4A7C15ull;
        };

        /**
         * @struct GradCheckResult
         * @brief What a check found.
         */
        struct GradCheckResult
        {
            /** @brief Whether every probe stayed inside the tolerance. */
            bool passed = false;

            /** @brief The largest relative error seen. */
            double max_relative_error = 0.0;

            /** @brief Which coordinate (full mode) or direction (directional) was worst. */
            std::size_t worst_index = 0;

            /** @brief The analytic value at the worst probe. */
            double worst_analytic = 0.0;

            /** @brief The finite-difference value at the worst probe. */
            double worst_numeric = 0.0;

            /** @brief How many probes ran. */
            std::size_t probes = 0;

            /** @brief A one-line summary suitable for a test failure message. */
            std::string to_string() const;
        };

        /**
         * @brief Evaluates the scalar under test at a point.
         *
         * @param inputs The point, in FLOAT64.
         * @return The scalar value at that point.
         */
        using ScalarFunction = std::function<double(SushiRuntime::span<const double> inputs)>;

        /**
         * @brief Fills @p gradient with the analytic gradient at @p inputs.
         *
         * @param inputs The point, in FLOAT64.
         * @param gradient The output, exactly as long as @p inputs.
         */
        using GradientFunction = std::function<void(SushiRuntime::span<const double> inputs,
                                                    SushiRuntime::span<double> gradient)>;

        /**
         * @brief Moves coordinates off a kink at zero, in place.
         *
         * @param inputs The point to adjust.
         * @param options Supplies the step size and the margin.
         */
        void push_off_kinks(SushiRuntime::span<double> inputs, const GradCheckOptions& options = {});

        /**
         * @brief Compares every coordinate of the gradient against a central difference.
         *
         * 2n evaluations of @p value plus one of @p gradient. Intended for a
         * single operation; the directional mode is the one to reach for above
         * a couple of hundred inputs.
         *
         * @param value The scalar under test.
         * @param gradient Its analytic gradient.
         * @param inputs The point to check at; restored on return.
         * @param options The tolerances and step sizes.
         * @return The comparison's outcome.
         * @throws Error If @p inputs is empty.
         */
        GradCheckResult check_full_jacobian(const ScalarFunction& value,
                                            const GradientFunction& gradient,
                                            SushiRuntime::span<double> inputs,
                                            const GradCheckOptions& options = {});

        /**
         * @brief Compares the gradient against central differences along random directions.
         *
         * 2k evaluations of @p value plus one of @p gradient, whatever n is.
         * Each direction is a unit vector drawn from a fixed, seeded stream, so
         * a failure reproduces exactly.
         *
         * @param value The scalar under test.
         * @param gradient Its analytic gradient.
         * @param inputs The point to check at; restored on return.
         * @param directions How many random directions to probe.
         * @param options The tolerances and step sizes.
         * @return The comparison's outcome.
         * @throws Error If @p inputs is empty or @p directions is zero.
         */
        GradCheckResult check_directional(const ScalarFunction& value,
                                          const GradientFunction& gradient,
                                          SushiRuntime::span<double> inputs,
                                          std::size_t directions = 8,
                                          const GradCheckOptions& options = {});

    } // namespace Autograd
} // namespace SushiAI

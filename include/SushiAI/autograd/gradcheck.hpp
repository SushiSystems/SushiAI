/**************************************************************************/
/* gradcheck.hpp                                                          */
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
 * @file gradcheck.hpp
 * @brief Declares finite-difference verification of analytic gradients.
 * @author Mustafa Garip
 */

#pragma once

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
         * @param inputs Coordinates to evaluate at; restored on return.
         * @param options Numerical step sizes and comparison tolerance.
         * @return Comparison result across all input coordinates.
         */
        GradCheckResult check_full_jacobian(const ScalarFunction& value,
                                            const GradientFunction& gradient,
                                            SushiRuntime::span<double> inputs,
                                            const GradCheckOptions& options = {});

        /**
         * @brief Compares the gradient against central differences along random directions.
         * @param inputs Coordinates to evaluate at; restored on return.
         * @param directions Number of random unit directions to probe.
         * @param options Numerical step sizes and comparison tolerance.
         * @return Comparison result across all sampled directions.
         */
        GradCheckResult check_directional(const ScalarFunction& value,
                                          const GradientFunction& gradient,
                                          SushiRuntime::span<double> inputs,
                                          std::size_t directions = 8,
                                          const GradCheckOptions& options = {});

    } // namespace Autograd
} // namespace SushiAI

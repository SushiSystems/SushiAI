/**************************************************************************/
/* gradcheck.cpp                                                          */
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
 * @file gradcheck.cpp
 * @brief Implements finite-difference gradient verification utilities.
 * @author Mustafa Garip
 */

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /** @brief The central-difference step for a coordinate of magnitude @p magnitude. */
            double step_for(double magnitude, const GradCheckOptions& options) noexcept
            {
                return std::max(options.step_scale * std::abs(magnitude), options.min_step);
            }

            /** @brief The largest absolute coordinate, used to size a directional step. */
            double magnitude_of(SushiRuntime::span<const double> inputs) noexcept
            {
                double peak = 0.0;
                for (double v : inputs) peak = std::max(peak, std::abs(v));
                return peak;
            }

            /**
             * @brief Relative error with an absolute fallback near zero.
             *
             * Dividing by max(|a|, |n|) is the right measure while either is
             * meaningfully non-zero; below the floor the ratio amplifies noise
             * into a failure, so the floor becomes the denominator and the
             * comparison degrades gracefully into an absolute one.
             */
            double relative_error(double analytic, double numeric, double floor) noexcept
            {
                const double scale = std::max({std::abs(analytic), std::abs(numeric), floor});
                return std::abs(analytic - numeric) / scale;
            }
        }

        std::string GradCheckResult::to_string() const noexcept
        {
            return SushiLog::format_text(
                "gradcheck {}: max relative error {} over {} probe(s); worst at index {} "
                "(analytic {}, numeric {})",
                passed ? "passed" : "FAILED", SushiLog::scientific(max_relative_error, 3), probes,
                worst_index, SushiLog::scientific(worst_analytic, 9),
                SushiLog::scientific(worst_numeric, 9));
        }

        void push_off_kinks(SushiRuntime::span<double> inputs, const GradCheckOptions& options) noexcept
        {
            for (double& v : inputs)
            {
                const double margin = options.kink_margin * step_for(v, options);
                if (std::abs(v) >= margin) continue;
                v = (v < 0.0 ? -margin : margin);
            }
        }

        GradCheckResult check_full_jacobian(const ScalarFunction& value,
                                            const GradientFunction& gradient,
                                            SushiRuntime::span<double> inputs,
                                            const GradCheckOptions& options) noexcept
        {
            SA_FATAL_IF(inputs.empty(), "check_full_jacobian needs at least one input");

            std::vector<double> analytic(inputs.size(), 0.0);
            gradient(SushiRuntime::span<const double>(inputs.data(), inputs.size()),
                     SushiRuntime::span<double>(analytic.data(), analytic.size()));

            GradCheckResult result;
            result.passed = true;
            result.probes = inputs.size();

            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const double original = inputs[i];
                const double h = step_for(original, options);

                inputs[i] = original + h;
                const double up = value(SushiRuntime::span<const double>(inputs.data(), inputs.size()));

                inputs[i] = original - h;
                const double down = value(SushiRuntime::span<const double>(inputs.data(), inputs.size()));

                inputs[i] = original;

                const double numeric = (up - down) / (2.0 * h);
                const double error = relative_error(analytic[i], numeric, options.error_floor);

                if (error > result.max_relative_error)
                {
                    result.max_relative_error = error;
                    result.worst_index = i;
                    result.worst_analytic = analytic[i];
                    result.worst_numeric = numeric;
                }

                if (error > options.tolerance) result.passed = false;
            }

            return result;
        }

        GradCheckResult check_directional(const ScalarFunction& value,
                                          const GradientFunction& gradient,
                                          SushiRuntime::span<double> inputs,
                                          std::size_t directions,
                                          const GradCheckOptions& options) noexcept
        {
            SA_FATAL_IF(inputs.empty(), "check_directional needs at least one input");
            SA_FATAL_IF(directions == 0, "check_directional needs at least one direction");

            std::vector<double> analytic(inputs.size(), 0.0);
            gradient(SushiRuntime::span<const double>(inputs.data(), inputs.size()),
                     SushiRuntime::span<double>(analytic.data(), analytic.size()));

            const std::vector<double> base(inputs.begin(), inputs.end());
            const double h = step_for(std::max(magnitude_of(base), 1.0), options);

            std::mt19937_64 engine(options.seed);
            std::normal_distribution<double> normal(0.0, 1.0);

            std::vector<double> direction(inputs.size(), 0.0);

            GradCheckResult result;
            result.passed = true;
            result.probes = directions;

            for (std::size_t d = 0; d < directions; ++d)
            {
                // Normalized Gaussian vectors sample uniformly on the unit sphere.
                double norm = 0.0;
                for (double& component : direction)
                {
                    component = normal(engine);
                    norm += component * component;
                }
                norm = std::sqrt(norm);
                if (norm == 0.0) { direction[0] = 1.0; norm = 1.0; }
                for (double& component : direction) component /= norm;

                double projected = 0.0;
                for (std::size_t i = 0; i < analytic.size(); ++i)
                    projected += analytic[i] * direction[i];

                for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = base[i] + h * direction[i];
                const double up = value(SushiRuntime::span<const double>(inputs.data(), inputs.size()));

                for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = base[i] - h * direction[i];
                const double down = value(SushiRuntime::span<const double>(inputs.data(), inputs.size()));

                std::copy(base.begin(), base.end(), inputs.begin());

                const double numeric = (up - down) / (2.0 * h);
                const double error = relative_error(projected, numeric, options.error_floor);

                if (error > result.max_relative_error)
                {
                    result.max_relative_error = error;
                    result.worst_index = d;
                    result.worst_analytic = projected;
                    result.worst_numeric = numeric;
                }

                if (error > options.tolerance) result.passed = false;
            }

            return result;
        }

    } // namespace Autograd
} // namespace SushiAI

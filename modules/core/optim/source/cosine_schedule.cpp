/**************************************************************************/
/* cosine_schedule.cpp                                                    */
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
 * @file cosine_schedule.cpp
 * @brief Implements the warm-up and the cosine decay of a CosineSchedule, and its check.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/cosine_schedule.hpp>

#include <cmath>
#include <cstdint>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief Holds pi, the argument at which the cosine of the decay reaches -1. */
            constexpr double K_PI = 3.14159265358979323846;
        } // namespace

        double CosineSchedule::rate_at(std::int64_t iteration) const noexcept
        {
            SA_FATAL_IF(iteration < 0, "a schedule is read at iteration 0 or later; got {}",
                        iteration);

            if (iteration < warmup_iterations)
            {
                const double fraction =
                    static_cast<double>(iteration) / static_cast<double>(warmup_iterations);
                return base_rate * fraction;
            }

            if (iteration >= total_iterations)
                return final_rate;

            // warmup_iterations <= iteration < total_iterations, so the divisor is positive.
            const double progress =
                static_cast<double>(iteration - warmup_iterations)
                / static_cast<double>(total_iterations - warmup_iterations);
            const double weight = 0.5 * (1.0 + std::cos(K_PI * progress));

            return final_rate + (base_rate - final_rate) * weight;
        }

        Status validate_cosine_schedule(const CosineSchedule& schedule)
        {
            SA_FAIL_IF(schedule.warmup_iterations < 0, StatusCode::InvalidDescription,
                       "a warm-up of {} iterations is negative", schedule.warmup_iterations);

            SA_FAIL_IF(schedule.total_iterations < schedule.warmup_iterations,
                       StatusCode::InvalidDescription,
                       "a decay that ends at iteration {} ends before the warm-up of {} "
                       "iterations does",
                       schedule.total_iterations, schedule.warmup_iterations);

            // The comparison is false for a NaN, which is refused with the negative rates.
            SA_FAIL_IF(!(schedule.final_rate >= 0.0), StatusCode::InvalidDescription,
                       "a final rate of {} is negative or not a number", schedule.final_rate);

            SA_FAIL_IF(schedule.final_rate > schedule.base_rate, StatusCode::InvalidDescription,
                       "a final rate of {} is above the base rate of {}; a cosine schedule "
                       "decays",
                       schedule.final_rate, schedule.base_rate);

            return Status();
        }

    } // namespace Optim
} // namespace SushiAI

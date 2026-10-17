/**************************************************************************/
/* step_schedule.cpp                                                      */
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
 * @file step_schedule.cpp
 * @brief Implements the learning-rate schedule and its validation.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/step_schedule.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        double StepSchedule::rate_at(std::int64_t iteration) const noexcept
        {
            SA_FATAL_IF(iteration < 0, "a schedule is read at iteration 0 or later; got {}",
                        iteration);
            SA_FATAL_IF(steps.size() != scales.size(),
                        "a schedule holds {} steps and {} scales; the lists must match",
                        steps.size(), scales.size());

            if (iteration < warmup_iterations)
            {
                const double fraction =
                    static_cast<double>(iteration) / static_cast<double>(warmup_iterations);
                return base_rate * std::pow(fraction, warmup_power);
            }

            double rate = base_rate;
            for (std::size_t i = 0; i < steps.size(); ++i)
            {
                if (steps[i] <= iteration)
                    rate *= scales[i];
            }
            return rate;
        }

        Status validate_step_schedule(const StepSchedule& schedule)
        {
            SA_FAIL_IF(schedule.steps.size() != schedule.scales.size(),
                       StatusCode::InvalidDescription,
                       "a step schedule lists {} steps and {} scales; the lists must match",
                       schedule.steps.size(), schedule.scales.size());

            SA_FAIL_IF(schedule.warmup_iterations < 0, StatusCode::InvalidDescription,
                       "a warm-up of {} iterations is negative", schedule.warmup_iterations);

            // The comparison is false for a NaN, which is refused with the negative powers.
            SA_FAIL_IF(!(schedule.warmup_power >= 0.0), StatusCode::InvalidDescription,
                       "a warm-up power of {} is negative or not a number",
                       schedule.warmup_power);

            for (std::size_t i = 1; i < schedule.steps.size(); ++i)
            {
                SA_FAIL_IF(schedule.steps[i] <= schedule.steps[i - 1],
                           StatusCode::InvalidDescription,
                           "step {} at iteration {} does not come after step {} at iteration {}; "
                           "steps must strictly ascend",
                           i, schedule.steps[i], i - 1, schedule.steps[i - 1]);
            }

            return Status();
        }

    } // namespace Optim
} // namespace SushiAI

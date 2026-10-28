/**************************************************************************/
/* run_schedule.cpp                                                       */
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
 * @file run_schedule.cpp
 * @brief Implements the reader that gives a training run one source of its learning rate.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/run_schedule.hpp>

#include <utility>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/optim/rate_schedule_section.hpp>

namespace SushiAI
{
    namespace Optim
    {
        Result<std::optional<RateSchedule>> read_rate_schedule(
            const Config::Value& schedule, const Config::Value& optimizer_attributes) noexcept
        {
            if (schedule.is_null())
                return std::optional<RateSchedule>();

            SA_FAIL_IF(optimizer_attributes.has(K_LEARNING_RATE_KEY),
                       StatusCode::InvalidDescription,
                       "the rate is stated twice: optimizer '{}' and schedule 'base_rate'. "
                       "A file with a schedule states the rate there; remove optimizer '{}'",
                       K_LEARNING_RATE_KEY, K_LEARNING_RATE_KEY);

            SA_TRY_ASSIGN(RateSchedule read, read_rate_schedule_section(schedule));

            // The comparison is false for a NaN and for the 0 a key left out leaves.
            const double base_rate = base_rate_of(read);
            SA_FAIL_IF(!(base_rate > 0.0), StatusCode::InvalidDescription,
                       "schedule 'base_rate' must be stated and above zero; got {}", base_rate);

            return std::optional<RateSchedule>(std::move(read));
        }

    } // namespace Optim
} // namespace SushiAI

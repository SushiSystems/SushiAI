/**************************************************************************/
/* apply_step_schedule.hpp                                                */
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
 * @file apply_step_schedule.hpp
 * @brief Declares the function that hands a schedule's rate to an optimizer before an update.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Sets the rate of the next update of @p optimizer from @p schedule.
         * @tparam Schedule A type with rate_at(std::int64_t): StepSchedule or CosineSchedule.
         * @pre The schedule passed its validation, and no rate it gives is negative or not
         *      finite; either stops the process.
         * @return The rate set: schedule.rate_at(optimizer.step_count() + 1).
         */
        template<typename Schedule>
        double apply_step_schedule(const Schedule& schedule, Optimizer& optimizer)
        {
            // Updates are numbered from 1, and step_count() counts the ones applied.
            const std::int64_t iteration = static_cast<std::int64_t>(optimizer.step_count()) + 1;
            const double rate = schedule.rate_at(iteration);

            optimizer.set_learning_rate(rate);
            return rate;
        }

        /**
         * @brief Sets the rate of the next update of @p optimizer from the schedule held.
         * @pre As the form over one schedule type.
         * @return The rate set: rate_at(schedule, optimizer.step_count() + 1).
         */
        double apply_step_schedule(const RateSchedule& schedule, Optimizer& optimizer);

    } // namespace Optim
} // namespace SushiAI

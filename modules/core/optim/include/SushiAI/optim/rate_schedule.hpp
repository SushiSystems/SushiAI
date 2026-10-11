/**************************************************************************/
/* rate_schedule.hpp                                                      */
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
 * @file rate_schedule.hpp
 * @brief Declares the schedule of a run, which is one of the schedules this module has.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <variant>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /** @brief Holds the schedule a training file states, whichever type it names. */
        using RateSchedule = std::variant<StepSchedule, CosineSchedule>;

        /**
         * @brief Returns the learning rate @p schedule gives update @p iteration.
         * @pre The schedule held passed its own validation; a negative iteration is fatal.
         */
        [[nodiscard]] double rate_at(const RateSchedule& schedule, std::int64_t iteration);

        /** @brief Returns the base_rate of the schedule @p schedule holds. */
        [[nodiscard]] double base_rate_of(const RateSchedule& schedule);

        /**
         * @brief Checks the schedule @p schedule holds with the validation of its own type.
         * @return What validate_step_schedule or validate_cosine_schedule returns.
         */
        [[nodiscard]] Status validate_rate_schedule(const RateSchedule& schedule);

    } // namespace Optim
} // namespace SushiAI

/**************************************************************************/
/* rate_schedule.cpp                                                      */
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
 * @file rate_schedule.cpp
 * @brief Implements the reads that are the same for every schedule a run may hold.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/rate_schedule.hpp>

#include <cstdint>
#include <variant>

namespace SushiAI
{
    namespace Optim
    {
        double rate_at(const RateSchedule& schedule, std::int64_t iteration) noexcept
        {
            return std::visit([iteration](const auto& held)
            {
                return held.rate_at(iteration);
            }, schedule);
        }

        double base_rate_of(const RateSchedule& schedule) noexcept
        {
            return std::visit([](const auto& held)
            {
                return held.base_rate;
            }, schedule);
        }

        namespace
        {
            /** @brief Checks a step schedule. */
            [[nodiscard]] Status validate_held(const StepSchedule& schedule)
            {
                return validate_step_schedule(schedule);
            }

            /** @brief Checks a cosine schedule. */
            [[nodiscard]] Status validate_held(const CosineSchedule& schedule)
            {
                return validate_cosine_schedule(schedule);
            }
        } // namespace

        Status validate_rate_schedule(const RateSchedule& schedule) noexcept
        {
            return std::visit([](const auto& held)
            {
                return validate_held(held);
            }, schedule);
        }

    } // namespace Optim
} // namespace SushiAI

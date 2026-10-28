/**************************************************************************/
/* apply_step_schedule.cpp                                                */
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
 * @file apply_step_schedule.cpp
 * @brief Implements the function that hands a schedule's rate to an optimizer.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/apply_step_schedule.hpp>

#include <variant>

namespace SushiAI
{
    namespace Optim
    {
        double apply_step_schedule(const RateSchedule& schedule, Optimizer& optimizer) noexcept
        {
            return std::visit([&optimizer](const auto& held)
            {
                return apply_step_schedule(held, optimizer);
            }, schedule);
        }

    } // namespace Optim
} // namespace SushiAI

/**************************************************************************/
/* rate_schedule_section.hpp                                              */
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
 * @file rate_schedule_section.hpp
 * @brief Declares the reader and the writer of a schedule section of any type.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/rate_schedule.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Reads the schedule section of a training file by the "type" it states.
         * @param section The "schedule" object; one that states no "type" is read as "step".
         * @return The schedule; InvalidDescription for a section that is no object, a "type"
         *         that is no string or that no schedule has, naming every type there is; or
         *         what the reader of that type refuses.
         */
        [[nodiscard]] Result<RateSchedule> read_rate_schedule_section(
            const Config::Value& section);

        /**
         * @brief Writes @p schedule as the schedule section of a training file.
         * @return The "schedule" object with the "type" of the schedule held and every key,
         *         or what the writer of that type refuses.
         */
        [[nodiscard]] Result<Config::Value> write_rate_schedule_section(
            const RateSchedule& schedule);

    } // namespace Optim
} // namespace SushiAI

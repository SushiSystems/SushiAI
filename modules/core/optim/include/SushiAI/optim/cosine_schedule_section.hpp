/**************************************************************************/
/* cosine_schedule_section.hpp                                            */
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
 * @file cosine_schedule_section.hpp
 * @brief Declares the reader and the writer of a cosine schedule section of a training file.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /** @brief Holds the "type" a schedule section states when it describes a CosineSchedule. */
        inline constexpr std::string_view K_COSINE_SCHEDULE_TYPE = "cosine";

        /**
         * @brief Reads the schedule section of a training file as a CosineSchedule.
         * @param section The "schedule" object, with its "type" or with the type taken out.
         * @return The schedule, a key left out keeping the default of its field, or
         *         InvalidDescription naming the key: a key the section does not have, a
         *         value of another kind, another "type", a rate that is not finite, or what
         *         validate_cosine_schedule refuses.
         */
        [[nodiscard]] Result<CosineSchedule> read_cosine_schedule_section(
            const Config::Value& section);

        /**
         * @brief Writes a CosineSchedule as the schedule section of a training file.
         * @return The "schedule" object with its "type" and every key, or InvalidDescription
         *         for a rate that is not finite and for what validate_cosine_schedule refuses.
         */
        [[nodiscard]] Result<Config::Value> write_cosine_schedule_section(
            const CosineSchedule& schedule);

    } // namespace Optim
} // namespace SushiAI

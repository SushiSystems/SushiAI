/**************************************************************************/
/* run_schedule.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file run_schedule.hpp
 * @brief Declares the reader that gives a training run one source of its learning rate.
 * @author Mustafa Garip
 */

#pragma once

#include <optional>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/rate_schedule.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /** @brief Holds the optimizer key that states a rate held for the whole run. */
        inline constexpr std::string_view K_LEARNING_RATE_KEY = "learning_rate";

        /**
         * @brief Reads the schedule of a run, of any type; the rate is stated by one section.
         * @param schedule The "schedule" value of a training file; null when it has none.
         * @param optimizer_attributes The optimizer's keys, as Config::TrainSpec holds them.
         * @return No schedule for a null @p schedule; the schedule; or InvalidDescription when
         *         the optimizer states "learning_rate" beside a schedule, when "base_rate" is
         *         left out or not above zero, or for what read_rate_schedule_section refuses.
         */
        [[nodiscard]] Result<std::optional<RateSchedule>> read_rate_schedule(
            const Config::Value& schedule, const Config::Value& optimizer_attributes) noexcept;

    } // namespace Optim
} // namespace SushiAI

/**************************************************************************/
/* failure_log_category.hpp                                               */
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
 * @file failure_log_category.hpp
 * @brief Declares K_FAILURE_LOG_CATEGORY, the log category of the module failure.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiLog/log/log_category.hpp>
#include <SushiAI/log/sushiai_log_project.hpp>

namespace SushiAI
{
    /** @brief Names the records of the module failure, written as [sushiai.failure]. */
    inline constexpr SushiLog::LogCategory K_FAILURE_LOG_CATEGORY =
        SushiLog::make_log_category(K_SUSHIAI_LOG_PROJECT, "failure");
} // namespace SushiAI

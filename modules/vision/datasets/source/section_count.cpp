/**************************************************************************/
/* section_count.cpp                                                      */
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
 * @file section_count.cpp
 * @brief Implements the reader of a whole number of a data section into a 32-bit count.
 * @author Mustafa Garip
 */

#include "section_count.hpp"

#include <limits>
#include <string>

#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Datasets::Detail
{
    Result<std::int32_t> read_section_count(const Config::Value& value, std::string_view key,
                                            std::string_view context, std::int32_t lowest)
    {
        constexpr std::int64_t K_HIGHEST = std::numeric_limits<std::int32_t>::max();

        const std::string named = std::string(context) + " '" + std::string(key) + "'";
        SA_TRY_ASSIGN(const std::int64_t read, value.as_int(named));

        SA_FAIL_IF(read < lowest || read > K_HIGHEST, StatusCode::InvalidDescription,
                   "{} is {}; it must be between {} and {}", named, read, lowest, K_HIGHEST);

        return static_cast<std::int32_t>(read);
    }
} // namespace SushiAI::Vision::Datasets::Detail

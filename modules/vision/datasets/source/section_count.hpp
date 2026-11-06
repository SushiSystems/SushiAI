/**************************************************************************/
/* section_count.hpp                                                      */
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
 * @file section_count.hpp
 * @brief Declares the reader of a whole number of a data section into a 32-bit count.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI::Vision::Datasets::Detail
{
    /**
     * @brief Reads @p value as a whole number in [@p lowest, the largest int32].
     * @param key Names the value in a message, after @p context.
     * @return The number; InvalidDescription, naming the bounds, for one outside them.
     *         The failure of Config::Value::as_int when @p value is not a whole number.
     */
    [[nodiscard]] Result<std::int32_t> read_section_count(const Config::Value& value,
                                                          std::string_view key,
                                                          std::string_view context,
                                                          std::int32_t lowest);
} // namespace SushiAI::Vision::Datasets::Detail

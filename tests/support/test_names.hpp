/**************************************************************************/
/* test_names.hpp                                                         */
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
 * @file test_names.hpp
 * @brief Declares the function that spells a row name as a gtest parameter name.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace SushiAI::Test
{
    /** @brief Spells @p text as a test-parameter name: anything but a letter or digit becomes `_`. */
    inline std::string parameter_name(std::string_view text)
    {
        std::string name(text);
        std::replace_if(name.begin(), name.end(),
                        [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
        return name;
    }
} // namespace SushiAI::Test

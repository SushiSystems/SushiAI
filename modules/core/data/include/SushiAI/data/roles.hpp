/**************************************************************************/
/* roles.hpp                                                              */
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
 * @file roles.hpp
 * @brief Declares the names of the target roles more than one module reads.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>

namespace SushiAI
{
    namespace Data
    {
        /** @brief Names the role that holds one class index per sample. */
        inline constexpr std::string_view K_ROLE_LABELS = "labels";

        /** @brief Names the role that holds four box coordinates per object slot. */
        inline constexpr std::string_view K_ROLE_BOXES = "boxes";

        /** @brief Names the role that holds one class index per object slot. */
        inline constexpr std::string_view K_ROLE_CLASSES = "classes";

        /** @brief Names the role that holds 1 for an object slot in use and 0 for padding. */
        inline constexpr std::string_view K_ROLE_VALID = "valid";

        /** @brief Names the role that holds one class index per pixel. */
        inline constexpr std::string_view K_ROLE_MASK = "mask";

        /** @brief Names the role that holds the real numbers a sample regresses to. */
        inline constexpr std::string_view K_ROLE_VALUES = "values";

    } // namespace Data
} // namespace SushiAI

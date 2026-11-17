/**************************************************************************/
/* caffe_v1_types.hpp                                                     */
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
 * @file caffe_v1_types.hpp
 * @brief Declares the table that names the layer types of Caffe's V1 layer message.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace SushiAI
{
    namespace Interchange
    {
        namespace Detail
        {
            /** @brief Names one V1 layer type: its number, its enumerator and its V2 name. */
            struct CaffeV1Type
            {
                std::int64_t number;         /**< The value of V1LayerParameter.LayerType. */
                std::string_view identifier; /**< The enumerator the prototxt spells. */
                std::string_view type_name;  /**< The string the V2 message gives the layer. */
            };

            /** @brief Returns the V1 type numbered @p number, or null. */
            [[nodiscard]] const CaffeV1Type* find_v1_type_by_number(std::int64_t number) noexcept;

            /** @brief Returns the V1 type spelled @p identifier, or null. */
            [[nodiscard]] const CaffeV1Type*
            find_v1_type_by_identifier(std::string_view identifier) noexcept;
        } // namespace Detail
    } // namespace Interchange
} // namespace SushiAI

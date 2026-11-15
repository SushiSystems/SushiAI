/**************************************************************************/
/* pickle_data.hpp                                                        */
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
 * @file pickle_data.hpp
 * @brief Declares the conversion of the plain data of a pickle tree into a configuration value.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/pickle_reader.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Converts None, a bool, a number, a text, a list, a tuple or a dict of @p value
         *        into the configuration value of the same shape.
         *
         * A tuple becomes an array. A dict key that is a whole number becomes its decimal text.
         * @return InvalidDescription naming @p context and the path of a node of another kind,
         *         or of a dict key that is neither a text nor a whole number.
         */
        [[nodiscard]] Result<Config::Value> pickle_to_value(const PickleValue& value,
                                                            std::string_view context) noexcept;
    } // namespace IO
} // namespace SushiAI

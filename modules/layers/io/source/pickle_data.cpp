/**************************************************************************/
/* pickle_data.cpp                                                        */
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
 * @file pickle_data.cpp
 * @brief Defines the conversion of plain pickle data into a configuration value.
 * @author Mustafa Garip
 */

#include <SushiAI/io/pickle_data.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/pickle_reader.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the deepest nesting the conversion follows. */
            constexpr std::size_t K_MAX_DEPTH = 64;

            /** @brief Converts @p value found at @p path, @p depth levels down. */
            Result<Config::Value> convert(const PickleValue& value, const std::string& path,
                                          std::string_view context, std::size_t depth)
            {
                SA_FAIL_IF(depth > K_MAX_DEPTH, StatusCode::InvalidDescription,
                           "{}: the data at '{}' nests deeper than {} levels", context, path,
                           K_MAX_DEPTH);
                switch (value.kind)
                {
                    case PickleKind::None:
                        return Config::Value();
                    case PickleKind::Bool:
                        return Config::Value::boolean(value.boolean);
                    case PickleKind::Int:
                        return Config::Value::integer(value.integer);
                    case PickleKind::Float:
                        return Config::Value::real(value.real);
                    case PickleKind::String:
                        return Config::Value::string(value.text);
                    case PickleKind::List:
                    case PickleKind::Tuple:
                    {
                        std::vector<Config::Value> entries;
                        entries.reserve(value.items.size());
                        for (std::size_t i = 0; i < value.items.size(); ++i)
                        {
                            SA_TRY_ASSIGN(Config::Value entry,
                                          convert(*value.items[i],
                                                  path + "[" + std::to_string(i) + "]", context,
                                                  depth + 1));
                            entries.push_back(std::move(entry));
                        }
                        return Config::Value::array(std::move(entries));
                    }
                    case PickleKind::Dict:
                    {
                        std::vector<std::string> keys;
                        std::vector<Config::Value> values;
                        for (const auto& entry : value.entries)
                        {
                            const PickleValue& key = *entry.first;
                            SA_FAIL_IF(key.kind != PickleKind::String &&
                                           key.kind != PickleKind::Int,
                                       StatusCode::InvalidDescription,
                                       "{}: the dict at '{}' has a key that is neither a text "
                                       "nor a whole number",
                                       context, path);
                            keys.push_back(key.kind == PickleKind::String
                                               ? key.text
                                               : std::to_string(key.integer));
                            SA_TRY_ASSIGN(Config::Value converted,
                                          convert(*entry.second, path + "." + keys.back(),
                                                  context, depth + 1));
                            values.push_back(std::move(converted));
                        }
                        return Config::Value::object(std::move(keys), std::move(values));
                    }
                    default:
                        break;
                }
                SA_FAIL_IF(true, StatusCode::InvalidDescription,
                           "{}: the value at '{}' is not plain data", context, path);
                return Config::Value();
            }
        } // namespace

        Result<Config::Value> pickle_to_value(const PickleValue& value,
                                              std::string_view context) noexcept
        {
            return convert(value, std::string(), context, 0);
        }
    } // namespace IO
} // namespace SushiAI

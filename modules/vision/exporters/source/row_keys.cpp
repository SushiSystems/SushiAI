/**************************************************************************/
/* row_keys.cpp                                                           */
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
 * @file row_keys.cpp
 * @brief Defines the readers of a row's keys that every exporter shares.
 * @author Mustafa Garip
 */

#include "row_keys.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the entries of an [h, w] pair. */
                constexpr std::size_t K_PAIR_SIZE = 2;

                /** @brief Returns the context of a key of the row, for a diagnostic. */
                std::string describe_key(const RowKeys& keys, std::string_view key)
                {
                    std::string text = "row ";
                    if (keys.row.has_value())
                        text += std::to_string(*keys.row) + " ";
                    return text + "'" + keys.spec->type + "' key '" + std::string(key) + "'";
                }

            } // namespace

            Status reject_unknown_keys(const RowKeys& keys,
                                       SushiRuntime::span<const std::string_view> allowed)
            {
                const Config::Value& attributes = keys.spec->attributes;
                if (attributes.kind() != Config::Kind::Object)
                    return Status();
                for (const std::string& key : attributes.keys())
                {
                    bool known = false;
                    for (const std::string_view name : allowed)
                        known = known || name == key;
                    SA_FAIL_IF(!known, StatusCode::Unsupported,
                               "the key '{}' has no counterpart the {} importer reads back", key,
                               keys.importer);
                }
                return Status();
            }

            Result<std::int64_t> read_int_key(const RowKeys& keys, std::string_view key,
                                              std::int64_t fallback)
            {
                if (!keys.spec->attributes.has(key))
                    return fallback;
                SA_TRY_ASSIGN(const Config::Value& stated,
                              keys.spec->attributes.at(key, describe_key(keys, key)));
                return stated.as_int(describe_key(keys, key));
            }

            Result<bool> read_bool_key(const RowKeys& keys, std::string_view key, bool fallback)
            {
                if (!keys.spec->attributes.has(key))
                    return fallback;
                SA_TRY_ASSIGN(const Config::Value& stated,
                              keys.spec->attributes.at(key, describe_key(keys, key)));
                return stated.as_bool(describe_key(keys, key));
            }

            Result<double> read_real_key(const RowKeys& keys, std::string_view key,
                                         double fallback)
            {
                if (!keys.spec->attributes.has(key))
                    return fallback;
                SA_TRY_ASSIGN(const Config::Value& stated,
                              keys.spec->attributes.at(key, describe_key(keys, key)));
                return stated.as_real(describe_key(keys, key));
            }

            Result<std::string> read_text_key(const RowKeys& keys, std::string_view key,
                                              std::string_view fallback)
            {
                if (!keys.spec->attributes.has(key))
                    return std::string(fallback);
                SA_TRY_ASSIGN(const Config::Value& stated,
                              keys.spec->attributes.at(key, describe_key(keys, key)));
                SA_TRY_ASSIGN(const std::string& text, stated.as_string(describe_key(keys, key)));
                return text;
            }

            Result<std::vector<std::int64_t>> read_pair_key(const RowKeys& keys,
                                                            std::string_view key,
                                                            std::int64_t fallback)
            {
                if (!keys.spec->attributes.has(key))
                    return std::vector<std::int64_t>{fallback, fallback};
                const std::string context = describe_key(keys, key);
                SA_TRY_ASSIGN(const Config::Value& stated, keys.spec->attributes.at(key, context));
                if (stated.kind() == Config::Kind::Int)
                {
                    SA_TRY_ASSIGN(const std::int64_t both, stated.as_int(context));
                    return std::vector<std::int64_t>{both, both};
                }
                SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, stated.as_array(context));
                SA_FAIL_IF(entries.size() != K_PAIR_SIZE, StatusCode::InvalidDescription,
                           "{} lists {} entries, and a pair lists 2", context, entries.size());
                std::vector<std::int64_t> pair;
                for (const Config::Value& entry : entries)
                {
                    SA_TRY_ASSIGN(const std::int64_t value, entry.as_int(context));
                    pair.push_back(value);
                }
                return pair;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI

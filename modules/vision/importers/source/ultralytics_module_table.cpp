/**************************************************************************/
/* ultralytics_module_table.cpp                                           */
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
 * @file ultralytics_module_table.cpp
 * @brief Defines the table of Ultralytics module translators and the helpers they share.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the divisor Ultralytics rounds every scaled width up to. */
                constexpr std::int64_t K_CHANNEL_DIVISOR = 8;

                /** @brief Lists every module the translation reads, one translator each. */
                constexpr ModuleTableRow K_MODULE_TABLE[] = {
                    {"Conv", &translate_conv},
                    {"C3k2", &translate_c3k2},
                    {"SPPF", &translate_sppf},
                    {"C2PSA", &translate_c2psa},
                    {"nn.Upsample", &translate_upsample},
                    {"Concat", &translate_concat},
                    {"Detect", &translate_detect},
                };
            } // namespace

            const ModuleTableRow* find_module_translator(std::string_view module) noexcept
            {
                for (const ModuleTableRow& row : K_MODULE_TABLE)
                {
                    if (row.module == module)
                        return &row;
                }
                return nullptr;
            }

            Result<std::int64_t> source_channels(const ModuleEntry& entry,
                                                 const ModuleProgress& progress,
                                                 std::int64_t reference)
            {
                const std::int64_t row = reference < 0 ? entry.index + reference : reference;
                SA_FAIL_IF(row < -1 || row >= entry.index, StatusCode::InvalidDescription,
                           "entry {} ({}) reads {}, which names no earlier row", entry.index,
                           entry.module, reference);
                if (row == -1)
                    return progress.input_channels;
                return progress.channels[static_cast<std::size_t>(row)];
            }

            std::int64_t scaled_channels(const ModuleProgress& progress,
                                         std::int64_t stated) noexcept
            {
                const double capped =
                    static_cast<double>(std::min(stated, progress.scale.max_channels)) *
                    progress.scale.width;
                const double units = std::ceil(capped / static_cast<double>(K_CHANNEL_DIVISOR));
                return static_cast<std::int64_t>(units) * K_CHANNEL_DIVISOR;
            }

            std::int64_t scaled_repeats(const ModuleEntry& entry,
                                        const ModuleProgress& progress) noexcept
            {
                if (entry.repeats <= 1)
                    return entry.repeats;
                const double scaled =
                    std::nearbyint(static_cast<double>(entry.repeats) * progress.scale.depth);
                return std::max<std::int64_t>(static_cast<std::int64_t>(scaled), 1);
            }

            Result<std::int64_t> argument_int(const ModuleEntry& entry, std::size_t position,
                                              std::int64_t fallback)
            {
                if (position >= entry.args.size() || entry.args[position].is_null())
                    return fallback;
                SA_FAIL_IF(entry.args[position].kind() != Config::Kind::Int,
                           StatusCode::InvalidDescription,
                           "entry {} ({}): argument {} is not a whole number", entry.index,
                           entry.module, position);
                return entry.args[position].as_int(entry.module);
            }

            Result<double> argument_real(const ModuleEntry& entry, std::size_t position,
                                         double fallback)
            {
                if (position >= entry.args.size() || entry.args[position].is_null())
                    return fallback;
                SA_FAIL_IF(entry.args[position].kind() != Config::Kind::Int &&
                               entry.args[position].kind() != Config::Kind::Real,
                           StatusCode::InvalidDescription,
                           "entry {} ({}): argument {} is not a number", entry.index,
                           entry.module, position);
                return entry.args[position].as_real(entry.module);
            }

            Result<bool> argument_bool(const ModuleEntry& entry, std::size_t position,
                                       bool fallback)
            {
                if (position >= entry.args.size() || entry.args[position].is_null())
                    return fallback;
                SA_FAIL_IF(entry.args[position].kind() != Config::Kind::Bool,
                           StatusCode::InvalidDescription,
                           "entry {} ({}): argument {} is not a bool", entry.index, entry.module,
                           position);
                return entry.args[position].as_bool(entry.module);
            }

            Status reject_extra_arguments(const ModuleEntry& entry, std::size_t most)
            {
                SA_FAIL_IF(entry.args.size() > most, StatusCode::Unsupported,
                           "entry {} ({}) states {} arguments, and the translation reads at "
                           "most {}",
                           entry.index, entry.module, entry.args.size(), most);
                return Status();
            }

            void append_row(ModuleProgress& progress, const ModuleEntry& entry,
                            std::string_view type, std::vector<std::string> keys,
                            std::vector<Config::Value> values)
            {
                keys.insert(keys.begin(), "type");
                values.insert(values.begin(), Config::Value::string(std::string(type)));
                const bool reads_above = !entry.listed_from && entry.from.size() == 1 &&
                                         entry.from.front() == -1;
                if (!reads_above)
                {
                    std::vector<Config::Value> from;
                    for (const std::int64_t reference : entry.from)
                        from.push_back(Config::Value::integer(reference));
                    keys.insert(keys.begin() + 1, "from");
                    values.insert(values.begin() + 1, Config::Value::array(std::move(from)));
                }
                progress.layers.push_back(Config::Value::object(std::move(keys), std::move(values)));
                if (std::find(progress.types.begin(), progress.types.end(), type) ==
                    progress.types.end())
                    progress.types.emplace_back(type);
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

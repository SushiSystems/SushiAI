/**************************************************************************/
/* ultralytics_module_c3k2.cpp                                            */
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
 * @file ultralytics_module_c3k2.cpp
 * @brief Defines the translation of an Ultralytics C3k2 entry into a c3k2 row.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

#include <cstdint>
#include <string>
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
                /** @brief Returns the inner block a c3k2 row names for the two switches. */
                const char* inner_block(bool attention, bool c3k) noexcept
                {
                    if (attention)
                        return "bottleneck_psa";
                    return c3k ? "c3k" : "bottleneck";
                }
            } // namespace

            Status translate_c3k2(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 6));
                SA_TRY_ASSIGN(const std::int64_t stated, argument_int(entry, 0, 0));
                SA_TRY_ASSIGN(const bool stated_c3k, argument_bool(entry, 1, false));
                SA_TRY_ASSIGN(const double expansion, argument_real(entry, 2, 0.5));
                SA_TRY_ASSIGN(const bool attention, argument_bool(entry, 3, false));
                SA_TRY_ASSIGN(const std::int64_t groups, argument_int(entry, 4, 1));
                SA_TRY_ASSIGN(const bool shortcut, argument_bool(entry, 5, true));
                SA_FAIL_IF(groups != 1, StatusCode::Unsupported,
                           "entry {} (C3k2) states {} groups, and only 1 is translated",
                           entry.index, groups);

                // Ultralytics turns the inner block into C3k for the m, l and x scales.
                const std::string& scale = progress.scale.name;
                const bool c3k = stated_c3k || scale == "m" || scale == "l" || scale == "x";
                progress.legacy = false;

                const std::int64_t filters = scaled_channels(progress, stated);
                const std::int64_t hidden =
                    static_cast<std::int64_t>(static_cast<double>(filters) * expansion);
                append_row(progress, entry, "c3k2",
                           {"filters", "hidden", "repeats", "inner", "shortcut"},
                           {Config::Value::integer(filters), Config::Value::integer(hidden),
                            Config::Value::integer(scaled_repeats(entry, progress)),
                            Config::Value::string(inner_block(attention, c3k)),
                            Config::Value::boolean(shortcut)});
                progress.channels.push_back(filters);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

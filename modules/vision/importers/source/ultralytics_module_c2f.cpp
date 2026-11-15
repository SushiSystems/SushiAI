/**************************************************************************/
/* ultralytics_module_c2f.cpp                                             */
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
 * @file ultralytics_module_c2f.cpp
 * @brief Defines the translation of an Ultralytics C2f entry into a c2f row.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

#include <cstdint>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            Status translate_c2f(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 4));
                SA_TRY_ASSIGN(const std::int64_t stated, argument_int(entry, 0, 0));
                SA_TRY_ASSIGN(const bool shortcut, argument_bool(entry, 1, false));
                SA_TRY_ASSIGN(const std::int64_t groups, argument_int(entry, 2, 1));
                SA_TRY_ASSIGN(const double expansion, argument_real(entry, 3, 0.5));
                SA_FAIL_IF(groups != 1, StatusCode::Unsupported,
                           "entry {} (C2f) states {} groups, and only 1 is translated",
                           entry.index, groups);

                const std::int64_t filters = scaled_channels(progress, stated);
                const std::int64_t hidden =
                    static_cast<std::int64_t>(static_cast<double>(filters) * expansion);
                append_row(progress, entry, "c2f", {"filters", "hidden", "repeats", "shortcut"},
                           {Config::Value::integer(filters), Config::Value::integer(hidden),
                            Config::Value::integer(scaled_repeats(entry, progress)),
                            Config::Value::boolean(shortcut)});
                progress.channels.push_back(filters);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

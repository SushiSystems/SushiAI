/**************************************************************************/
/* ultralytics_module_sppf.cpp                                            */
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
 * @file ultralytics_module_sppf.cpp
 * @brief Defines the translation of an Ultralytics SPPF entry into an sppf row.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

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
            Status translate_sppf(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 4));
                SA_TRY_ASSIGN(const std::int64_t stated, argument_int(entry, 0, 0));
                SA_TRY_ASSIGN(const std::int64_t kernel, argument_int(entry, 1, 5));
                SA_TRY_ASSIGN(const std::int64_t pools, argument_int(entry, 2, 3));
                SA_TRY_ASSIGN(const bool shortcut, argument_bool(entry, 3, false));

                // Ultralytics' source builds cv1 linear; checkpoints before it pickled a SiLU.
                const std::string path = std::to_string(entry.index) + ".cv1.act";
                const std::string_view activation = find_module_class(progress, path);
                SA_FAIL_IF(!activation.empty() && activation != "SiLU" &&
                               activation != "Identity",
                           StatusCode::Unsupported,
                           "entry {} (SPPF) pickles the activation {} at '{}', and only SiLU and "
                           "Identity are translated",
                           entry.index, activation, path);

                const std::int64_t filters = scaled_channels(progress, stated);
                std::vector<std::string> keys = {"filters", "kernel", "pools", "shortcut"};
                std::vector<Config::Value> values = {
                    Config::Value::integer(filters), Config::Value::integer(kernel),
                    Config::Value::integer(pools), Config::Value::boolean(shortcut)};
                if (activation == "SiLU")
                {
                    keys.emplace_back("activate_narrowing");
                    values.push_back(Config::Value::boolean(true));
                }
                append_row(progress, entry, "sppf", std::move(keys), std::move(values));
                progress.channels.push_back(filters);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

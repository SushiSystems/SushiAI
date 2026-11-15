/**************************************************************************/
/* ultralytics_module_conv.cpp                                            */
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
 * @file ultralytics_module_conv.cpp
 * @brief Defines the translation of an Ultralytics Conv entry into a conv row.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

#include <cstdint>
#include <string>
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
            Status translate_conv(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 4));
                SA_TRY_ASSIGN(const std::int64_t stated, argument_int(entry, 0, 0));
                SA_TRY_ASSIGN(const std::int64_t kernel, argument_int(entry, 1, 1));
                SA_TRY_ASSIGN(const std::int64_t stride, argument_int(entry, 2, 1));
                SA_TRY_ASSIGN(const std::int64_t padding, argument_int(entry, 3, -1));
                SA_TRY_ASSIGN(const std::int64_t input, source_channels(entry, progress, -1));
                static_cast<void>(input);

                const std::int64_t filters = scaled_channels(progress, stated);
                std::vector<std::string> keys = {"filters", "kernel", "stride"};
                std::vector<Config::Value> values = {Config::Value::integer(filters),
                                                     Config::Value::integer(kernel),
                                                     Config::Value::integer(stride)};
                // A padding the file states replaces the half kernel the conv row takes.
                if (padding >= 0)
                {
                    keys.emplace_back("padding");
                    values.push_back(Config::Value::integer(padding));
                }
                append_row(progress, entry, "conv", std::move(keys), std::move(values));
                progress.channels.push_back(filters);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

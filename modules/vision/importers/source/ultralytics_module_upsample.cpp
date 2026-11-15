/**************************************************************************/
/* ultralytics_module_upsample.cpp                                        */
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
 * @file ultralytics_module_upsample.cpp
 * @brief Defines the translation of an Ultralytics nn.Upsample entry into an upsample row.
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
                /** @brief Holds the only interpolation mode the translation reads. */
                constexpr const char* K_NEAREST = "nearest";

                /** @brief Returns whether @p entry states no mode or the nearest one. */
                bool is_nearest(const ModuleEntry& entry)
                {
                    if (entry.args.size() < 3)
                        return true;
                    const Config::Value& mode = entry.args[2];
                    if (mode.kind() != Config::Kind::String)
                        return false;
                    const Result<const std::string&> text = mode.as_string("mode");
                    return text.has_value() && text.value() == K_NEAREST;
                }
            } // namespace

            Status translate_upsample(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 3));
                SA_FAIL_IF(!entry.args.empty() && !entry.args[0].is_null(),
                           StatusCode::Unsupported,
                           "entry {} (nn.Upsample) states an output size, and only a scale is "
                           "translated",
                           entry.index);
                SA_FAIL_IF(!is_nearest(entry), StatusCode::Unsupported,
                           "entry {} (nn.Upsample) states a mode other than {}", entry.index,
                           K_NEAREST);
                SA_TRY_ASSIGN(const std::int64_t scale, argument_int(entry, 1, 1));
                SA_TRY_ASSIGN(const std::int64_t channels, source_channels(entry, progress, -1));

                append_row(progress, entry, "upsample", {"scale"}, {Config::Value::integer(scale)});
                progress.channels.push_back(channels);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

/**************************************************************************/
/* ultralytics_module_detect.cpp                                          */
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
 * @file ultralytics_module_detect.cpp
 * @brief Defines the translation of an Ultralytics Detect entry into a detect row.
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
            namespace
            {
                /** @brief Holds the box bins a detect row takes when it states none. */
                constexpr std::int64_t K_ROW_REG_MAX = 1;
            } // namespace

            Status translate_detect(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 1));
                SA_TRY_ASSIGN(const std::int64_t classes,
                              argument_int(entry, 0, progress.classes));
                SA_FAIL_IF(classes < 1, StatusCode::InvalidDescription,
                           "entry {} (Detect) states {} classes", entry.index, classes);
                for (const std::int64_t reference : entry.from)
                {
                    SA_TRY_ASSIGN(const std::int64_t part,
                                  source_channels(entry, progress, reference));
                    static_cast<void>(part);
                }

                std::vector<std::string> keys = {"classes"};
                std::vector<Config::Value> values = {Config::Value::integer(classes)};
                if (progress.reg_max != K_ROW_REG_MAX)
                {
                    keys.emplace_back("reg_max");
                    values.push_back(Config::Value::integer(progress.reg_max));
                }
                if (!progress.end2end)
                {
                    keys.emplace_back("end2end");
                    values.push_back(Config::Value::boolean(false));
                }
                if (progress.legacy)
                {
                    keys.emplace_back("class_branch");
                    values.push_back(Config::Value::string("plain"));
                }
                append_row(progress, entry, "detect", std::move(keys), std::move(values));
                progress.channels.push_back(0);
                progress.head = entry.index;
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI

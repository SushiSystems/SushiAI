/**************************************************************************/
/* darknet_row_table.cpp                                                  */
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
 * @file darknet_row_table.cpp
 * @brief Defines the table of the rows the Darknet exporter writes.
 * @author Mustafa Garip
 */

#include "darknet_row_table.hpp"

#include <string_view>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every row type that opens a section the importer reads back. */
                constexpr DarknetRowEntry K_DARKNET_ROWS[] = {
                    {"conv2d", &write_darknet_conv2d},
                    {"maxpool2d", &write_darknet_maxpool2d},
                    {"upsample", &write_darknet_upsample},
                    {"concat", &write_darknet_concat},
                    {"slice_channels", &write_darknet_slice_channels},
                    {"add", &write_darknet_add},
                };
            } // namespace

            const DarknetRowEntry* find_darknet_row(std::string_view type) noexcept
            {
                for (const DarknetRowEntry& entry : K_DARKNET_ROWS)
                {
                    if (entry.type == type)
                        return &entry;
                }
                return nullptr;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI

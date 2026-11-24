/**************************************************************************/
/* darknet_row_table.hpp                                                  */
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
 * @file darknet_row_table.hpp
 * @brief Declares the table of the rows the Darknet exporter writes, one writer per row type.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/status.hpp>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /**
             * @brief Writes one row as .cfg sections and records its value.
             * @return Unsupported, with a reason that does not name the row, when the row
             *         states what the Darknet importer cannot read back; any other code stops
             *         the export.
             */
            using DarknetRowWriter = Status (*)(const DarknetRowCall& call,
                                                DarknetExportProgress& progress);

            /** @brief Holds one row of the table: a row type and its writer. */
            struct DarknetRowEntry
            {
                std::string_view type;  /**< The row type, such as "conv2d". */
                DarknetRowWriter write; /**< What writes a row of the type. */
            };

            /** @brief Returns the entry of row type @p type, or null when Darknet refuses it. */
            [[nodiscard]] const DarknetRowEntry* find_darknet_row(std::string_view type) noexcept;

            /**
             * @brief Writes a "conv2d" row, with the batchnorm2d and the activation rows that
             *        follow it and read it alone, as one [convolutional] section.
             */
            [[nodiscard]] Status write_darknet_conv2d(const DarknetRowCall& call,
                                                      DarknetExportProgress& progress);

            /** @brief Writes a "maxpool2d" row as a [maxpool] section. */
            [[nodiscard]] Status write_darknet_maxpool2d(const DarknetRowCall& call,
                                                         DarknetExportProgress& progress);

            /** @brief Writes an "upsample" row as an [upsample] section. */
            [[nodiscard]] Status write_darknet_upsample(const DarknetRowCall& call,
                                                        DarknetExportProgress& progress);

            /** @brief Writes a "concat" row as a [route] over two layers or more. */
            [[nodiscard]] Status write_darknet_concat(const DarknetRowCall& call,
                                                      DarknetExportProgress& progress);

            /** @brief Writes a "slice_channels" row as a [route] with groups and group_id. */
            [[nodiscard]] Status write_darknet_slice_channels(const DarknetRowCall& call,
                                                              DarknetExportProgress& progress);

            /** @brief Writes an "add" row as a linear [shortcut]. */
            [[nodiscard]] Status write_darknet_add(const DarknetRowCall& call,
                                                   DarknetExportProgress& progress);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
